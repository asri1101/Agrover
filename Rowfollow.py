#!/usr/bin/env python3
"""
WAVE_ROVER Row-Following (Pi Camera/USB cam) + IMU yaw-damping + serial JSON motor commands

You said:
- You can already drive the rover by sending JSON over serial, e.g. {"T":1,"L":0.2,"R":0.2}
- IMU data is “already accessible” (Waveshare can stream chassis feedback incl. IMU)
So this script:
1) Opens the serial port (115200) and starts a read thread that parses JSON feedback
2) Enables continuous chassis feedback (so we can read IMU continuously)
3) Reads camera frames (Picamera2 OR OpenCV VideoCapture)
4) Computes row center error using ExG vegetation segmentation
5) Computes steering via PID + yaw-rate damping from IMU
6) Sends {"T":1,"L":..., "R":...}\n continuously

Docs used:
- CMD_SPEED_CTRL {"T":1,"L":...,"R":...} (Wave Rover uses -0.5..+0.5 speed range) :contentReference[oaicite:0]{index=0}
- IMU fetch/feedback commands {"T":126}, {"T":130}, {"T":131,"cmd":1} :contentReference[oaicite:1]{index=1}

Install:
  pip3 install pyserial numpy opencv-python
  sudo apt install -y python3-picamera2   # if using Pi camera module

Run:
  python3 wave_rover_row_follow.py /dev/ttyUSB0
  # or /dev/serial0 depending on your wiring

Controls:
  q = quit
"""

import argparse
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import cv2
import serial

# --- Optional Pi camera (Picamera2) ---
PICAMERA2_AVAILABLE = True
try:
    from picamera2 import Picamera2
except Exception:
    PICAMERA2_AVAILABLE = False


# =========================
# CONFIG (tune these)
# =========================
@dataclass
class Config:
    # Motor command limits (Wave Rover wiki says -0.5..+0.5 typical) :contentReference[oaicite:2]{index=2}
    V_MAX: float = 0.50         # absolute max wheel command
    BASE_SPEED: float = 0.20    # forward speed while tracking (m/s-ish in docs; Wave Rover maps to PWM)

    # Vision
    CAM_W: int = 640
    CAM_H: int = 480
    ROI_Y_START_FRAC: float = 0.60    # use bottom 40%
    EXG_THRESH: int = 40             # tune for your lighting (20–70 typical)
    MORPH_K: int = 5                 # morphology kernel size
    MORPH_ITERS: int = 1
    HIST_SMOOTH_K: int = 21          # odd; histogram smoothing
    MIN_PEAK_SEP_FRAC: float = 0.20  # peaks must be separated by >= 20% of image width

    # Control
    KP: float = 0.90   # NOTE: error is normalized; start here, tune carefully
    KI: float = 0.00
    KD: float = 0.20
    MAX_STEER: float = 0.25          # clamp steering term (in wheel-speed units)

    # IMU damping (uses yaw_rate if available; otherwise estimates from yaw)
    KYAW: float = 0.08               # subtract KYAW * yaw_rate(rad/s)
    YAW_RATE_FALLBACK_SMOOTH: float = 0.6  # EMA smoothing if using fallback

    # Serial / Waveshare feedback
    ENABLE_CONTINUOUS_FEEDBACK: bool = True
    FEEDBACK_INTERVAL_MS: int = 50   # optional; only applied if supported
    SERIAL_TIMEOUT_S: float = 0.2

    # Camera backend
    CAMERA_BACKEND: str = "picamera2"   # "picamera2" or "opencv"
    OPENCV_CAM_INDEX: int = 0           # used if CAMERA_BACKEND="opencv"

    # UI
    SHOW_WINDOWS: bool = True
    PRINT_DEBUG_HZ: float = 2.0         # print status ~2 Hz


CFG = Config()


# =========================
# PID
# =========================
class PID:
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error: float, dt: float) -> float:
        if dt <= 1e-6:
            return 0.0
        self.integral += error * dt
        d = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * d


# =========================
# Serial: Waveshare JSON I/O
# =========================
class WaveRoverSerial:
    def __init__(self, port: str, baud: int = 115200, timeout: float = 0.2):
        self.ser = serial.Serial(port, baudrate=baud, timeout=timeout, dsrdtr=None)
        # match your snippet
        self.ser.setRTS(False)
        self.ser.setDTR(False)

        self._lock = threading.Lock()
        self.latest_msg: Optional[Dict[str, Any]] = None
        self.latest_imu: Optional[Dict[str, Any]] = None

        # yaw + yaw_rate derived fallback
        self._yaw_rad: Optional[float] = None
        self._yaw_rate_rad_s: float = 0.0
        self._yaw_prev_t: Optional[float] = None

        self._stop = False
        self._thread = threading.Thread(target=self._read_loop, daemon=True)

    def start(self):
        self._thread.start()

    def close(self):
        self._stop = True
        try:
            time.sleep(0.05)
        except Exception:
            pass
        try:
            self.ser.close()
        except Exception:
            pass

    def send_json(self, obj: Dict[str, Any]):
        msg = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
        with self._lock:
            self.ser.write(msg)

    def set_wheels(self, left: float, right: float):
        left = float(np.clip(left, -CFG.V_MAX, CFG.V_MAX))
        right = float(np.clip(right, -CFG.V_MAX, CFG.V_MAX))
        # CMD_SPEED_CTRL is {"T":1,"L":...,"R":...} :contentReference[oaicite:3]{index=3}
        self.send_json({"T": 1, "L": left, "R": right})

    def stop_motors(self):
        self.set_wheels(0.0, 0.0)

    def enable_feedback(self):
        # Continuous chassis feedback: {"T":131,"cmd":1} :contentReference[oaicite:4]{index=4}
        if CFG.ENABLE_CONTINUOUS_FEEDBACK:
            self.send_json({"T": 131, "cmd": 1})
            # Some firmwares support interval setting (documented on the JSON command page) :contentReference[oaicite:5]{index=5}
            if CFG.FEEDBACK_INTERVAL_MS > 0:
                self.send_json({"T": 142, "cmd": int(CFG.FEEDBACK_INTERVAL_MS)})

    def _read_loop(self):
        while not self._stop:
            try:
                line = self.ser.readline()
                if not line:
                    continue
                s = line.decode("utf-8", errors="ignore").strip()
                if not s:
                    continue

                # Many firmwares print extra text; only parse if JSON-ish
                if not (s.startswith("{") and s.endswith("}")):
                    continue

                msg = json.loads(s)
                if not isinstance(msg, dict):
                    continue

                with self._lock:
                    self.latest_msg = msg

                    # Heuristic: detect IMU-bearing messages
                    # Waveshare docs say IMU data is included in chassis feedback :contentReference[oaicite:6]{index=6}
                    if any(k.lower() in ("imu", "yaw", "pitch", "roll", "heading", "quat", "ax", "gx") for k in msg.keys()):
                        self.latest_imu = msg

                # Update derived yaw / yaw_rate if possible
                self._update_yaw_from_msg(msg)

            except Exception:
                continue

    def _update_yaw_from_msg(self, msg: Dict[str, Any]):
        """
        We don't assume exact field names because Waveshare feedback formats vary by firmware.
        We try common patterns:
          - yaw / heading (degrees or radians)
          - imu:{yaw:..., ...}
          - euler:{yaw:...}
        If yaw_rate is provided directly, we use it.
        Otherwise we compute yaw_rate by finite difference.
        """
        now = time.time()

        # 1) Try direct yaw_rate fields
        # Common guesses: "gz" (gyro z), "gyroZ", "yaw_rate", "wz", etc.
        for k in ("yaw_rate", "yawRate", "wz", "gz", "gyroZ", "gyro_z"):
            if k in msg and isinstance(msg[k], (int, float)):
                # We *assume* it's rad/s if small, deg/s if large; convert if needed.
                val = float(msg[k])
                if abs(val) > 6.5:  # likely deg/s
                    val = np.deg2rad(val)
                self._yaw_rate_rad_s = val
                return

        # 2) Find a yaw angle
        yaw = None

        def pick_yaw(d: Dict[str, Any]) -> Optional[float]:
            for kk in ("yaw", "heading", "Yaw", "Heading"):
                if kk in d and isinstance(d[kk], (int, float)):
                    return float(d[kk])
            return None

        yaw = pick_yaw(msg)
        if yaw is None:
            for container_key in ("imu", "IMU", "euler", "Euler", "att", "attitude"):
                if container_key in msg and isinstance(msg[container_key], dict):
                    yaw = pick_yaw(msg[container_key])
                    if yaw is not None:
                        break

        if yaw is None:
            return

        # Decide if yaw is degrees or radians
        yaw_rad = float(yaw)
        if abs(yaw_rad) > 6.5:  # > ~373 deg? actually > 2*pi => likely degrees
            yaw_rad = np.deg2rad(yaw_rad)

        # unwrap + yaw_rate from finite diff
        if self._yaw_rad is not None and self._yaw_prev_t is not None:
            dt = now - self._yaw_prev_t
            if dt > 1e-3:
                dy = np.arctan2(np.sin(yaw_rad - self._yaw_rad), np.cos(yaw_rad - self._yaw_rad))  # wrapped diff
                est = dy / dt
                # EMA smoothing
                a = CFG.YAW_RATE_FALLBACK_SMOOTH
                self._yaw_rate_rad_s = a * self._yaw_rate_rad_s + (1 - a) * est

        self._yaw_rad = yaw_rad
        self._yaw_prev_t = now

    def get_yaw_rate(self) -> float:
        return float(self._yaw_rate_rad_s)


# =========================
# Vision: ExG + corridor center
# =========================
def exg_mask(bgr: np.ndarray, thresh: int) -> np.ndarray:
    B, G, R = cv2.split(bgr)
    exg = (2 * G.astype(np.int16) - R.astype(np.int16) - B.astype(np.int16))
    exg = np.clip(exg, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(exg, thresh, 255, cv2.THRESH_BINARY)
    return mask

def morph_cleanup(mask: np.ndarray, k: int, iters: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)
    return mask

def smooth_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    return np.convolve(x, np.ones(k, dtype=np.float32) / k, mode="same")

def find_corridor_center(roi_mask: np.ndarray, w_full: int) -> Tuple[Optional[int], Tuple[int, int], np.ndarray]:
    hist = np.sum(roi_mask > 0, axis=0).astype(np.float32)
    hist = smooth_1d(hist, CFG.HIST_SMOOTH_K)

    w = hist.shape[0]
    half = w // 2
    lp = int(np.argmax(hist[:half]))
    rp = int(np.argmax(hist[half:]) + half)

    min_sep = int(w_full * CFG.MIN_PEAK_SEP_FRAC)
    if (rp - lp) < min_sep:
        return None, (lp, rp), hist

    center = (lp + rp) // 2
    return center, (lp, rp), hist


# =========================
# Camera backends
# =========================
class Camera:
    def __init__(self):
        self.backend = CFG.CAMERA_BACKEND.lower()

        if self.backend == "picamera2":
            if not PICAMERA2_AVAILABLE:
                raise RuntimeError("Picamera2 not available. Set CAMERA_BACKEND='opencv' or install python3-picamera2.")
            self.picam2 = Picamera2()
            cfg = self.picam2.create_video_configuration(
                main={"format": "RGB888", "size": (CFG.CAM_W, CFG.CAM_H)},
                controls={"FrameRate": 30},
            )
            self.picam2.configure(cfg)
            self.picam2.start()
            time.sleep(0.2)
            self.cap = None
        elif self.backend == "opencv":
            self.cap = cv2.VideoCapture(CFG.OPENCV_CAM_INDEX)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.CAM_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.CAM_H)
            self.picam2 = None
            time.sleep(0.2)
        else:
            raise ValueError("CAMERA_BACKEND must be 'picamera2' or 'opencv'")

    def read_bgr(self) -> np.ndarray:
        if self.backend == "picamera2":
            rgb = self.picam2.capture_array()
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                raise RuntimeError("Failed to read from OpenCV camera")
            return frame

    def close(self):
        try:
            if self.picam2 is not None:
                self.picam2.stop()
        except Exception:
            pass
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass


# =========================
# Main loop
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=str, help="Serial port, e.g. /dev/ttyUSB0 or /dev/serial0")
    args = parser.parse_args()

    rover = WaveRoverSerial(args.port, baud=115200, timeout=CFG.SERIAL_TIMEOUT_S)
    rover.start()
    rover.enable_feedback()

    cam = Camera()
    pid = PID(CFG.KP, CFG.KI, CFG.KD)

    print("Running. Press 'q' to quit.")

    last_t = time.time()
    last_print = 0.0
    last_good_center = None

    try:
        while True:
            now = time.time()
            dt = now - last_t
            last_t = now

            frame = cam.read_bgr()
            h, w, _ = frame.shape
            y0 = int(h * CFG.ROI_Y_START_FRAC)
            roi = frame[y0:h, :]

            mask = exg_mask(roi, CFG.EXG_THRESH)
            mask = morph_cleanup(mask, CFG.MORPH_K, CFG.MORPH_ITERS)

            center, (lp, rp), _hist = find_corridor_center(mask, w)
            img_center = w // 2

            confident = True
            if center is None:
                confident = False
                if last_good_center is not None:
                    center = last_good_center
                else:
                    center = img_center
            else:
                last_good_center = center

            error_px = (img_center - center)
            error = error_px / float(w)  # normalized approx [-0.5, 0.5]

            steer = pid.update(error, dt)

            # IMU yaw damping (from chassis feedback)
            yaw_rate = rover.get_yaw_rate()
            steer = steer - CFG.KYAW * yaw_rate

            # clamp steer into wheel-speed space
            steer = float(np.clip(steer, -CFG.MAX_STEER, CFG.MAX_STEER))

            # optional slow-down if low confidence
            base = CFG.BASE_SPEED * (0.6 if not confident else 1.0)

            left = base - steer
            right = base + steer

            rover.set_wheels(left, right)

            # UI
            if CFG.SHOW_WINDOWS:
                vis = frame.copy()
                cv2.rectangle(vis, (0, y0), (w - 1, h - 1), (255, 255, 255), 1)
                cv2.line(vis, (img_center, y0), (img_center, h - 1), (255, 255, 255), 1)
                cv2.line(vis, (center, y0), (center, h - 1), (0, 255, 0) if confident else (0, 255, 255), 2)
                cv2.circle(vis, (lp, y0 + (h - y0) // 2), 6, (255, 255, 0), -1)
                cv2.circle(vis, (rp, y0 + (h - y0) // 2), 6, (255, 255, 0), -1)

                txt = f"err_px={error_px:+4d} steer={steer:+.3f} L={left:+.2f} R={right:+.2f} yawRate={yaw_rate:+.2f}"
                if not confident:
                    txt += " LOW_CONF"
                cv2.putText(vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow("WAVE_ROVER RowFollow", vis)
                cv2.imshow("mask(ROI)", mask)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            # debug prints
            if CFG.PRINT_DEBUG_HZ > 0 and (now - last_print) > (1.0 / CFG.PRINT_DEBUG_HZ):
                last_print = now
                print(f"[dbg] err_px={error_px:+4d} steer={steer:+.3f} L={left:+.2f} R={right:+.2f} yawRate={yaw_rate:+.2f}")

    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping motors...")
        try:
            rover.stop_motors()
        except Exception:
            pass
        rover.close()
        cam.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()