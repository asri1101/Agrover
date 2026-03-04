#!/usr/bin/env python3
"""
WAVE_ROVER Row-Following (rpicam-vid MJPEG) + IMU yaw-rate damping + serial JSON motor commands

Fixes based on your screenshot:
- IMU packet fields are: r, p, y (roll, pitch, yaw) and gx, gy, gz (gyro rates)
- You can request IMU packets with: {"T":126}
So we:
1) Add rover.request_imu() and poll it at ~20 Hz
2) Parse yaw rate from gz (preferred) and yaw angle from y (degrees -> rad)
3) Remove the unused/broken Camera class (Picamera2/OpenCV VideoCapture path)
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
import subprocess


class RpiCamMJPEG:
    def __init__(self, width=640, height=480, fps=30):
        self.proc = subprocess.Popen(
            [
                "rpicam-vid",
                "-t", "0",
                "--width", str(width),
                "--height", str(height),
                "--framerate", str(fps),
                "--codec", "mjpeg",
                "-o", "-"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0
        )
        if self.proc.stdout is None:
            raise RuntimeError("Failed to start rpicam-vid.")
        self.buf = bytearray()

    def read_bgr(self) -> np.ndarray:
        # Find complete JPEG frames (FFD8 ... FFD9)
        while True:
            chunk = self.proc.stdout.read(4096)
            if not chunk:
                raise RuntimeError("rpicam-vid stopped producing data.")
            self.buf.extend(chunk)

            start = self.buf.find(b"\xff\xd8")
            end = self.buf.find(b"\xff\xd9")
            if start != -1 and end != -1 and end > start:
                jpg = self.buf[start:end + 2]
                del self.buf[:end + 2]

                arr = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame

    def close(self):
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
        except Exception:
            pass


# =========================
# CONFIG (tune these)
# =========================
@dataclass
class Config:
    # Motor command limits (Wave Rover typical -0.5..+0.5)
    V_MAX: float = 0.50
    BASE_SPEED: float = 0.20

    # Vision
    CAM_W: int = 640
    CAM_H: int = 480
    ROI_Y_START_FRAC: float = 0.60    # use bottom 40%
    EXG_THRESH: int = 40
    MORPH_K: int = 5
    MORPH_ITERS: int = 1
    HIST_SMOOTH_K: int = 21          # odd
    MIN_PEAK_SEP_FRAC: float = 0.20  # peaks must be separated by >= 20% of image width

    # Control
    KP: float = 0.90
    KI: float = 0.00
    KD: float = 0.20
    MAX_STEER: float = 0.25

    # IMU damping
    KYAW: float = 0.08               # subtract KYAW * yaw_rate(rad/s)
    YAW_RATE_FALLBACK_SMOOTH: float = 0.6  # EMA if gz not present

    # Serial
    SERIAL_TIMEOUT_S: float = 0.2

    # UI
    SHOW_WINDOWS: bool = True
    PRINT_DEBUG_HZ: float = 2.0

    # IMU polling
    IMU_POLL_HZ: float = 20.0        # how often to send {"T":126}


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
        self.ser.setRTS(False)
        self.ser.setDTR(False)

        self._lock = threading.Lock()
        self.latest_msg: Optional[Dict[str, Any]] = None

        # derived yaw + yaw_rate
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
        self.send_json({"T": 1, "L": left, "R": right})

    def stop_motors(self):
        self.set_wheels(0.0, 0.0)

    def request_imu(self):
        # Per your screenshot: {"T":126} returns r,p,y,gx,gy,gz,...
        self.send_json({"T": 126})

    def _read_loop(self):
        while not self._stop:
            try:
                line = self.ser.readline()
                if not line:
                    continue
                s = line.decode("utf-8", errors="ignore").strip()
                if not s:
                    continue
                if not (s.startswith("{") and s.endswith("}")):
                    continue

                msg = json.loads(s)
                if not isinstance(msg, dict):
                    continue

                with self._lock:
                    self.latest_msg = msg

                self._update_yaw_from_msg(msg)

            except Exception:
                continue

    def _update_yaw_from_msg(self, msg: Dict[str, Any]):
        """
        Screenshot format example:
          {"T":1002,"r":...,"p":...,"y":-111.79,"gx":...,"gy":...,"gz":...}

        - yaw angle: msg["y"]  (very likely degrees)
        - yaw rate:  msg["gz"] (likely rad/s if small, deg/s if large)
        """
        now = time.time()

        # Prefer gyro z as yaw rate
        has_gz = False
        if "gz" in msg and isinstance(msg["gz"], (int, float)):
            val = float(msg["gz"])
            if abs(val) > 6.5:  # likely deg/s
                val = np.deg2rad(val)
            self._yaw_rate_rad_s = val
            has_gz = True

        # Yaw angle for potential fallback / unwrapping
        if "y" not in msg or not isinstance(msg["y"], (int, float)):
            return

        yaw_val = float(msg["y"])
        yaw_rad = yaw_val
        if abs(yaw_rad) > 6.5:  # likely degrees
            yaw_rad = np.deg2rad(yaw_rad)

        # If gz not present, estimate yaw_rate by finite diff on yaw angle
        if (not has_gz) and (self._yaw_rad is not None) and (self._yaw_prev_t is not None):
            dt = now - self._yaw_prev_t
            if dt > 1e-3:
                dy = np.arctan2(np.sin(yaw_rad - self._yaw_rad), np.cos(yaw_rad - self._yaw_rad))
                est = dy / dt
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
# Main loop
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=str, help="Serial port, e.g. /dev/ttyUSB0 or /dev/serial0")
    args = parser.parse_args()

    rover = WaveRoverSerial(args.port, baud=115200, timeout=CFG.SERIAL_TIMEOUT_S)
    rover.start()

    cam = RpiCamMJPEG(CFG.CAM_W, CFG.CAM_H, fps=30)
    pid = PID(CFG.KP, CFG.KI, CFG.KD)

    print("Running. Press 'q' to quit.")

    last_t = time.time()
    last_print = 0.0
    last_good_center = None

    next_imu_t = time.time()

    try:
        while True:
            now = time.time()
            dt = now - last_t
            last_t = now

            # Poll IMU at fixed rate
            if CFG.IMU_POLL_HZ > 0 and now >= next_imu_t:
                rover.request_imu()
                next_imu_t = now + 1.0 / CFG.IMU_POLL_HZ

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
                center = last_good_center if last_good_center is not None else img_center
            else:
                last_good_center = center

            error_px = (img_center - center)
            error = error_px / float(w)  # normalized approx [-0.5, 0.5]

            steer = pid.update(error, dt)

            # IMU yaw-rate damping (gz)
            yaw_rate = rover.get_yaw_rate()
            steer = steer - CFG.KYAW * yaw_rate

            steer = float(np.clip(steer, -CFG.MAX_STEER, CFG.MAX_STEER))

            base = CFG.BASE_SPEED * (0.6 if not confident else 1.0)
            left = base - steer
            right = base + steer

            rover.set_wheels(left, right)

            if CFG.SHOW_WINDOWS:
                vis = frame.copy()
                cv2.rectangle(vis, (0, y0), (w - 1, h - 1), (255, 255, 255), 1)
                cv2.line(vis, (img_center, y0), (img_center, h - 1), (255, 255, 255), 1)
                cv2.line(vis, (center, y0), (center, h - 1), (0, 255, 0) if confident else (0, 255, 255), 2)
                cv2.circle(vis, (lp, y0 + (h - y0) // 2), 6, (255, 255, 0), -1)
                cv2.circle(vis, (rp, y0 + (h - y0) // 2), 6, (255, 255, 0), -1)

                txt = f"err_px={error_px:+4d} steer={steer:+.3f} L={left:+.2f} R={right:+.2f} yawRate={yaw_rate:+.2f}rad/s"
                if not confident:
                    txt += " LOW_CONF"
                cv2.putText(vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow("WAVE_ROVER RowFollow", vis)
                cv2.imshow("mask(ROI)", mask)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if CFG.PRINT_DEBUG_HZ > 0 and (now - last_print) > (1.0 / CFG.PRINT_DEBUG_HZ):
                last_print = now
                print(f"[dbg] err_px={error_px:+4d} steer={steer:+.3f} L={left:+.2f} R={right:+.2f} yawRate={yaw_rate:+.2f} rad/s")

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