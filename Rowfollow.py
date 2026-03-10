#!/usr/bin/env python3
"""
WAVE_ROVER Row-Following (rpicam-vid MJPEG) + IMU yaw-rate damping + serial JSON motor commands
+ Full state machine for U-turn headland navigation

States:
  ROW_FOLLOWING  -> normal PID corridor following
  TURN_STOP      -> pause briefly before turning
  TURN_PHASE1    -> rotate 90 deg (first quarter turn)
  TURN_ADVANCE   -> drive forward one row spacing
  TURN_PHASE2    -> rotate another 90 deg (now facing opposite direction)
  REALIGN        -> creep forward until ultrasonics detect the new row
"""

import argparse
import csv
import json
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import cv2
import serial
import subprocess
import pigpio

# ── GPIO pin assignments (change to match your wiring) ──────────────────────
TRIG_L = 23
ECHO_L = 24
TRIG_R = 27
ECHO_R = 22
# ────────────────────────────────────────────────────────────────────────────

ser, pi = None, None


def get_distance(Trig, Echo) -> Optional[float]:
    """Returns distance in metres, or None on timeout."""
    pi.gpio_trigger(Trig, 10, 1)

    start = time.monotonic()
    while pi.read(Echo) == 0:
        if time.monotonic() - start > 0.03:
            return None

    t0 = time.monotonic()
    while pi.read(Echo) == 1:
        if time.monotonic() - t0 > 0.03:
            return None

    t1 = time.monotonic()
    return (t1 - t0) * 343 / 2   # metres


class RpiCamMJPEG:
    def __init__(self, width=640, height=480, fps=30):
        self.proc = subprocess.Popen(
            ["rpicam-vid", "-t", "0",
             "--width", str(width), "--height", str(height),
             "--framerate", str(fps), "--codec", "mjpeg", "-o", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
        )
        if self.proc.stdout is None:
            raise RuntimeError("Failed to start rpicam-vid.")
        self.buf = bytearray()

    def read_bgr(self) -> np.ndarray:
        while True:
            chunk = self.proc.stdout.read(4096)
            if not chunk:
                raise RuntimeError("rpicam-vid stopped producing data.")
            self.buf.extend(chunk)

            start = self.buf.find(b"\xff\xd8")
            end   = self.buf.find(b"\xff\xd9")
            if start != -1 and end != -1 and end > start:
                jpg = self.buf[start:end + 2]
                del self.buf[:end + 2]
                arr   = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame

    def close(self):
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
        except Exception:
            pass


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class Config:
    # Motor limits
    V_MAX:       float = 0.50
    BASE_SPEED:  float = 0.20
    TURN_SPEED:  float = 0.18   # wheel speed during on-spot rotation
    CREEP_SPEED: float = 0.12   # slow forward during REALIGN

    # Vision
    CAM_W: int = 640
    CAM_H: int = 480
    ROI_Y_START_FRAC:  float = 0.60
    EXG_THRESH:        int   = 40
    MORPH_K:           int   = 5
    MORPH_ITERS:       int   = 1
    HIST_SMOOTH_K:     int   = 21
    MIN_PEAK_SEP_FRAC: float = 0.20

    # PID
    KP: float = 0.90
    KI: float = 0.00
    KD: float = 0.20
    MAX_STEER: float = 0.25

    # IMU damping
    KYAW:                    float = 0.08
    YAW_RATE_FALLBACK_SMOOTH: float = 0.6

    # Serial
    SERIAL_TIMEOUT_S: float = 0.2

    # UI
    SHOW_WINDOWS:   bool  = True
    PRINT_DEBUG_HZ: float = 2.0

    # IMU polling
    IMU_POLL_HZ: float = 20.0

    # ── Turn / headland parameters ────────────────────────────────────────
    # End-of-row triggers
    EOR_SIDE_OPEN_M:   float = 0.80   # both side sensors > this → row ended
    EOR_CONFIRM_COUNT: int   = 5      # consecutive frames needed to trigger

    # Turn geometry
    ROW_SPACING_M:   float = 0.55   # lateral distance between row centres
    TURN_ANGLE_DEG:  float = 90.0   # each phase turns this many degrees
    TURN_TIMEOUT_S:  float = 8.0    # abort turn phase if it takes too long
    TURN_DIRECTION:  int   = -1     # +1 = turn left first, -1 = turn right first
                                    # set so rover steps TOWARD the next row

    # REALIGN: look for both side walls closer than this
    REALIGN_WALL_M:  float = 0.60
    REALIGN_TIMEOUT_S: float = 6.0

    # Brief pause before starting turn (seconds)
    PRE_TURN_PAUSE_S: float = 0.4

    # ── COLMAP capture (runs during ROW_FOLLOWING) ─────────────────────────
    CAPTURE_ENABLED:    bool  = False
    CAPTURE_INTERVAL_S: float = 2.0    # seconds between captures
    CAPTURE_SETTLE_S:   float = 0.25   # pause after stopping before shutter
    CAPTURE_WIDTH:      int   = 1920
    CAPTURE_HEIGHT:     int   = 1080
    CAPTURE_TIMEOUT_MS: int   = 1500
    CAPTURE_OUTPUT_DIR: str   = "colmap_sessions"


CFG = Config()


# =============================================================================
# Rover state machine
# =============================================================================
class State(Enum):
    ROW_FOLLOWING = auto()
    TURN_STOP     = auto()   # pause
    TURN_PHASE1   = auto()   # rotate 90 °
    TURN_ADVANCE  = auto()   # drive one row spacing
    TURN_PHASE2   = auto()   # rotate another 90 °
    REALIGN       = auto()   # creep until next row walls appear


# =============================================================================
# PID
# =============================================================================
class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral   = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral   = 0.0
        self.prev_error = 0.0

    def update(self, error: float, dt: float) -> float:
        if dt <= 1e-6:
            return 0.0
        self.integral  += error * dt
        d = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * d


# =============================================================================
# Serial: Waveshare JSON I/O
# =============================================================================
class WaveRoverSerial:
    def __init__(self, port: str, baud: int = 115200, timeout: float = 0.2):
        self.ser = serial.Serial(port, baudrate=baud, timeout=timeout,
                                 dsrdtr=None)
        self.ser.setRTS(False)
        self.ser.setDTR(False)

        self._lock = threading.Lock()
        self.latest_msg: Optional[Dict[str, Any]] = None

        self._yaw_rad:       Optional[float] = None
        self._yaw_rate_rad_s: float          = 0.0
        self._yaw_prev_t:    Optional[float] = None

        self._stop   = False
        self._thread = threading.Thread(target=self._read_loop, daemon=True)

    def start(self):
        self._thread.start()

    def close(self):
        self._stop = True
        time.sleep(0.05)
        try:
            self.ser.close()
        except Exception:
            pass

    def send_json(self, obj: Dict[str, Any]):
        msg = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
        with self._lock:
            self.ser.write(msg)

    def set_wheels(self, left: float, right: float):
        left  = float(np.clip(left,  -CFG.V_MAX, CFG.V_MAX))
        right = float(np.clip(right, -CFG.V_MAX, CFG.V_MAX))
        self.send_json({"T": 1, "L": left, "R": right})

    def stop_motors(self):
        self.set_wheels(0.0, 0.0)

    def request_imu(self):
        self.send_json({"T": 126})

    def _read_loop(self):
        while not self._stop:
            try:
                line = self.ser.readline()
                if not line:
                    continue
                s = line.decode("utf-8", errors="ignore").strip()
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
        now     = time.time()
        has_gz  = False

        if "gz" in msg and isinstance(msg["gz"], (int, float)):
            val = float(msg["gz"])
            if abs(val) > 6.5:
                val = np.deg2rad(val)
            self._yaw_rate_rad_s = val
            has_gz = True

        if "y" not in msg or not isinstance(msg["y"], (int, float)):
            return

        yaw_rad = float(msg["y"])
        if abs(yaw_rad) > 6.5:
            yaw_rad = np.deg2rad(yaw_rad)

        if (not has_gz) and (self._yaw_rad is not None) and (self._yaw_prev_t is not None):
            dt = now - self._yaw_prev_t
            if dt > 1e-3:
                dy  = np.arctan2(np.sin(yaw_rad - self._yaw_rad),
                                 np.cos(yaw_rad - self._yaw_rad))
                est = dy / dt
                a   = CFG.YAW_RATE_FALLBACK_SMOOTH
                self._yaw_rate_rad_s = a * self._yaw_rate_rad_s + (1 - a) * est

        self._yaw_rad    = yaw_rad
        self._yaw_prev_t = now

    def get_imu_snapshot(self) -> Optional[Dict[str, Any]]:
        """Return a copy of the most recent IMU message (or None)."""
        with self._lock:
            return dict(self.latest_msg) if self.latest_msg else None

    def get_yaw_rad(self) -> Optional[float]:
        return self._yaw_rad

    def get_yaw_rate(self) -> float:
        return float(self._yaw_rate_rad_s)


# =============================================================================
# Vision helpers
# =============================================================================
def exg_mask(bgr: np.ndarray, thresh: int) -> np.ndarray:
    B, G, R = cv2.split(bgr)
    exg = (2 * G.astype(np.int16) - R.astype(np.int16) - B.astype(np.int16))
    exg = np.clip(exg, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(exg, thresh, 255, cv2.THRESH_BINARY)
    return mask


def morph_cleanup(mask: np.ndarray, k: int, iters: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=iters)
    return mask


def smooth_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    return np.convolve(x, np.ones(k, dtype=np.float32) / k, mode="same")


def find_corridor_center(roi_mask: np.ndarray, w_full: int
                         ) -> Tuple[Optional[int], Tuple[int, int], np.ndarray]:
    hist = np.sum(roi_mask > 0, axis=0).astype(np.float32)
    hist = smooth_1d(hist, CFG.HIST_SMOOTH_K)
    w    = hist.shape[0]
    half = w // 2
    lp   = int(np.argmax(hist[:half]))
    rp   = int(np.argmax(hist[half:]) + half)

    min_sep = int(w_full * CFG.MIN_PEAK_SEP_FRAC)
    if (rp - lp) < min_sep:
        return None, (lp, rp), hist

    return (lp + rp) // 2, (lp, rp), hist


# =============================================================================
# Angle helpers
# =============================================================================
def normalize_deg(a: float) -> float:
    """Wrap to (-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0


def angle_diff_deg(target: float, current: float) -> float:
    """Signed shortest angular distance (degrees)."""
    return normalize_deg(target - current)


def yaw_deg(rover: WaveRoverSerial) -> Optional[float]:
    r = rover.get_yaw_rad()
    return np.degrees(r) if r is not None else None


# =============================================================================
# Turn sub-routines
# =============================================================================
def rotate_to_heading(rover: WaveRoverSerial, target_deg: float,
                      tolerance: float = 2.5) -> bool:
    """
    Spin in place until IMU yaw reaches target_deg (±tolerance).
    Returns True on success, False on timeout.
    """
    deadline = time.time() + CFG.TURN_TIMEOUT_S
    while time.time() < deadline:
        current = yaw_deg(rover)
        if current is None:
            time.sleep(0.02)
            continue

        diff = angle_diff_deg(target_deg, current)
        if abs(diff) <= tolerance:
            rover.stop_motors()
            return True

        # Spin direction follows sign of diff
        spin = CFG.TURN_SPEED if diff > 0 else -CFG.TURN_SPEED
        rover.set_wheels(-spin, spin)   # tank-turn
        time.sleep(0.02)

    rover.stop_motors()
    print("[WARN] rotate_to_heading timed out")
    return False


def drive_straight_timed(rover: WaveRoverSerial, distance_m: float):
    """
    Drive forward for a fixed time calculated from BASE_SPEED.
    Simple dead-reckoning; good enough for short headland advances.
    Assumes speed ≈ BASE_SPEED in m/s — calibrate ADVANCE_MPS if needed.
    """
    ADVANCE_MPS = 0.15   # tweak to match actual rover speed at BASE_SPEED
    duration    = distance_m / ADVANCE_MPS
    rover.set_wheels(CFG.BASE_SPEED, CFG.BASE_SPEED)
    time.sleep(duration)
    rover.stop_motors()


def realign_to_row(rover: WaveRoverSerial) -> bool:
    """
    Creep forward until both side ultrasonics see crop walls again.
    Returns True on success, False on timeout.
    """
    deadline = time.time() + CFG.REALIGN_TIMEOUT_S
    while time.time() < deadline:
        dL = get_distance(TRIG_L, ECHO_L)
        dR = get_distance(TRIG_R, ECHO_R)

        l_ok = (dL is not None) and (dL < CFG.REALIGN_WALL_M)
        r_ok = (dR is not None) and (dR < CFG.REALIGN_WALL_M)

        if l_ok and r_ok:
            rover.stop_motors()
            print("[INFO] Row re-acquired")
            return True

        rover.set_wheels(CFG.CREEP_SPEED, CFG.CREEP_SPEED)
        time.sleep(0.05)

    rover.stop_motors()
    print("[WARN] realign_to_row timed out — row not found")
    return False


def execute_uturn(rover: WaveRoverSerial):
    """
    U-turn that moves the rover onto the NEXT (adjacent) row:

      Row N  →→→→→→→→→ [end]
                           ↓  phase 1: rotate 90° toward next row
                           ↓  phase 2: advance one row spacing (lateral step)
      Row N+1 ←←←←←←← [start]
                           ↑  phase 3: rotate 90° to face back into field

    TURN_DIRECTION = +1  →  first turn is LEFT  (counter-clockwise, +yaw)
    TURN_DIRECTION = -1  →  first turn is RIGHT (clockwise, -yaw)
    Set it so the rover steps toward the unworked row, not away from it.
    """
    start_yaw = yaw_deg(rover)
    if start_yaw is None:
        print("[WARN] No IMU data — skipping turn")
        return

    d = CFG.TURN_DIRECTION   # +1 or -1
    print(f"[TURN] Starting U-turn from yaw={start_yaw:.1f}°  dir={'LEFT' if d>0 else 'RIGHT'}")

    # ── Phase 1: rotate 90° toward the next row ──────────────────────────
    target1 = normalize_deg(start_yaw + d * CFG.TURN_ANGLE_DEG)
    print(f"[TURN] Phase 1: rotate to {target1:.1f}°")
    rotate_to_heading(rover, target1)
    time.sleep(0.1)

    # ── Phase 2: drive forward ONE row spacing (this is the lateral step) ─
    # After a 90° turn the rover is now facing sideways across the field,
    # so driving straight moves it exactly one row spacing to the side.
    print(f"[TURN] Phase 2: advance {CFG.ROW_SPACING_M:.2f} m to next row")
    drive_straight_timed(rover, CFG.ROW_SPACING_M)
    time.sleep(0.1)

    # ── Phase 3: rotate another 90° in the SAME direction (total = 180°) ──
    # Now the rover faces back into the field, aligned with the next row.
    target2 = normalize_deg(start_yaw + d * 2 * CFG.TURN_ANGLE_DEG)
    print(f"[TURN] Phase 3: rotate to {target2:.1f}°")
    rotate_to_heading(rover, target2)
    time.sleep(0.1)

    # ── Phase 4: creep forward until side sensors confirm the new row ──────
    print("[TURN] Phase 4: realigning to new row…")
    realign_to_row(rover)
    time.sleep(0.15)

    print("[TURN] U-turn complete — resuming row following")


# =============================================================================
# End-of-row detector  (debounced)
# =============================================================================
class EndOfRowDetector:
    def __init__(self):
        self._count = 0

    def update(self, dL: Optional[float], dR: Optional[float]) -> bool:
        """
        Returns True when end-of-row is confirmed for N consecutive frames.
        Triggers when both side sensors read open space (no crop walls).
        """
        side_open = (dL is not None and dL > CFG.EOR_SIDE_OPEN_M and
                     dR is not None and dR > CFG.EOR_SIDE_OPEN_M)

        if side_open:
            self._count += 1
        else:
            self._count = 0

        return self._count >= CFG.EOR_CONFIRM_COUNT

    def reset(self):
        self._count = 0


# =============================================================================
# COLMAP capture helpers
# =============================================================================
def create_session_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    session = Path(CFG.CAPTURE_OUTPUT_DIR) / f"session_{stamp}"
    (session / "images").mkdir(parents=True, exist_ok=True)
    return session


def capture_still(path: str) -> bool:
    """Capture a single JPEG still via rpicam-still. Returns True on success."""
    try:
        result = subprocess.run(
            [
                "rpicam-still",
                "-o", path,
                "--width", str(CFG.CAPTURE_WIDTH),
                "--height", str(CFG.CAPTURE_HEIGHT),
                "-t", str(CFG.CAPTURE_TIMEOUT_MS),
                "--nopreview", "-n",
            ],
            capture_output=True,
            timeout=CFG.CAPTURE_TIMEOUT_MS / 1000 + 5,
        )
        return result.returncode == 0 and os.path.isfile(path)
    except Exception as e:
        print(f"  [CAP] capture_still failed: {e}")
        return False


# =============================================================================
# Main
# =============================================================================
def main():
    global pi

    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=str,
                        help="Serial port, e.g. /dev/ttyUSB0")
    parser.add_argument("--capture", action="store_true",
                        help="Enable COLMAP image capture during row following")
    parser.add_argument("--capture-interval", type=float, default=CFG.CAPTURE_INTERVAL_S,
                        help="Seconds between captures (default: %(default)s)")
    args = parser.parse_args()

    CFG.CAPTURE_ENABLED = args.capture
    CFG.CAPTURE_INTERVAL_S = args.capture_interval

    rover = WaveRoverSerial(args.port, baud=115200,
                            timeout=CFG.SERIAL_TIMEOUT_S)
    rover.start()

    cam = RpiCamMJPEG(CFG.CAM_W, CFG.CAM_H, fps=30)
    pid = PID(CFG.KP, CFG.KI, CFG.KD)
    eor = EndOfRowDetector()

    # ── pigpio ───────────────────────────────────────────────────────────
    pi = pigpio.pi()
    if not pi.connected:
        print("Failed to connect to pigpio. Run: sudo systemctl start pigpiod")
        return

    for trig, echo in [(TRIG_L, ECHO_L), (TRIG_R, ECHO_R)]:
        pi.set_mode(trig, pigpio.OUTPUT)
        pi.set_mode(echo, pigpio.INPUT)
        pi.write(trig, 0)
    time.sleep(0.1)

    # ── COLMAP capture session ────────────────────────────────────────────
    cap_session_dir = None
    cap_csv_file    = None
    cap_csv_writer  = None
    cap_jsonl_file  = None
    cap_frame_idx   = 0
    cap_last_t      = 0.0

    if CFG.CAPTURE_ENABLED:
        cap_session_dir = create_session_dir()
        cap_jsonl_file  = open(cap_session_dir / "metadata.jsonl", "w")
        cap_csv_file    = open(cap_session_dir / "capture_log.csv", "w", newline="")
        cap_csv_writer  = csv.writer(cap_csv_file)
        cap_csv_writer.writerow([
            "frame", "timestamp", "filename",
            "roll", "pitch", "yaw", "gx", "gy", "gz",
        ])
        print(f"[CAP] COLMAP capture ON — session: {cap_session_dir}")
        print(f"[CAP] Interval: {CFG.CAPTURE_INTERVAL_S}s, "
              f"resolution: {CFG.CAPTURE_WIDTH}x{CFG.CAPTURE_HEIGHT}")

    print("Running. Press 'q' to quit.")

    state           = State.ROW_FOLLOWING
    last_t          = time.time()
    last_print      = 0.0
    last_good_center = None
    next_imu_t      = time.time()

    try:
        while True:
            now = time.time()
            dt  = max(now - last_t, 1e-4)
            last_t = now

            # ── IMU polling ──────────────────────────────────────────────
            if CFG.IMU_POLL_HZ > 0 and now >= next_imu_t:
                rover.request_imu()
                next_imu_t = now + 1.0 / CFG.IMU_POLL_HZ

            # ── Read side ultrasonics ─────────────────────────────────────
            dL = get_distance(TRIG_L, ECHO_L)
            dR = get_distance(TRIG_R, ECHO_R)

            # ════════════════════════════════════════════════════════════
            # STATE MACHINE
            # ════════════════════════════════════════════════════════════

            if state == State.ROW_FOLLOWING:
                # ── End-of-row check ─────────────────────────────────────
                if eor.update(dL, dR):
                    print("[STATE] End of row detected → TURN_STOP")
                    rover.stop_motors()
                    eor.reset()
                    pid.reset()
                    state        = State.TURN_STOP
                    turn_pause_t = now
                    continue

                # ── COLMAP capture (periodic still) ──────────────────────
                if CFG.CAPTURE_ENABLED and (now - cap_last_t) >= CFG.CAPTURE_INTERVAL_S:
                    rover.stop_motors()
                    time.sleep(CFG.CAPTURE_SETTLE_S)

                    rover.request_imu()
                    time.sleep(0.05)
                    imu = rover.get_imu_snapshot()

                    fname = f"frame_{cap_frame_idx:05d}.jpg"
                    fpath = str(cap_session_dir / "images" / fname)
                    cap_ts = time.time()

                    ok = capture_still(fpath)
                    if ok:
                        r  = imu.get("r")  if imu else None
                        p  = imu.get("p")  if imu else None
                        y  = imu.get("y")  if imu else None
                        gx = imu.get("gx") if imu else None
                        gy = imu.get("gy") if imu else None
                        gz = imu.get("gz") if imu else None

                        record = {
                            "frame": cap_frame_idx,
                            "timestamp": cap_ts,
                            "filename": fname,
                            "imu": {"r": r, "p": p, "y": y,
                                    "gx": gx, "gy": gy, "gz": gz},
                        }
                        cap_jsonl_file.write(json.dumps(record) + "\n")
                        cap_jsonl_file.flush()
                        cap_csv_writer.writerow([
                            cap_frame_idx, f"{cap_ts:.3f}", fname,
                            r, p, y, gx, gy, gz,
                        ])
                        cap_csv_file.flush()
                        print(f"[CAP] #{cap_frame_idx:04d} saved  yaw={y}")
                        cap_frame_idx += 1
                    else:
                        print(f"[CAP] #{cap_frame_idx:04d} FAILED — skipped")

                    cap_last_t = time.time()
                    last_t = time.time()
                    continue

                # ── Vision-based PID row following ───────────────────────
                frame = cam.read_bgr()
                h, w, _ = frame.shape
                y0  = int(h * CFG.ROI_Y_START_FRAC)
                roi = frame[y0:h, :]

                mask   = exg_mask(roi, CFG.EXG_THRESH)
                mask   = morph_cleanup(mask, CFG.MORPH_K, CFG.MORPH_ITERS)
                center, (lp, rp), _hist = find_corridor_center(mask, w)
                img_center = w // 2

                confident = True
                if center is None:
                    confident = False
                    center = last_good_center if last_good_center is not None \
                             else img_center
                else:
                    last_good_center = center

                error_px = img_center - center
                error    = error_px / float(w)

                steer     = pid.update(error, dt)
                yaw_rate  = rover.get_yaw_rate()
                steer     = steer - CFG.KYAW * yaw_rate
                steer     = float(np.clip(steer, -CFG.MAX_STEER, CFG.MAX_STEER))

                base  = CFG.BASE_SPEED * (0.6 if not confident else 1.0)
                left  = base - steer
                right = base + steer
                rover.set_wheels(left, right)

                # ── Visualisation ─────────────────────────────────────────
                if CFG.SHOW_WINDOWS:
                    vis = frame.copy()
                    cv2.rectangle(vis, (0, y0), (w-1, h-1), (255,255,255), 1)
                    cv2.line(vis, (img_center, y0), (img_center, h-1), (255,255,255), 1)
                    cv2.line(vis, (center, y0), (center, h-1),
                             (0,255,0) if confident else (0,255,255), 2)
                    cv2.circle(vis, (lp, y0 + (h-y0)//2), 6, (255,255,0), -1)
                    cv2.circle(vis, (rp, y0 + (h-y0)//2), 6, (255,255,0), -1)

                    dL_txt = f"{dL:.2f}" if dL else "---"
                    dR_txt = f"{dR:.2f}" if dR else "---"
                    hud = (f"STATE:{state.name}  err={error_px:+4d}px  "
                           f"steer={steer:+.3f}  L={left:+.2f} R={right:+.2f}  "
                           f"US L={dL_txt} R={dR_txt}m")
                    cv2.putText(vis, hud, (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255,255,255), 1, cv2.LINE_AA)
                    cv2.imshow("WAVE_ROVER RowFollow", vis)
                    cv2.imshow("mask(ROI)", mask)

                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break

                if CFG.PRINT_DEBUG_HZ > 0 and (now - last_print) > (1.0 / CFG.PRINT_DEBUG_HZ):
                    last_print = now
                    print(f"[dbg] {state.name} err={error_px:+4d} "
                          f"steer={steer:+.3f} L={left:+.2f} R={right:+.2f} "
                          f"yawRate={yaw_rate:+.2f}rad/s "
                          f"US L={dL_txt} R={dR_txt}m")

            # ── TURN_STOP: brief pause before turning ─────────────────────
            elif state == State.TURN_STOP:
                rover.stop_motors()
                if now - turn_pause_t >= CFG.PRE_TURN_PAUSE_S:
                    print("[STATE] TURN_STOP → executing U-turn")
                    state = State.ROW_FOLLOWING   # execute_uturn is blocking;
                    execute_uturn(rover)           # it returns ready to follow
                    last_good_center = None
                    eor.reset()
                    pid.reset()
                    last_t = time.time()

            # ── Key-press quit from non-FOLLOWING states ──────────────────
            if CFG.SHOW_WINDOWS:
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

    except KeyboardInterrupt:
        pass

    finally:
        print("\nStopping motors…")
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
        if pi and pi.connected:
            pi.stop()

        if CFG.CAPTURE_ENABLED:
            if cap_csv_file:
                cap_csv_file.close()
            if cap_jsonl_file:
                cap_jsonl_file.close()
            print(f"\n[CAP] Session complete — {cap_frame_idx} frames saved")
            print(f"[CAP] Output: {cap_session_dir}")
            print(f"[CAP] To reconstruct on your computer:")
            print(f"  colmap automatic_reconstructor \\")
            print(f"    --workspace_path {cap_session_dir} \\")
            print(f"    --image_path {cap_session_dir}/images")


if __name__ == "__main__":
    main()