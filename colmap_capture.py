#!/usr/bin/env python3
"""
COLMAP data-collection script for WAVE ROVER.

Drives the rover slowly down a row, pausing at regular intervals to capture
sharp still images with full IMU metadata.  Output is a timestamped session
folder containing:

    session_YYYYMMDD_HHMMSS/
        images/          ← COLMAP input
        metadata.jsonl   ← one JSON object per capture (IMU, timestamp, etc.)
        capture_log.csv  ← same data in CSV for quick inspection

Typical usage:
    python colmap_capture.py /dev/ttyUSB0

Press Ctrl-C to stop collection gracefully.
"""

import argparse
import csv
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import serial


# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class CaptureConfig:
    # Driving
    DRIVE_SPEED: float = 0.12          # slow forward speed (m/s-ish wheel cmd)
    DRIVE_DURATION_S: float = 0.6      # drive this long between captures
    SETTLE_TIME_S: float = 0.25        # wait after stopping for vibration to die

    # Camera (rpicam-still)
    IMG_WIDTH: int = 1920
    IMG_HEIGHT: int = 1080
    STILL_TIMEOUT_MS: int = 1500       # rpicam-still capture timeout

    # IMU
    IMU_POLL_HZ: float = 20.0

    # Serial
    BAUD: int = 115200
    SERIAL_TIMEOUT: float = 0.2

    # Session
    OUTPUT_DIR: str = "colmap_sessions"


CFG = CaptureConfig()


# ─── Serial (simplified from Rowfollow.py) ───────────────────────────────────

class RoverSerial:
    """Minimal serial interface: motor commands + IMU reading."""

    def __init__(self, port: str):
        self.ser = serial.Serial(
            port, baudrate=CFG.BAUD, timeout=CFG.SERIAL_TIMEOUT, dsrdtr=None
        )
        self.ser.setRTS(False)
        self.ser.setDTR(False)

        self._lock = threading.Lock()
        self._latest_imu: Optional[Dict[str, Any]] = None
        self._stop = False
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

    def _send(self, obj: Dict[str, Any]):
        msg = (json.dumps(obj, separators=(",", ":")) + "\n").encode()
        with self._lock:
            self.ser.write(msg)

    def set_wheels(self, left: float, right: float):
        self._send({"T": 1, "L": float(left), "R": float(right)})

    def stop_motors(self):
        self.set_wheels(0.0, 0.0)

    def request_imu(self):
        self._send({"T": 126})

    def get_imu(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._latest_imu) if self._latest_imu else None

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
                if isinstance(msg, dict) and "y" in msg:
                    with self._lock:
                        self._latest_imu = msg
            except Exception:
                continue


# ─── Still capture via rpicam-still ──────────────────────────────────────────

def capture_still(path: str, width: int, height: int, timeout_ms: int) -> bool:
    """
    Capture a single JPEG still using rpicam-still.
    Returns True on success.
    """
    try:
        result = subprocess.run(
            [
                "rpicam-still",
                "-o", path,
                "--width", str(width),
                "--height", str(height),
                "-t", str(timeout_ms),
                "--nopreview",
                "-n",
            ],
            capture_output=True,
            timeout=timeout_ms / 1000 + 5,
        )
        return result.returncode == 0 and os.path.isfile(path)
    except Exception as e:
        print(f"  [WARN] capture_still failed: {e}")
        return False


# ─── Session setup ───────────────────────────────────────────────────────────

def create_session_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    session = Path(CFG.OUTPUT_DIR) / f"session_{stamp}"
    (session / "images").mkdir(parents=True, exist_ok=True)
    return session


# ─── Main collection loop ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="COLMAP data collection — drive and capture"
    )
    parser.add_argument("port", help="Serial port, e.g. /dev/ttyUSB0")
    parser.add_argument("--speed", type=float, default=CFG.DRIVE_SPEED,
                        help="Forward wheel speed between captures")
    parser.add_argument("--interval", type=float, default=CFG.DRIVE_DURATION_S,
                        help="Seconds to drive between captures")
    parser.add_argument("--width", type=int, default=CFG.IMG_WIDTH)
    parser.add_argument("--height", type=int, default=CFG.IMG_HEIGHT)
    args = parser.parse_args()

    CFG.DRIVE_SPEED = args.speed
    CFG.DRIVE_DURATION_S = args.interval
    CFG.IMG_WIDTH = args.width
    CFG.IMG_HEIGHT = args.height

    session_dir = create_session_dir()
    img_dir = session_dir / "images"
    jsonl_path = session_dir / "metadata.jsonl"
    csv_path = session_dir / "capture_log.csv"

    print(f"Session folder: {session_dir}")
    print(f"Config: speed={CFG.DRIVE_SPEED}, interval={CFG.DRIVE_DURATION_S}s, "
          f"resolution={CFG.IMG_WIDTH}x{CFG.IMG_HEIGHT}")
    print(f"Press Ctrl-C to stop.\n")

    rover = RoverSerial(args.port)
    rover.start()

    # Wait for first IMU packet
    print("Waiting for IMU…")
    rover.request_imu()
    t0 = time.time()
    while rover.get_imu() is None:
        rover.request_imu()
        time.sleep(0.1)
        if time.time() - t0 > 5:
            print("[WARN] No IMU response after 5 s — continuing without IMU")
            break
    print("IMU ready.\n")

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "timestamp", "filename",
        "roll", "pitch", "yaw",
        "gx", "gy", "gz",
        "elapsed_s", "est_distance_m",
    ])

    jsonl_file = open(jsonl_path, "w")

    frame_idx = 0
    start_time = time.time()
    est_distance = 0.0
    next_imu_t = time.time()

    try:
        while True:
            # ── 1. Drive forward ──────────────────────────────────────────
            print(f"[{frame_idx:04d}] Driving {CFG.DRIVE_DURATION_S:.2f}s "
                  f"at speed {CFG.DRIVE_SPEED:.2f}…")
            rover.set_wheels(CFG.DRIVE_SPEED, CFG.DRIVE_SPEED)

            drive_start = time.monotonic()
            while (time.monotonic() - drive_start) < CFG.DRIVE_DURATION_S:
                now = time.time()
                if CFG.IMU_POLL_HZ > 0 and now >= next_imu_t:
                    rover.request_imu()
                    next_imu_t = now + 1.0 / CFG.IMU_POLL_HZ
                time.sleep(0.02)

            est_distance += CFG.DRIVE_SPEED * CFG.DRIVE_DURATION_S

            # ── 2. Stop and let vibrations settle ─────────────────────────
            rover.stop_motors()
            time.sleep(CFG.SETTLE_TIME_S)

            # ── 3. Poll IMU right before capture ──────────────────────────
            rover.request_imu()
            time.sleep(0.05)
            imu = rover.get_imu()

            # ── 4. Capture still image ────────────────────────────────────
            fname = f"frame_{frame_idx:05d}.jpg"
            fpath = str(img_dir / fname)
            capture_ts = time.time()

            print(f"[{frame_idx:04d}] Capturing {fname}…", end=" ", flush=True)
            ok = capture_still(fpath, CFG.IMG_WIDTH, CFG.IMG_HEIGHT,
                               CFG.STILL_TIMEOUT_MS)

            if ok:
                print("OK")
            else:
                print("FAILED — skipping")
                frame_idx += 1
                continue

            # ── 5. Record metadata ────────────────────────────────────────
            elapsed = capture_ts - start_time
            r  = imu.get("r", None) if imu else None
            p  = imu.get("p", None) if imu else None
            y  = imu.get("y", None) if imu else None
            gx = imu.get("gx", None) if imu else None
            gy = imu.get("gy", None) if imu else None
            gz = imu.get("gz", None) if imu else None

            record = {
                "frame": frame_idx,
                "timestamp": capture_ts,
                "filename": fname,
                "imu": {"r": r, "p": p, "y": y, "gx": gx, "gy": gy, "gz": gz},
                "elapsed_s": round(elapsed, 3),
                "est_distance_m": round(est_distance, 3),
            }

            jsonl_file.write(json.dumps(record) + "\n")
            jsonl_file.flush()

            csv_writer.writerow([
                frame_idx, f"{capture_ts:.3f}", fname,
                r, p, y, gx, gy, gz,
                f"{elapsed:.3f}", f"{est_distance:.3f}",
            ])
            csv_file.flush()

            print(f"       IMU y={y}  dist≈{est_distance:.2f}m  "
                  f"elapsed={elapsed:.1f}s")

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n\nStopping…")

    finally:
        rover.stop_motors()
        rover.close()
        csv_file.close()
        jsonl_file.close()

        print(f"\nCollection complete.")
        print(f"  Frames captured : {frame_idx}")
        print(f"  Est. distance   : {est_distance:.2f} m")
        print(f"  Session folder  : {session_dir}")
        print(f"\nTo reconstruct, copy the session folder to your computer and run:")
        print(f"  colmap automatic_reconstructor \\")
        print(f"    --workspace_path {session_dir} \\")
        print(f"    --image_path {session_dir}/images")


if __name__ == "__main__":
    main()
