#!/usr/bin/env python3
"""
Periodic Raspberry Pi camera capture for COLMAP / Gaussian Splatting sessions.

Creates a session directory shaped like:

  colmap_sessions/session_YYYYMMDD_HHMMSS/
    images/frame_00000.jpg
    metadata.jsonl
    capture_log.csv

Usage:
  python3 colmap_capture.py --interval 2 --count 120
  python3 colmap_capture.py --interval 1.5 --duration 180 --width 1920 --height 1080
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def create_session_dir(root: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    session = root / f"session_{stamp}"
    (session / "images").mkdir(parents=True, exist_ok=True)
    return session


def capture_still(path: Path, width: int, height: int, timeout_ms: int) -> bool:
    cmd = [
        "rpicam-still",
        "-o", str(path),
        "--width", str(width),
        "--height", str(height),
        "-t", str(timeout_ms),
        "--nopreview",
        "-n",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout_ms / 1000 + 5,
        )
    except FileNotFoundError:
        print("[ERROR] rpicam-still was not found. Install Raspberry Pi camera tools first.")
        return False
    except subprocess.TimeoutExpired:
        print(f"[WARN] Capture timed out: {path.name}")
        return False

    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="ignore").strip()
        print(f"[WARN] Capture failed for {path.name}: {err}")
        return False

    return path.is_file() and path.stat().st_size > 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture regular still photos for 3D reconstruction.")
    parser.add_argument("--output-dir", default="colmap_sessions", help="Root output folder.")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between photos.")
    parser.add_argument("--count", type=int, default=0, help="Number of photos to capture. 0 means use --duration or run forever.")
    parser.add_argument("--duration", type=float, default=0.0, help="Capture duration in seconds. 0 means use --count or run forever.")
    parser.add_argument("--width", type=int, default=1920, help="Photo width.")
    parser.add_argument("--height", type=int, default=1080, help="Photo height.")
    parser.add_argument("--timeout-ms", type=int, default=1500, help="rpicam-still capture timeout in milliseconds.")
    parser.add_argument("--settle", type=float, default=0.0, help="Extra pause before each photo, useful if the rover stops first.")
    args = parser.parse_args()

    if args.interval <= 0:
        print("[ERROR] --interval must be greater than zero.")
        return 2

    session = create_session_dir(Path(args.output_dir))
    metadata_path = session / "metadata.jsonl"
    csv_path = session / "capture_log.csv"

    print(f"[CAP] Session: {session}")
    print(f"[CAP] Saving {args.width}x{args.height} photos every {args.interval:.2f}s")
    print("[CAP] Press Ctrl+C to stop.")

    captured = 0
    started = time.time()
    next_capture = started

    with metadata_path.open("w") as metadata_file, csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "timestamp", "filename", "width", "height"])

        try:
            while True:
                now = time.time()
                if args.count and captured >= args.count:
                    break
                if args.duration and now - started >= args.duration:
                    break

                sleep_for = next_capture - now
                if sleep_for > 0:
                    time.sleep(sleep_for)

                if args.settle > 0:
                    time.sleep(args.settle)

                filename = f"frame_{captured:05d}.jpg"
                image_path = session / "images" / filename
                timestamp = time.time()

                if capture_still(image_path, args.width, args.height, args.timeout_ms):
                    record = {
                        "frame": captured,
                        "timestamp": timestamp,
                        "filename": filename,
                        "camera": "raspberry_pi_side",
                        "width": args.width,
                        "height": args.height,
                    }
                    metadata_file.write(json.dumps(record) + "\n")
                    metadata_file.flush()
                    writer.writerow([captured, f"{timestamp:.3f}", filename, args.width, args.height])
                    csv_file.flush()
                    print(f"[CAP] #{captured:05d} saved: {filename}")
                    captured += 1

                next_capture += args.interval

        except KeyboardInterrupt:
            print("\n[CAP] Stopped by user.")

    print(f"[CAP] Complete: {captured} photos saved")
    print(f"[CAP] Next: python3 gaussian_splat_pipeline.py {session}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
