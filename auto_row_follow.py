import serial
import argparse
import csv
import json
import os
import re
import subprocess
import threading
import time
from enum import Enum, auto
from pathlib import Path
import cv2
import numpy as np

import colmap_capture

try:
    import pigpio
except ImportError:
    pigpio = None

# Side ultrasonic (HC-SR04) GPIO wiring -- same pins as Rowfollow.py.
TRIG_L = 23
ECHO_L = 24
TRIG_R = 27
ECHO_R = 22

"""
This is a file that establishes a class RoverSerial and writes instructions for its autonomous navigation
around rows of a farm.
"""

class Rover:
    def __init__(self, port: str):
        self.ser = serial.Serial(port, baudrate=115200, dsrdtr=None)
        self.ser.setRTS(False)
        self.ser.setDTR(False)

        # True whenever the last command told the rover to move; used to gate
        # the COLMAP capture so we only shoot frames while actually driving.
        self.moving = False

        self._thread = threading.Thread(target=self.read_serial)
        self._thread.daemon = True
        self._thread.start()
    def start(self):
        self._thread.start()
    def move_forward(self):
        self.moving = True
        self.ser.write(b'{"T":1,"L":0.2,"R":0.2}\n')
    def stop(self):
        self.moving = False
        self.ser.write(b'{"T":1,"L":0,"R":0}\n')
    def turn_left(self):
        self.moving = True
        self.ser.write(b'{"T":1,"L":0.2,"R":-0.2}\n')
    def turn_right(self):
        self.moving = True
        self.ser.write(b'{"T":1,"L":-0.2,"R":0.2}\n')
    def get_imu(self):
        self.ser.write(b'{"T":126}\n')
    def read_serial(self):
        while True:
            data = self.ser.readline().decode('utf-8')
            if data:
                print(f"Received: {data}", end='')
    def close(self):
        self.ser.close()
        self._thread.join()

class State(Enum):
    FOLLOWING = auto()
    CHANGE_ROW = auto()
    TURNING = auto()

def find_usb_video_device():
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        result = None

    if result:
        current_device_is_usb = False
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                current_device_is_usb = False
                continue
            if not line.startswith("\t"):
                current_device_is_usb = "usb" in stripped.lower() or "logitech" in stripped.lower()
                continue
            if current_device_is_usb:
                match = re.search(r"/dev/video\d+", stripped)
                if match:
                    return match.group(0)

    for index in range(10):
        device = f"/dev/video{index}"
        if os.path.exists(device):
            return device

    raise RuntimeError("Could not find a /dev/video* camera device.")

class Webcam:
    """USB webcams on Raspberry Pi are exposed by Linux V4L2 as /dev/video*."""
    def __init__(self, device="auto", width=640, height=480, fps=30):
        if device == "auto":
            device = find_usb_video_device()
        self.device = device
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at {device}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Webcam did not return a frame.")
        return frame

    def close(self):
        self.cap.release()

class Ultrasonics:
    """Side HC-SR04 range sensors read through pigpio (same wiring as Rowfollow.py).

    Each reading is a distance in metres, or None on echo timeout (target beyond
    ~5 m / no echo). Used to detect the end of a row: when both sides open up to
    the headland, their ranges shoot up together.
    """
    def __init__(self, trig_l=TRIG_L, echo_l=ECHO_L, trig_r=TRIG_R, echo_r=ECHO_R):
        if pigpio is None:
            raise RuntimeError("pigpio is not installed; ultrasonic sensors unavailable.")
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("Could not connect to pigpiod. Run: sudo systemctl start pigpiod")

        self._pins = {"L": (trig_l, echo_l), "R": (trig_r, echo_r)}
        for trig, echo in self._pins.values():
            self.pi.set_mode(trig, pigpio.OUTPUT)
            self.pi.set_mode(echo, pigpio.INPUT)
            self.pi.write(trig, 0)
        time.sleep(0.1)

    def _distance(self, trig, echo):
        """Distance in metres, or None on timeout."""
        self.pi.gpio_trigger(trig, 10, 1)

        start = time.monotonic()
        while self.pi.read(echo) == 0:
            if time.monotonic() - start > 0.03:
                return None

        t0 = time.monotonic()
        while self.pi.read(echo) == 1:
            if time.monotonic() - t0 > 0.03:
                return None

        return (time.monotonic() - t0) * 343 / 2

    def left(self):
        return self._distance(*self._pins["L"])

    def right(self):
        return self._distance(*self._pins["R"])

    def close(self):
        if self.pi is not None and self.pi.connected:
            self.pi.stop()

class MotionCapture:
    """Background COLMAP still capture that fires at a fixed rate while the rover moves.

    Keeps a single Picamera2 stream open for the whole run so grabbing a frame is
    just a buffer copy + JPEG encode -- fast enough to sustain a true 5 Hz. Only
    captures when rover.moving is True, so we build a clean reconstruction dataset
    without shooting duplicate stills while parked. Falls back to per-frame
    rpicam-still if picamera2 is unavailable (which typically can't reach 5 Hz).
    """
    def __init__(self, rover: "Rover", rate_hz: float = 5.0, width: int = 1920,
                 height: int = 1080, timeout_ms: int = 100,
                 output_dir: str = "colmap_sessions"):
        self.rover = rover
        self.interval = 1.0 / rate_hz
        self.width = width
        self.height = height
        self.timeout_ms = timeout_ms
        self.output_dir = output_dir
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _open_picamera(self):
        """Open and start a persistent Picamera2 stream, or return None if unavailable."""
        try:
            from picamera2 import Picamera2
        except ImportError:
            return None
        try:
            picam2 = Picamera2()
            config = picam2.create_still_configuration(main={"size": (self.width, self.height)})
            picam2.configure(config)
            picam2.start()
            time.sleep(0.5)  # let auto exposure / white balance settle once
            return picam2
        except Exception as exc:
            print(f"[CAP] Could not start Picamera2 ({exc}); using rpicam-still fallback.")
            return None

    def _run(self):
        session = colmap_capture.create_session_dir(Path(self.output_dir))
        metadata_path = session / "metadata.jsonl"
        csv_path = session / "capture_log.csv"

        picam2 = self._open_picamera()
        backend = "picamera2 (persistent)" if picam2 is not None else "rpicam-still (per-frame)"
        print(f"[CAP] Motion capture session: {session} ({1.0 / self.interval:.1f} Hz while moving, {backend})")

        captured = 0
        next_capture = time.monotonic()
        try:
            with metadata_path.open("w") as metadata_file, csv_path.open("w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["frame", "timestamp", "filename", "width", "height"])

                while not self._stop.is_set():
                    now = time.monotonic()
                    if now < next_capture:
                        # Stay responsive to stop() while waiting for the next slot.
                        self._stop.wait(min(next_capture - now, 0.05))
                        continue
                    next_capture += self.interval
                    # If capture fell behind (or the rover was parked), resync rather
                    # than firing a burst to catch up.
                    if next_capture < now:
                        next_capture = now + self.interval

                    if not self.rover.moving:
                        continue

                    filename = f"frame_{captured:05d}.jpg"
                    image_path = session / "images" / filename
                    timestamp = time.time()

                    if picam2 is not None:
                        ok = self._capture_persistent(picam2, image_path)
                    else:
                        ok = colmap_capture.capture_still(image_path, self.width, self.height, self.timeout_ms)

                    if ok:
                        record = {
                            "frame": captured,
                            "timestamp": timestamp,
                            "filename": filename,
                            "camera": "raspberry_pi_side",
                            "width": self.width,
                            "height": self.height,
                        }
                        metadata_file.write(json.dumps(record) + "\n")
                        metadata_file.flush()
                        writer.writerow([captured, f"{timestamp:.3f}", filename, self.width, self.height])
                        csv_file.flush()
                        captured += 1
        finally:
            if picam2 is not None:
                picam2.stop()
                picam2.close()

        print(f"[CAP] Motion capture complete: {captured} photos in {session}")

    def _capture_persistent(self, picam2, path: Path) -> bool:
        try:
            picam2.capture_file(str(path))
        except Exception as exc:
            print(f"[WARN] Capture failed for {path.name}: {exc}")
            return False
        return path.is_file() and path.stat().st_size > 0

def create_rough_mask(frame):
    B, G, R = cv2.split(frame)
    R = R.astype(np.float32)
    G = G.astype(np.float32)
    B = B.astype(np.float32)
    exg = (2 * G - R - B)
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX)
    _, mask = cv2.threshold(exg_norm, 150, 255, cv2.THRESH_BINARY)
    return mask.astype(np.uint8)

def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

def smooth_1d(x, k):
    x = np.asarray(x, dtype=np.float32)
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x, kernel, mode="same")

def find_center(mask):
    height, width = mask.shape
    cropped = mask[0:height//2, 0:width]
    column_sums = np.sum(cropped, axis=0)
    smooth_column_sums = smooth_1d(column_sums, 21)
    lowest_sum = np.argmin(smooth_column_sums)
    return lowest_sum

def calculate_error(mask):
    width = mask.shape[1]
    center = find_center(mask)
    # Normalized px error: positive => corridor center is left of image center
    return (width // 2 - center) / width

class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error: float, dt: float) -> float:
        if dt <= 1e-6:
            return 0.0
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

KP = 0.90
KI = 0.00
KD = 0.20
MAX_STEER = 0.25
BASE_SPEED = 0.20
ROW_CLEAR_TIME_S = 1.0
RIGHT_TURN_TIME_S = 1.2
ROW_STEP_TIME_S = 2.0
REALIGN_TIME_S = 1.0

# End-of-row detection via the side ultrasonics: when BOTH side ranges open up to
# the headland (read beyond EOR_SIDE_OPEN_M) and stay open for more than a couple
# of seconds, the crop rows have ended and we should change rows.
EOR_SIDE_OPEN_M = 0.80   # a side reading beyond this counts as "open / no crop wall"
EOR_HOLD_S = 2.0         # both sides must stay open this long to confirm end of row
# Vision fallback, used only when no ultrasonics are wired: sustained low green
# coverage in the mask also signals the corridor has opened out.
CROP_END_COVERAGE = 0.03

_pid = PID(KP, KI, KD)
_pid_last_t = None

def reset_pid():
    global _pid_last_t
    _pid.reset()
    _pid_last_t = None

def pid_control(error):
    """Return clamped steer correction from normalized row error."""
    global _pid_last_t
    now = time.monotonic()
    if _pid_last_t is None:
        dt = 0.05
    else:
        dt = now - _pid_last_t
    _pid_last_t = now

    steer = _pid.update(error, dt)
    return float(np.clip(steer, -MAX_STEER, MAX_STEER))

def apply_steering(rover: Rover, steer: float, base_speed: float = BASE_SPEED):
    left = base_speed - steer
    right = base_speed + steer
    rover.moving = True
    rover.ser.write(
        f'{{"T":1,"L":{left:.2f},"R":{right:.2f}}}\n'.encode("utf-8")
    )

class RowNavigator:
    """Continuously works the rover through the farm as a state machine.

        FOLLOWING  -- PID-follow the crop corridor until the row ends
        CHANGE_ROW -- drive straight out of the row to clear the crop end
        TURNING    -- U-turn (right, cross to next row, right, realign)
    ... then back to FOLLOWING for the next row, indefinitely.

    step() is non-blocking and runs one iteration per main-loop tick so COLMAP
    capture and the optional preview keep flowing through every state.
    """
    def __init__(self, rover: Rover, camera: Webcam, ultrasonics: "Ultrasonics" = None):
        self.rover = rover
        self.camera = camera
        self.ultrasonics = ultrasonics
        self.state = State.FOLLOWING
        self._phase_started = time.monotonic()  # when the current timed phase began
        self._turn_steps = []                    # queued (action, duration) for TURNING
        self._turn_index = 0
        self._open_since = None                  # when both sides first opened to headland
        self.last_frame = None                   # most recent frame/mask, for preview
        self.last_mask = None

    def step(self):
        if self.state == State.FOLLOWING:
            self._follow()
        elif self.state == State.CHANGE_ROW:
            self._change_row()
        elif self.state == State.TURNING:
            self._turning()

    def _follow(self):
        frame = self.camera.read()
        mask = clean_mask(create_rough_mask(frame))
        self.last_frame, self.last_mask = frame, mask

        if self._row_has_ended():
            self._enter_change_row()
            return

        steer = pid_control(calculate_error(mask))
        apply_steering(self.rover, steer)

    def _row_has_ended(self):
        """True once both sides have opened to the headland for EOR_HOLD_S seconds."""
        now = time.monotonic()
        if self.ultrasonics is not None:
            dL = self.ultrasonics.left()
            dR = self.ultrasonics.right()
            # A large reading on BOTH sides means the crop walls are gone. A None
            # (echo timeout) is treated as "not confirmed open" so a glitchy read
            # can't trip the turn on its own.
            open_now = (dL is not None and dL > EOR_SIDE_OPEN_M and
                        dR is not None and dR > EOR_SIDE_OPEN_M)
        else:
            # Vision fallback when no ultrasonics are wired.
            coverage = float(np.count_nonzero(self.last_mask)) / self.last_mask.size
            open_now = coverage < CROP_END_COVERAGE

        if not open_now:
            self._open_since = None
            return False
        if self._open_since is None:
            self._open_since = now
        return now - self._open_since >= EOR_HOLD_S

    def _enter_change_row(self):
        print("[NAV] End of row -> CHANGE_ROW")
        self.state = State.CHANGE_ROW
        self._phase_started = time.monotonic()
        self.rover.move_forward()  # keep driving straight to fully clear the row end

    def _change_row(self):
        if time.monotonic() - self._phase_started >= ROW_CLEAR_TIME_S:
            self._enter_turning()

    def _enter_turning(self):
        print("[NAV] Row cleared -> TURNING")
        self.state = State.TURNING
        # Timed U-turn into the adjacent row: turn, cross the gap, turn, realign.
        self._turn_steps = [
            (self.rover.turn_right, RIGHT_TURN_TIME_S),
            (self.rover.move_forward, ROW_STEP_TIME_S),
            (self.rover.turn_right, RIGHT_TURN_TIME_S),
            (self.rover.move_forward, REALIGN_TIME_S),
        ]
        self._turn_index = 0
        self._start_turn_step()

    def _start_turn_step(self):
        action, _ = self._turn_steps[self._turn_index]
        action()
        self._phase_started = time.monotonic()

    def _turning(self):
        _, duration = self._turn_steps[self._turn_index]
        if time.monotonic() - self._phase_started < duration:
            return
        self._turn_index += 1
        if self._turn_index >= len(self._turn_steps):
            self._enter_following()
        else:
            self._start_turn_step()

    def _enter_following(self):
        print("[NAV] Aligned -> FOLLOWING")
        self.rover.stop()
        reset_pid()
        self._open_since = None
        self.state = State.FOLLOWING

def main():
    parser = argparse.ArgumentParser(description="Autonomous row following with COLMAP capture while moving.")
    parser.add_argument("port", type=str, help="Rover serial port, e.g. /dev/ttyUSB0")
    parser.add_argument("--camera", default="auto", help="V4L2 camera device, e.g. /dev/video2, or auto")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--capture-rate", type=float, default=5.0, help="COLMAP capture rate in Hz while the rover moves.")
    parser.add_argument("--capture-width", type=int, default=1920, help="COLMAP still width.")
    parser.add_argument("--capture-height", type=int, default=1080, help="COLMAP still height.")
    parser.add_argument("--no-capture", action="store_true", help="Disable COLMAP capture.")
    parser.add_argument("--preview", action="store_true", help="Show camera/mask windows.")
    args = parser.parse_args()

    rover = Rover(args.port)
    camera = Webcam(args.camera, args.width, args.height, args.fps)
    print(f"Using camera: {camera.device}")

    ultrasonics = None
    try:
        ultrasonics = Ultrasonics()
        print("[NAV] Ultrasonic end-of-row detection enabled.")
    except Exception as exc:
        print(f"[NAV] Ultrasonics unavailable ({exc}); using vision fallback for end-of-row.")

    capturer = None
    if not args.no_capture:
        capturer = MotionCapture(
            rover,
            rate_hz=args.capture_rate,
            width=args.capture_width,
            height=args.capture_height,
        )
        capturer.start()

    reset_pid()
    navigator = RowNavigator(rover, camera, ultrasonics)
    last_state = None
    try:
        while True:
            navigator.step()
            if navigator.state != last_state:
                print(f"[NAV] State: {navigator.state.name}")
                last_state = navigator.state

            if args.preview:
                frame = navigator.last_frame if navigator.last_frame is not None else camera.read()
                mask = navigator.last_mask if navigator.last_mask is not None else clean_mask(create_rough_mask(frame))
                cv2.imshow("Camera", frame)
                cv2.imshow("Mask", mask)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # FOLLOWING is paced by the blocking camera.read(); CHANGE_ROW and
                # TURNING are pure timers, so nudge the loop to avoid a busy-spin.
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        rover.stop()
        if capturer is not None:
            capturer.stop()
        if ultrasonics is not None:
            ultrasonics.close()
        camera.close()
        rover.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
