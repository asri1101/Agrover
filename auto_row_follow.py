import serial
import argparse
import threading
import time
from enum import Enum, auto
import cv2
import numpy as np
from PIL import Image

"""
This is a file that establishes a class RoverSerial and writes instructions for its autonomous navigation
around rows of a farm.
"""

class Rover:
    def __init__(self, port: str):
        parser = argparse.ArgumentParser(description='Serial JSON Communication')
        parser.add_argument('port', type=str, help='Serial port name (e.g., COM1 or /dev/ttyUSB0)')

        args = parser.parse_args()

        self.ser = serial.Serial(args.port, baudrate=115200, dsrdtr=None)
        self.ser.setRTS(False)
        self.ser.setDTR(False)

        self._thread = threading.Thread(target=self.read_serial)
        self._thread.daemon = True
        self._thread.start()
    def start(self):
        self._thread.start()
    def move_forward(self):
        self.ser.write(b'{"T":1,"L":0.2,"R":0.2}\n')
    def stop(self):
        self.ser.write(b'{"T":1,"L":0,"R":0}\n')
    def turn_left(self):
        self.ser.write(b'{"T":1,"L":0.2,"R":-0.2}\n')
    def turn_right(self):
        self.ser.write(b'{"T":1,"L":-0.2,"R":0.2}\n')
    def stop(self):
        self.ser.write(b'{"T":1,"L":0,"R":0}\n')
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

def create_rough_mask():
    filename = "applerow.jpg"
    with Image.open(filename) as image:
        image.load()
    R, G, B = image.split()
    R = np.array(R).astype(np.float32)
    G = np.array(G).astype(np.float32)
    B = np.array(B).astype(np.float32)
    exg = (2 * G - R - B)
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX)
    _, mask = cv2.threshold(exg_norm, 150, 255, cv2.THRESH_BINARY)
    return mask

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
    rover.ser.write(
        f'{{"T":1,"L":{left:.2f},"R":{right:.2f}}}\n'.encode("utf-8")
    )

def follow_row(rover: Rover):
    mask = clean_mask(create_rough_mask())
    error = calculate_error(mask)
    steer = pid_control(error)
    apply_steering(rover, steer)

def next_row(rover: Rover):
    rover.stop()
    time.sleep(0.2)

    rover.move_forward()
    time.sleep(ROW_CLEAR_TIME_S)
    rover.stop()
    time.sleep(0.1)

    rover.turn_right()
    time.sleep(RIGHT_TURN_TIME_S)
    rover.stop()
    time.sleep(0.1)

    rover.move_forward()
    time.sleep(ROW_STEP_TIME_S)
    rover.stop()
    time.sleep(0.1)

    rover.turn_right()
    time.sleep(RIGHT_TURN_TIME_S)
    rover.stop()
    time.sleep(0.1)

    rover.move_forward()
    time.sleep(REALIGN_TIME_S)
    rover.stop()
    reset_pid()

def main():
    mask = create_rough_mask()
    smoothed = clean_mask(mask)
    cv2.imshow("Mask", smoothed.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
