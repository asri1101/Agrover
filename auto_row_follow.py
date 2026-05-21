import serial
import argparse
import threading
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
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    return mask

def follow_row(rover: Rover):
    pass

def next_row(rover: Rover):
    pass

def main():
    mask = create_rough_mask()
    smoothed = clean_mask(mask)
    cv2.imshow("Mask", smoothed.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    