# Agrover
Ag Robot for fruit localization

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