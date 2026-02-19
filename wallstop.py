import serial
import time
import argparse
import threading
import pigpio

TRIG = 24
ECHO = 23
STOP_DISTANCE = 0.5  # meters

ser = None
pi = None

def read_serial():
    while True:
        try:
            data = ser.readline().decode('utf-8', errors='ignore')
            if data:
                print(f"Received: {data}", end='')
        except Exception as e:
            print(f"Serial read error: {e}")

def get_distance():
    pi.gpio_trigger(TRIG, 10, 1)

    start = time.monotonic()
    while pi.read(ECHO) == 0:
        if time.monotonic() - start > 0.03:
            return None

    t0 = time.monotonic()
    while pi.read(ECHO) == 1:
        if time.monotonic() - t0 > 0.03:
            return None

    t1 = time.monotonic()

    pulse = t1 - t0
    distance = pulse * 343 / 2  # meters
    return distance

def main():
    global ser, pi  # IMPORTANT

    parser = argparse.ArgumentParser(description='Serial JSON + Ultrasonic stop')
    parser.add_argument('port', type=str, help='Serial port name (e.g., /dev/ttyUSB0)')
    args = parser.parse_args()

    ser = serial.Serial(args.port, baudrate=115200, timeout=0.2, dsrdtr=None)
    ser.setRTS(False)
    ser.setDTR(False)

    serial_recv_thread = threading.Thread(target=read_serial, daemon=True)
    serial_recv_thread.start()

    pi = pigpio.pi()
    if not pi.connected:
        print("Failed to connect to pigpio. Run: sudo systemctl start pigpiod")
        return

    pi.set_mode(TRIG, pigpio.OUTPUT)
    pi.set_mode(ECHO, pigpio.INPUT)
    pi.write(TRIG, 0)
    time.sleep(0.1)

    while True:
        d = get_distance()

        # Safety: if we can't read distance, STOP.
        if d is None:
            msg = b'{"T":1,"L":0,"R":0}\n'
            print("Distance: None -> STOP")
        else:
            print(f"{d:.2f} m")
            if d < STOP_DISTANCE:
                msg = b'{"T":1,"L":0,"R":0}\n'
            else:
                msg = b'{"T":1,"L":0.2,"R":0.2}\n'

        ser.write(msg)
        time.sleep(0.05)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    finally:
        print("Exiting program")
        try:
            if ser:
                ser.write(b'{"T":1,"L":0,"R":0}\n')  # stop on exit
                ser.close()
        except Exception:
            pass
        try:
            if pi:
                pi.stop()
        except Exception:
            pass
        print("Closed serial port")
