import serial
import time
import argparse
import threading

def read_serial():
    while True:
        data = ser.readline().decode('utf-8')
        if data:
            print(f"Received: {data}", end='')

def main():
    global ser
    parser = argparse.ArgumentParser(description='Serial JSON Communication')
    parser.add_argument('port', type=str, help='Serial port name (e.g., COM1 or /dev/ttyUSB0)')

    args = parser.parse_args()

    ser = serial.Serial(args.port, baudrate=115200, timeout=0.2, dsrdtr=None)
    ser.setRTS(False)
    ser.setDTR(False)

    serial_recv_thread = threading.Thread(target=read_serial)
    serial_recv_thread.daemon = True
    serial_recv_thread.start()

    try:
        while True:
            msg = b'{"T":1,"L":0.2,"R":0.2}\n'
            ser.write(msg)
            print("Sent:", msg)
            time.sleep(0.1)
    except KeyboardInterrupt:
        msg = b'{"T":1,"L":0,"R":0}\n'
        ser.write(msg)
        print("Sent:", msg)
    finally:
        ser.close()
        print("Closed serial port")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    finally:
        print("Exiting program")
        ser.close()
        print("Closed serial port")
