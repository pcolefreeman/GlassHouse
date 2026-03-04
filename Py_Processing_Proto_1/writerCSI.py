import serial
import os
import time
from datetime import datetime

PORT = "COM3"          # change to your port
BAUD = 460800
OUTPUT_DIR = r"C:\Users\19124\OneDrive\Documents\Senior_Cap\GitRepo\GlassHouse\CSIBin3_3"
INTERVAL_SECONDS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, f"csi_{timestamp}.bin")

with serial.Serial(PORT, BAUD, timeout=1) as ser:
    print("Waiting for ESP32...")
    while True:
        line = ser.readline()
        if b"LISTENER_AP_READY" in line:
            print("ESP32 ready, starting capture...")
            break

    while True:
        filename = get_filename()
        start_time = time.time()

        with open(filename, "wb") as f:
            print(f"Recording to {filename}")
            while time.time() - start_time < INTERVAL_SECONDS:
                data = ser.read(512)
                if data:
                    f.write(data)

        print(f"Saved: {filename}")