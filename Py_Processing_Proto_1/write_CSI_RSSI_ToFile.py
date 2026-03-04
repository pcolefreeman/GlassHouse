import serial
import os
from datetime import datetime

PORT = "COM3"   # change if needed
BAUD = 912600
OUTPUT_DIR = r"C:\Users\19124\OneDrive\Documents\Senior_Cap\GitRepo\GlassHouse\TestcsiCapture"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Timestamped filename so each run gets its own file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(OUTPUT_DIR, f"csi_{timestamp}.bin")

with serial.Serial(PORT, BAUD, timeout=1) as ser:
    # Wait for ESP32 ready signal
    print("Waiting for ESP32...")
    while True:
        line = ser.readline()
        if b"LISTENER_AP_READY" in line:
            print("ESP32 ready, recording...")
            break

    with open(output_file, "wb") as f:
        try:
            while True:
                data = ser.read(512)
                if data:
                    f.write(data)
        except KeyboardInterrupt:
            print(f"\nSaved to {output_file}")