# Copied and made changes to the writerCSI.py file
# Got the ESPs to not need flashing after this code is stopped via keyboard interrupt

import serial
import os
import time
from datetime import datetime
import glob
import subprocess

binFolder = r"C:\Users\19124\OneDrive\Documents\Senior_Cap\GitRepo\GlassHouse\CSIBin3_3" # *** replace to work on new computer

# *** Linux lines -> uncomment to work with linux OS
# ----- removes old .bin files on startup of writer code ------
# result = subprocess.run( # removes old .bin files on startup of writer code
#     f'find "{binFolder}" -name "* .bin" -type f -delete',
#     shell=True,
#     capture_output=True,
#     text=True
# )
#
# if result.returncode == 0:
#     print("All .bin files deleted successfully.")
# else:
#     print(f"Error: {result.stderr}")
# -------------------------------------------------------------

# *** Windows OS lines -> comment out if linux
# ----- removes old .bin files on startup of writer code ------
bin_files = glob.glob(os.path.join(binFolder, "*.bin")) 

if not bin_files:
    print("No .bin files found.")
else:
    for file in bin_files:
        os.remove(file)
        print(f"Deleted: {file}")
    print(f"\nDone! {len(bin_files)} file(s) deleted.")
# -------------------------------------------------------------

PORT = "COM3" # *** change to your port
BAUD = 921600
OUTPUT_DIR = r"C:\Users\19124\OneDrive\Documents\Senior_Cap\GitRepo\GlassHouse\CSIBin3_3" # *** replace to work on new computer
INTERVAL_SECONDS = 15

def get_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, f"csi_{timestamp}.bin")

# -------------------- SERIAL SETUP --------------------
ser = serial.Serial()
ser.port = PORT
ser.baudrate = BAUD
ser.timeout = 1

# Prevent reset on open/close
ser.dtr = False
ser.rts = False

ser.open()

# Give ESP32 a moment after port opens (without reset it's already running)
time.sleep(0.5)

# Flush any stale bytes in buffer
ser.reset_input_buffer()

print("Waiting for ESP32 ready signal (or already running)...")

# Wait for ready signal, but don't hang forever —
# if the ESP32 is already mid-run, it won't re-send LISTENER_AP_READY
READY_TIMEOUT = 5.0
start = time.time()
ready = False

while time.time() - start < READY_TIMEOUT:
    line = ser.readline()
    if b"LISTENER_AP_READY" in line:
        print("ESP32 ready signal received, starting capture...")
        ready = True
        break

if not ready:
    print("No ready signal received — ESP32 likely already running. Starting capture anyway...")

# -------------------- CAPTURE LOOP --------------------
try:
    while True:
        filename = get_filename()
        start_time = time.time()

        with open(filename, "wb") as f:
            print(f"Recording to {filename}")
            while time.time() - start_time < INTERVAL_SECONDS:
                try:
                    data = ser.read(512)
                    if data:
                        f.write(data)
                except serial.SerialException as e:
                    print(f"Serial error: {e}")
                    break

        print(f"Saved: {filename}")

except KeyboardInterrupt:
    print("\nCapture stopped by user.")

finally:
    # Close without triggering reset
    ser.dtr = False
    ser.rts = False
    ser.close()
    print("Serial port closed cleanly.")