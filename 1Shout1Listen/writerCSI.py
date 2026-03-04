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
