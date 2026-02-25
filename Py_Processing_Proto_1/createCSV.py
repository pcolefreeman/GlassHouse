import serial
import csv
import time

# 1. Setup Serial (Change 'COM3' or '/dev/ttyUSB0' to match your Pi)
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
file_name = "csi_training_data.csv"

print(f"Logging started. Saving to {file_name}...")

with open(file_name, mode='a', newline='') as f:
    writer = csv.writer(f)
    
    # Optional: Write a header if the file is new
    # writer.writerow(["timestamp", "node_id", "rssi", "csi_data"])

    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            
            if line.startswith("BRAIN_DATA"):
                # Split the string into a list
                parts = line.split(',')
                
                # parts[0] = "BRAIN_DATA"
                # parts[1] = MAC Address
                # parts[2] = RSSI 
                # parts[3] = The long string of CSI numbers
                
                timestamp = time.time()
                writer.writerow([timestamp, parts[1], parts[2], parts[3]])
                f.flush() # Forces the data onto the SD card immediately
                
        except KeyboardInterrupt:
            print("Logging stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")