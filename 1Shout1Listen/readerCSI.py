import struct
import os
import time
import glob

WATCH_DIR = r"C:\Users\19124\OneDrive\Documents\Senior_Cap\GitRepo\GlassHouse\CSIBin3_3" # *** replace to work on new computer
PROCESSED = set()

def parse_csi(csi_bytes):
    csi = []
    for j in range(0, len(csi_bytes) - 1, 2):
        imag = struct.unpack('b', bytes([csi_bytes[j]]))[0]
        real = struct.unpack('b', bytes([csi_bytes[j+1]]))[0]
        csi.append(complex(real, imag))
    return csi

def parse_bin_file(filepath):
    with open(filepath, "rb") as f:
        raw = f.read()

    frames = []
    i = 0
    while i < len(raw) - 2:
        if raw[i] == 0xAA and raw[i+1] == 0x55:
            offset = i + 2
            if offset + 15 > len(raw):
                break

            timestamp = struct.unpack_from('<I', raw, offset+2)[0]
            rssi      = struct.unpack_from('<b', raw, offset+6)[0]
            mac       = raw[offset+7:offset+13]
            csi_len   = struct.unpack_from('<H', raw, offset+13)[0]
            header_end = offset + 15

            if header_end + csi_len + 2 > len(raw):
                i += 1
                continue

            csi_bytes = raw[header_end:header_end + csi_len]

            mac_str     = ':'.join(f'{b:02X}' for b in mac)
            csi_complex = parse_csi(csi_bytes)
            amplitudes  = [abs(c) for c in csi_complex]

            frames.append({
                'timestamp_ms': timestamp,
                'rssi_dbm':     rssi,
                'mac':          mac_str,
                'csi_complex':  csi_complex,
                'amplitudes':   amplitudes,
            })
            i = header_end + csi_len + 2
        else:
            i += 1

    return frames

def is_file_stable(filepath, wait=1.5):
    """Returns True if file size hasn't changed in `wait` seconds."""
    size_before = os.path.getsize(filepath)
    time.sleep(wait)
    size_after = os.path.getsize(filepath)
    return size_before == size_after

def process_file(filepath):
    print(f"\n{'='*50}")
    print(f"Processing: {os.path.basename(filepath)}")
    frames = parse_bin_file(filepath)
    print(f"Frames parsed: {len(frames)}")

    for idx, frame in enumerate(frames):
        avg_amp = sum(frame['amplitudes']) / len(frame['amplitudes']) if frame['amplitudes'] else 0
        print(f"  Frame {idx:4d} | [{frame['timestamp_ms']}ms] "
              f"RSSI={frame['rssi_dbm']}dBm | "
              f"MAC={frame['mac']} | "
              f"Avg amplitude={avg_amp:.2f}")

# ---- Watch loop ----
print(f"Watching {WATCH_DIR} for new .bin files...")
while True:
    bin_files = sorted(glob.glob(os.path.join(WATCH_DIR, "*.bin")))

    # Skip the most recent file — it may still be being written to by the writer
    files_to_process = bin_files[:-1]

    for filepath in files_to_process:
        if filepath not in PROCESSED:
            if is_file_stable(filepath):
                process_file(filepath)
                PROCESSED.add(filepath)

    time.sleep(2)