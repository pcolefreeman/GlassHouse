# takes bin data from CSI_data made by writerCSI.py useful in collab with writer to debug specific nodes

import struct
import os
import time
import glob
import math

WATCH_DIR = r"C:\Users\19124\OneDrive\Documents\Senior_Cap\GitRepo\GlassHouse\CSI_data" # *** replace to work on new computer
PROCESSED = set()

# -------------------- KNOWN SHOUTER MACs --------------------
SHOUTER_MACS = { # MAC address/ Node ID
    "68:FE:71:90:60:A0": 1,   
    "68:FE:71:90:68:14": 2,
    "68:FE:71:90:6B:90": 3,
    # "XX:XX:XX:XX:XX:XX": 4,
}

# -------------------- FRAME FORMAT --------------------
# Header after 0xAA 0x55 magic (16 bytes):
#   ver(1) flags(1) ms(4) rssi(1) noise_floor(1) mac(6) csi_len(2)
HEADER_SIZE = 16

# -------------------- SUBCARRIER FILTERING --------------------
# Null and pilot subcarrier indices derived from actual captured data.
# These are zero-amplitude or anomalous subcarriers that carry no useful info.
NULL_SUBCARRIERS = set([
    0,                                              # DC leakage / anomalous high amplitude
    1,                                              # suspicious outlier (~11 vs neighbors ~22)
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,   # zero block 1
    64,                                             # DC null
    93, 94, 95, 96, 97, 98, 99,                    # zero block 2
])

# Subsample every Nth subcarrier from the remaining valid ones
SUBSAMPLE_N = 3

def filter_subcarriers(csi_complex):
    """
    1. Remove null/pilot subcarriers.
    2. Subsample every Nth remaining subcarrier.
    Returns filtered list of (original_index, complex_value) tuples.
    """
    valid = [(i, c) for i, c in enumerate(csi_complex) if i not in NULL_SUBCARRIERS]
    subsampled = valid[::SUBSAMPLE_N]
    return subsampled

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

    frames       = []
    skipped_macs = set()
    i            = 0

    while i < len(raw) - 2:
        # Skip debug lines starting with '#'
        if raw[i] == ord('#'):
            while i < len(raw) and raw[i] != ord('\n'):
                i += 1
            i += 1
            continue

        if raw[i] == 0xAA and raw[i+1] == 0x55:
            offset = i + 2
            if offset + HEADER_SIZE > len(raw):
                break

            # Parse 16-byte header
            # ver(1) flags(1) ms(4) rssi(1) noise_floor(1) mac(6) csi_len(2)
            timestamp   = struct.unpack_from('<I', raw, offset + 2)[0]
            rssi        = struct.unpack_from('<b', raw, offset + 6)[0]
            noise_floor = struct.unpack_from('<b', raw, offset + 7)[0]
            mac         = ':'.join(f'{b:02X}' for b in raw[offset+8:offset+14])
            csi_len     = struct.unpack_from('<H', raw, offset + 14)[0]
            header_end  = offset + HEADER_SIZE

            if header_end + csi_len > len(raw):
                i += 1
                continue

            # ---- MAC FILTER ----
            if mac not in SHOUTER_MACS:
                skipped_macs.add(mac)
                i = header_end + csi_len
                continue

            csi_bytes   = raw[header_end:header_end + csi_len]
            csi_complex = parse_csi(csi_bytes)

            # Apply null removal + subsampling
            filtered    = filter_subcarriers(csi_complex)
            sc_indices  = [idx for idx, _ in filtered]
            sc_complex  = [c   for _, c   in filtered]
            amplitudes  = [abs(c) for c in sc_complex]

            frames.append({
                'timestamp_ms': timestamp,
                'rssi_dbm':     rssi,
                'noise_floor':  noise_floor,
                'node #':       SHOUTER_MACS[mac],  # store node ID directly
                'sc_indices':   sc_indices,   # original subcarrier index for reference
                'csi_complex':  sc_complex,
                'amplitudes':   amplitudes,
            })
            i = header_end + csi_len
        else:
            i += 1

    return frames, skipped_macs

def is_file_stable(filepath, wait=1.5):
    size_before = os.path.getsize(filepath)
    time.sleep(wait)
    return size_before == os.path.getsize(filepath)

def process_file(filepath):
    print(f"\n{'='*50}")
    print(f"Processing: {os.path.basename(filepath)}")

    frames, skipped_macs = parse_bin_file(filepath)
    print(f"Frames parsed  : {len(frames)} (from known shouters)")
    if frames:
        print(f"Subcarriers    : {len(frames[0]['amplitudes'])} "
              f"(after null removal + every {SUBSAMPLE_N}rd subsampled from 128)")

    if skipped_macs:
        print(f"Skipped MACs   : {', '.join(skipped_macs)}")

    for idx, frame in enumerate(frames):
        phases = [math.atan2(c.imag, c.real) for c in frame['csi_complex']]

        print(f"  Frame {idx:4d} | [{frame['timestamp_ms']}ms] "
              f"RSSI={frame['rssi_dbm']}dBm | "
              f"NF={frame['noise_floor']}dBm | "
              f"Node={frame['node #']} | "
              f"Subcarriers={len(frame['amplitudes'])}")

        for sc_idx, (orig_idx, amp, phase) in enumerate(zip(frame['sc_indices'], frame['amplitudes'], phases)):
            print(f"    SC {sc_idx:3d} (orig={orig_idx:3d}) | Amp={amp:7.4f} | Phase={phase:8.4f} rad")

# ---- Watch loop ----
print(f"Watching {WATCH_DIR} for new .bin files...")
print(f"Filtering to MACs: {', '.join(SHOUTER_MACS)}")
print(f"Null subcarriers removed: {len(NULL_SUBCARRIERS)} | Subsample: every {SUBSAMPLE_N}rd\n")

while True:
    bin_files        = sorted(glob.glob(os.path.join(WATCH_DIR, "*.bin")))
    files_to_process = bin_files[:-1]

    for filepath in files_to_process:
        if filepath not in PROCESSED:
            if is_file_stable(filepath):
                process_file(filepath)
                PROCESSED.add(filepath)

    time.sleep(2)