import struct
import os
import time
import glob
import csv
import math
import numpy as np
from datetime import datetime

# -------------------- ROOM CONFIG --------------------
ROOM_WIDTH_FT  = 24.0
ROOM_HEIGHT_FT = 24.0
GRID_COLS      = 3
GRID_ROWS      = 3

# -------------------- CAPTURE CONFIG --------------------
WATCH_DIR        = r"C:\Users\19124\OneDrive\Documents\Senior_Cap\GitRepo\GlassHouse\CSI_data"   # *** change to your path
OUTPUT_CSV       = r"C:\Users\19124\OneDrive\Documents\Senior_Cap\GitRepo\GlassHouse\CSI_data\training_data.csv" # use path from above line add "\training_data.csv"
BUCKET_MS        = 50
MIN_FRAMES       = 2

# -------------------- SHOUTER MACs --------------------
# dictionary with MAC / shouter ID
SHOUTER_MACS = {
    "68:FE:71:90:60:A0": 1,
    # "68:FE:71:90:68:14": 2,
    # "68:FE:71:90:6B:90": 3,
    # "XX:XX:XX:XX:XX:XX": 4,
}

NUM_SHOUTERS = 4
SUBCARRIERS  = 128  # 256 byte CSI / 2 bytes per complex sample

# -------------------- FRAME FORMAT --------------------
# Header after 0xAA 0x55 magic (16 bytes):
# ver(1) flags(1) ms(4) rssi(1) noise_floor(1) mac(6) csi_len(2)
HEADER_SIZE = 16
MAGIC       = (0xAA, 0x55)

# -------------------- GRID HELPERS --------------------
def build_zone_map(cols, rows, width_ft, height_ft):
    zone_w = width_ft  / cols
    zone_h = height_ft / rows
    zones  = {}
    for r in range(rows):
        for c in range(cols):
            zone_id = r * cols + c + 1
            zones[zone_id] = {
                "label":   f"zone_{zone_id}",
                "row":     r + 1,
                "col":     c + 1,
                "x_start": round(c * zone_w,       2),
                "x_end":   round((c + 1) * zone_w, 2),
                "y_start": round(r * zone_h,        2),
                "y_end":   round((r + 1) * zone_h,  2),
            }
    return zones

def print_grid(zones, cols, rows):
    print("\n--- Zone Map ---")
    print(f"Room: {ROOM_WIDTH_FT}ft x {ROOM_HEIGHT_FT}ft  |  Grid: {cols}x{rows}\n")
    for r in range(1, rows + 1):
        row_str = ""
        for c in range(1, cols + 1):
            zid = (r - 1) * cols + c
            z   = zones[zid]
            row_str += f"[Zone {zid:2d} ({z['x_start']:.0f}-{z['x_end']:.0f}ft, {z['y_start']:.0f}-{z['y_end']:.0f}ft)]  "
        print(row_str)
    print()

# -------------------- PARSE HELPERS --------------------
def parse_csi_bytes(csi_bytes):
    """
    Parse raw CSI bytes into complex subcarrier values.
    ESP32 stores pairs as [imag, real] per subcarrier.
    """
    csi = []
    for j in range(0, len(csi_bytes) - 1, 2):
        imag = struct.unpack('b', bytes([csi_bytes[j]]))[0]
        real = struct.unpack('b', bytes([csi_bytes[j+1]]))[0]
        csi.append(complex(real, imag))
    return csi

def compute_snr(amplitudes, noise_floor_dbm):
    """
    Per-subcarrier SNR in dB.
    noise_floor is in dBm (negative int8 from ESP32, e.g. -92).
    SNR = 20*log10(amplitude) - noise_floor_dbm
    Clamps amplitude to avoid log(0).
    """
    noise_dbm = float(noise_floor_dbm)
    snr = []
    for amp in amplitudes:
        amp_clamped = max(amp, 1e-6)
        amp_dbm     = 20.0 * math.log10(amp_clamped)
        snr.append(round(amp_dbm - noise_dbm, 4))
    return snr

def sanitize_phase(phases):
    """
    Unwrap phase discontinuities across subcarriers.
    np.unwrap removes jumps larger than pi, making phase
    continuous and more useful as an ML feature.
    """
    return list(np.unwrap(phases))

def phase_difference(phases):
    """
    Difference between adjacent subcarriers.
    Removes Carrier Frequency Offset (CFO) entirely — the random
    per-frame phase rotation cancels out, leaving only the stable
    channel multipath signature.
    Returns N-1 values for N input phases.
    """
    unwrapped = np.unwrap(phases)
    return list(np.diff(unwrapped))
def normalize_amplitude(amplitudes):
    """
    Per-frame min-max normalization of amplitudes to [0, 1].
    Removes absolute magnitude differences caused by distance,
    leaving only the relative shape across subcarriers.
    """
    arr     = np.array(amplitudes, dtype=float)
    a_min   = arr.min()
    a_max   = arr.max()
    rng     = a_max - a_min
    if rng < 1e-9:
        return list(np.zeros_like(arr))
    return list((arr - a_min) / rng)

def extract_features(csi_complex, rssi, noise_floor):
    """
    Extract all features from one CSI frame.
    Returns dict with raw and processed features.
    """
    amplitudes = [abs(c)                          for c in csi_complex]
    phases     = [math.atan2(c.imag, c.real)      for c in csi_complex]

    return {
        "amplitudes":      amplitudes,
        "phases":          sanitize_phase(phases),
        "phase_diff":      phase_difference(phases),
        "amp_normalized":  normalize_amplitude(amplitudes),
        "snr":             compute_snr(amplitudes, noise_floor),
        "rssi":            rssi,
        "noise_floor":     noise_floor,
    }

def parse_bin_file(filepath):
    """
    Parse a .bin file into a list of frames.
    Updated for 17-byte header including noise_floor and antenna.
    """
    with open(filepath, "rb") as f:
        raw = f.read()

    frames = []
    i      = 0

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
            hdr_end     = offset + HEADER_SIZE

            if hdr_end + csi_len > len(raw):
                i += 1
                continue

            if mac not in SHOUTER_MACS:
                i = hdr_end + csi_len
                continue

            csi_bytes   = raw[hdr_end:hdr_end + csi_len]
            csi_complex = parse_csi_bytes(csi_bytes)
            features    = extract_features(csi_complex, rssi, noise_floor)

            frames.append({
                "timestamp_ms":  timestamp,
                "mac":           mac,
                "shouter_id":    SHOUTER_MACS[mac],
                "amplitudes":    features["amplitudes"],
                "phases":        features["phases"],
                "phase_diff":    features["phase_diff"],
                "amp_normalized":features["amp_normalized"],
                "snr":           features["snr"],
                "rssi":          features["rssi"],
                "noise_floor":   features["noise_floor"],
            })
            i = hdr_end + csi_len
        else:
            i += 1

    return frames

# -------------------- BUCKETING --------------------
def bucket_frames(frames, bucket_ms=BUCKET_MS):
    """
    Group frames into time buckets. Each bucket = one CSV row.
    Frames from the same shouter within a bucket are averaged.
    """
    if not frames:
        return []

    frames  = sorted(frames, key=lambda f: f["timestamp_ms"])
    t_start = frames[0]["timestamp_ms"]

    buckets = {}
    for frame in frames:
        bid = (frame["timestamp_ms"] - t_start) // bucket_ms
        if bid not in buckets:
            buckets[bid] = {s: [] for s in range(1, NUM_SHOUTERS + 1)}
        buckets[bid][frame["shouter_id"]].append(frame)

    samples = []
    for bid in sorted(buckets.keys()):
        bucket   = buckets[bid]
        t_bucket = t_start + bid * bucket_ms
        active   = sum(1 for s in bucket if len(bucket[s]) > 0)

        if active < MIN_FRAMES:
            continue

        sample = {"timestamp_ms": t_bucket}

        for sid in range(1, NUM_SHOUTERS + 1):
            frames_in = bucket[sid]
            px        = f"s{sid}"

            if frames_in:
                avg_amp       = np.mean([f["amplitudes"]     for f in frames_in], axis=0)
                avg_phase     = np.mean([f["phases"]         for f in frames_in], axis=0)
                avg_phase_diff= np.mean([f["phase_diff"]     for f in frames_in], axis=0)
                avg_amp_norm  = np.mean([f["amp_normalized"] for f in frames_in], axis=0)
                avg_snr       = np.mean([f["snr"]            for f in frames_in], axis=0)
                avg_rssi      = np.mean([f["rssi"]           for f in frames_in])
                avg_nf        = np.mean([f["noise_floor"]    for f in frames_in])

                for sc in range(len(avg_amp)):
                    sample[f"{px}_amp_{sc}"]      = round(float(avg_amp[sc]),      4)
                    sample[f"{px}_amp_norm_{sc}"] = round(float(avg_amp_norm[sc]), 4)
                    sample[f"{px}_phase_{sc}"]    = round(float(avg_phase[sc]),    4)
                    sample[f"{px}_snr_{sc}"]      = round(float(avg_snr[sc]),      4)

                # phase_diff has N-1 values
                for sc in range(len(avg_phase_diff)):
                    sample[f"{px}_pdiff_{sc}"] = round(float(avg_phase_diff[sc]), 4)

                sample[f"{px}_rssi"]        = round(float(avg_rssi), 2)
                sample[f"{px}_noise_floor"] = round(float(avg_nf),   2)

            else:
                # Shouter not heard — fill with NaN
                for sc in range(SUBCARRIERS):
                    sample[f"{px}_amp_{sc}"]      = float("nan")
                    sample[f"{px}_amp_norm_{sc}"] = float("nan")
                    sample[f"{px}_phase_{sc}"]    = float("nan")
                    sample[f"{px}_snr_{sc}"]      = float("nan")
                for sc in range(SUBCARRIERS - 1):
                    sample[f"{px}_pdiff_{sc}"]    = float("nan")
                sample[f"{px}_rssi"]        = float("nan")
                sample[f"{px}_noise_floor"] = float("nan")

        samples.append(sample)

    return samples

# -------------------- CSV --------------------
def build_csv_header():
    header = ["timestamp_ms", "label", "zone_id", "grid_row", "grid_col"]
    for s in range(1, NUM_SHOUTERS + 1):
        px = f"s{s}"
        for sc in range(SUBCARRIERS):
            header.append(f"{px}_amp_{sc}")
        for sc in range(SUBCARRIERS):
            header.append(f"{px}_amp_norm_{sc}")
        for sc in range(SUBCARRIERS):
            header.append(f"{px}_phase_{sc}")
        for sc in range(SUBCARRIERS - 1):      # N-1 differences
            header.append(f"{px}_pdiff_{sc}")
        for sc in range(SUBCARRIERS):
            header.append(f"{px}_snr_{sc}")
        header.append(f"{px}_rssi")
        header.append(f"{px}_noise_floor")
    return header

def append_samples_to_csv(samples, zone_info, zone_id, csv_path, header):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        for sample in samples:
            row              = {k: sample.get(k, float("nan")) for k in header}
            row["label"]     = zone_info["label"]
            row["zone_id"]   = zone_id
            row["grid_row"]  = zone_info["row"]
            row["grid_col"]  = zone_info["col"]
            writer.writerow(row)

# -------------------- FILE WATCHER --------------------
def is_file_stable(filepath, wait=1.5):
    size_before = os.path.getsize(filepath)
    time.sleep(wait)
    return size_before == os.path.getsize(filepath)

def process_file(filepath, zone_info, zone_id, header):
    if os.path.getsize(filepath) < 20:
        print(f"  Skipping {os.path.basename(filepath)} — too small.")
        return 0

    frames  = parse_bin_file(filepath)
    samples = bucket_frames(frames)

    if not samples:
        print(f"  No valid samples in {os.path.basename(filepath)}")
        return 0

    append_samples_to_csv(samples, zone_info, zone_id, OUTPUT_CSV, header)
    print(f"  Wrote {len(samples)} samples  |  "
          f"frames={len(frames)}  |  "
          f"file={os.path.basename(filepath)}")
    return len(samples)

# -------------------- MAIN --------------------
if __name__ == "__main__":
    os.makedirs(WATCH_DIR, exist_ok=True)

    zones  = build_zone_map(GRID_COLS, GRID_ROWS, ROOM_WIDTH_FT, ROOM_HEIGHT_FT)
    header = build_csv_header()

    print_grid(zones, GRID_COLS, GRID_ROWS)

    print("=== CSI Training Data Collector ===")
    print(f"Output CSV   : {OUTPUT_CSV}")
    print(f"Bucket size  : {BUCKET_MS}ms")
    print(f"Subcarriers  : {SUBCARRIERS} per shouter")
    print(f"Features     : amp, amp_normalized, phase (unwrapped), phase_diff (CFO-removed), SNR per subcarrier + RSSI, noise_floor per shouter")
    print(f"Shouters     : {len(SHOUTER_MACS)} active of {NUM_SHOUTERS} total\n")

    print("Select zone before each capture session.")
    print("Available zones:")
    for zid, z in zones.items():
        print(f"  {zid}: {z['label']}  "
              f"(row {z['row']}, col {z['col']})  "
              f"x={z['x_start']}-{z['x_end']}ft, y={z['y_start']}-{z['y_end']}ft")

    while True:
        try:
            zone_input = int(input("\nEnter zone number (or 0 to quit): "))
            if zone_input == 0:
                print("Exiting.")
                break
            if zone_input not in zones:
                print(f"Invalid zone. Choose from {list(zones.keys())}")
                continue

            current_zone = zones[zone_input]
            print(f"\nRecording for zone {zone_input}: {current_zone['label']}")
            print(f"Watching {WATCH_DIR} for new .bin files...")
            print("Press Ctrl+C to stop and select a new zone.\n")

            processed = set()

            try:
                while True:
                    bin_files        = sorted(glob.glob(os.path.join(WATCH_DIR, "*.bin")))
                    files_to_process = bin_files[:-1]

                    for filepath in files_to_process:
                        if filepath not in processed:
                            if is_file_stable(filepath):
                                process_file(filepath, current_zone, zone_input, header)
                                processed.add(filepath)

                    time.sleep(2)

            except KeyboardInterrupt:
                print(f"\nStopped recording for {current_zone['label']}.")
                print("Select next zone or press 0 to quit.\n")

        except ValueError:
            print("Please enter a valid zone number.")