import struct
import os
import sys
import time
import glob
import csv
import json
import math
import numpy as np
from datetime import datetime

# ================================================================================
# BASE OUTPUT DIRECTORY — all session folders created here automatically
# ================================================================================
BASE_OUTPUT_DIR = r"C:\Users\19124\OneDrive\Documents\Senior_Cap\GitRepo\GlassHouse\training_data" # ***  change for your computer

# -------------------- CAPTURE CONFIG --------------------
WATCH_DIR   = r"C:\Users\19124\OneDrive\Documents\Senior_Cap\GitRepo\GlassHouse\CSI_data"   #  *** directory to watch for incoming .bin files
BUCKET_MS   = 50
MIN_FRAMES  = 2

# -------------------- ROOM CONFIG (defaults, overridden by session prompt) ------
ROOM_WIDTH_FT  = 24.0
ROOM_LENGTH_FT = 24.0
GRID_COLS      = 3
GRID_ROWS      = 3

# -------------------- SHOUTER MACs --------------------
SHOUTER_MACS = {
    "68:FE:71:90:60:A0": 1,
    "68:FE:71:90:68:14": 2,
    "68:FE:71:90:6B:90": 3,
    # "XX:XX:XX:XX:XX:XX": 4,
}

NUM_SHOUTERS = 4 # *** change for system 
SUBCARRIERS  = 128  # 256 byte CSI / 2 bytes per complex sample

# -------------------- FRAME FORMAT --------------------
# Header after 0xAA 0x55 magic (16 bytes):
# ver(1) flags(1) ms(4) rssi(1) noise_floor(1) mac(6) csi_len(2)
HEADER_SIZE = 16
MAGIC       = (0xAA, 0x55)

# ================================================================================
# GRID STATE DESCRIPTORS  (per test process doc Section 3.1)
# ================================================================================
# Maps the user-facing menu option to the folder-name token used in the doc.
GRID_STATES = {
    "Occupied": "Occupied",
    "Standing": "Standing",
    "Seated":   "Seated",
    "Moving":   "Moving",
}

# ================================================================================
# SESSION METADATA  — collected once at startup
# ================================================================================
class SessionMeta:
    """Holds operator-level info that does not change within a session."""
    def __init__(self):
        self.operator    = ""
        self.subject_id  = ""
        self.room_width  = ROOM_WIDTH_FT
        self.room_length = ROOM_LENGTH_FT
        self.date        = datetime.now().strftime("%Y-%m-%d")

    def prompt(self):
        print("\n=== Session Setup ===")
        self.operator    = input("Operator name       : ").strip() or "Unknown"
        self.subject_id  = input("Subject ID          : ").strip() or "Subject_A"
        w = input(f"Room width  (ft) [{self.room_width:.0f}] : ").strip()
        h = input(f"Room length (ft) [{self.room_length:.0f}] : ").strip()
        if w:
            self.room_width  = float(w)
        if h:
            self.room_length = float(h)
        print(f"\nSession started  —  Operator: {self.operator}  |  Subject: {self.subject_id}"
              f"  |  Room: {self.room_width:.0f}x{self.room_length:.0f}ft"
              f"  |  Date: {self.date}\n")

# ================================================================================
# FOLDER / METADATA HELPERS  (per test process doc Section 3)
# ================================================================================
def build_grid_state_token(zone_input, state_key):
    """
    Build the [GridState] token used in the folder name.
      zone_input == 0  →  'Empty'
      zone_input >= 1  →  e.g. 'Grid5Occupied', 'Grid3Seated', etc.
    """
    if zone_input == 0:
        return "Empty"
    return f"Grid{zone_input}{state_key}"

def build_folder_name(session: SessionMeta, grid_state_token: str,
                      duration_s: int, run_index: int) -> str:
    """
    Produces:  <WxH>Room_<GridState>_<Duration>Seconds_Run<NN>
    e.g.       24x24Room_Grid5Occupied_10Seconds_Run03
    """
    room_token = f"{session.room_width:.0f}x{session.room_length:.0f}Room"
    run_token  = f"Run{run_index:02d}"
    return f"{room_token}_{grid_state_token}_{duration_s}Seconds_{run_token}"

def build_output_folder(session: SessionMeta, grid_state_token: str,
                        duration_s: int, run_index: int) -> str:
    """Create the output folder on disk and return its path."""
    folder_name = build_folder_name(session, grid_state_token, duration_s, run_index)
    folder_path = os.path.join(BASE_OUTPUT_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def write_metadata_json(folder_path: str, session: SessionMeta,
                        grid_state_token: str, duration_s: int,
                        run_index: int, posture: str, zone_id: int):
    """Write metadata.json per the schema in Section 6.3 of the test process doc."""
    meta = {
        "room_width_ft":    session.room_width,
        "room_length_ft":   session.room_length,
        "grid_state":       grid_state_token,
        "duration_seconds": duration_s,
        "run_index":        run_index,
        "date":             session.date,
        "operator":         session.operator,
        "subject_id":       session.subject_id,
        "posture":          posture,
        "zone_id":          zone_id,
        "shouters":         [f"ESP32_S{sid}" for sid in range(1, NUM_SHOUTERS + 1)],
        "notes":            "",
    }
    meta_path = os.path.join(folder_path, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

# ================================================================================
# GRID HELPERS
# ================================================================================
def build_zone_map(cols, rows, width_ft, length_ft):
    zone_w = width_ft  / cols
    zone_h = length_ft / rows
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

def print_grid(zones, cols, rows, width_ft, length_ft):
    print("\n--- Zone Map ---")
    print(f"Room: {width_ft:.0f}ft x {length_ft:.0f}ft  |  Grid: {cols}x{rows}\n")
    for r in range(1, rows + 1):
        row_str = ""
        for c in range(1, cols + 1):
            zid = (r - 1) * cols + c
            z   = zones[zid]
            row_str += (f"[Zone {zid:2d} "
                        f"({z['x_start']:.0f}-{z['x_end']:.0f}ft, "
                        f"{z['y_start']:.0f}-{z['y_end']:.0f}ft)]  ")
        print(row_str)
    print()

# ================================================================================
# PARSE HELPERS
# ================================================================================
def parse_csi_bytes(csi_bytes):
    """Parse raw CSI bytes into complex subcarrier values.
    ESP32 stores pairs as [imag, real] per subcarrier."""
    csi = []
    for j in range(0, len(csi_bytes) - 1, 2):
        imag = struct.unpack('b', bytes([csi_bytes[j]]))[0]
        real = struct.unpack('b', bytes([csi_bytes[j+1]]))[0]
        csi.append(complex(real, imag))
    return csi

def compute_snr(amplitudes, noise_floor_dbm):
    """Per-subcarrier SNR in dB.  SNR = 20*log10(amplitude) - noise_floor_dbm."""
    noise_dbm = float(noise_floor_dbm)
    snr = []
    for amp in amplitudes:
        amp_clamped = max(amp, 1e-6)
        amp_dbm     = 20.0 * math.log10(amp_clamped)
        snr.append(round(amp_dbm - noise_dbm, 4))
    return snr

def sanitize_phase(phases):
    """Unwrap phase discontinuities across subcarriers."""
    return list(np.unwrap(phases))

def phase_difference(phases):
    """CFO-removed phase difference between adjacent subcarriers (N-1 values)."""
    unwrapped = np.unwrap(phases)
    return list(np.diff(unwrapped))

def normalize_amplitude(amplitudes):
    """Per-frame min-max normalization of amplitudes to [0, 1]."""
    arr   = np.array(amplitudes, dtype=float)
    a_min = arr.min()
    a_max = arr.max()
    rng   = a_max - a_min
    if rng < 1e-9:
        return list(np.zeros_like(arr))
    return list((arr - a_min) / rng)

def extract_features(csi_complex, rssi, noise_floor):
    """Extract all features from one CSI frame."""
    amplitudes = [abs(c)                     for c in csi_complex]
    phases     = [math.atan2(c.imag, c.real) for c in csi_complex]
    return {
        "amplitudes":     amplitudes,
        "phases":         sanitize_phase(phases),
        "phase_diff":     phase_difference(phases),
        "amp_normalized": normalize_amplitude(amplitudes),
        "snr":            compute_snr(amplitudes, noise_floor),
        "rssi":           rssi,
        "noise_floor":    noise_floor,
    }

def parse_bin_file(filepath):
    """Parse a .bin file into a list of frame dicts."""
    with open(filepath, "rb") as f:
        raw = f.read()

    frames = []
    i      = 0

    while i < len(raw) - 2:
        # Skip debug comment lines starting with '#'
        if raw[i] == ord('#'):
            while i < len(raw) and raw[i] != ord('\n'):
                i += 1
            i += 1
            continue

        if raw[i] == 0xAA and raw[i+1] == 0x55:
            offset = i + 2
            if offset + HEADER_SIZE > len(raw):
                break

            # Parse 16-byte header: ver(1) flags(1) ms(4) rssi(1) noise_floor(1) mac(6) csi_len(2)
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
                "timestamp_ms":   timestamp,
                "mac":            mac,
                "shouter_id":     SHOUTER_MACS[mac],
                "amplitudes":     features["amplitudes"],
                "phases":         features["phases"],
                "phase_diff":     features["phase_diff"],
                "amp_normalized": features["amp_normalized"],
                "snr":            features["snr"],
                "rssi":           features["rssi"],
                "noise_floor":    features["noise_floor"],
            })
            i = hdr_end + csi_len
        else:
            i += 1

    return frames

# ================================================================================
# BUCKETING
# ================================================================================
def bucket_frames(frames, bucket_ms=BUCKET_MS):
    """Group frames into time buckets. Each bucket = one CSV row.
    Frames from the same shouter within a bucket are averaged."""
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
                avg_amp        = np.mean([f["amplitudes"]     for f in frames_in], axis=0)
                avg_phase      = np.mean([f["phases"]         for f in frames_in], axis=0)
                avg_phase_diff = np.mean([f["phase_diff"]     for f in frames_in], axis=0)
                avg_amp_norm   = np.mean([f["amp_normalized"] for f in frames_in], axis=0)
                avg_snr        = np.mean([f["snr"]            for f in frames_in], axis=0)
                avg_rssi       = np.mean([f["rssi"]           for f in frames_in])
                avg_nf         = np.mean([f["noise_floor"]    for f in frames_in])

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
                # Shouter not heard — fill with NaN so CSV row stays consistent
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

# ================================================================================
# CSV
# ================================================================================
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
        for sc in range(SUBCARRIERS - 1):   # N-1 phase differences
            header.append(f"{px}_pdiff_{sc}")
        for sc in range(SUBCARRIERS):
            header.append(f"{px}_snr_{sc}")
        header.append(f"{px}_rssi")
        header.append(f"{px}_noise_floor")
    return header

def append_samples_to_csv(samples, zone_info, zone_id, folder_path, header):
    """Write samples to data.csv inside the run's output folder."""
    csv_path   = os.path.join(folder_path, "data.csv")
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        for sample in samples:
            row             = {k: sample.get(k, float("nan")) for k in header}
            row["label"]    = zone_info["label"]
            row["zone_id"]  = zone_id
            row["grid_row"] = zone_info["row"]
            row["grid_col"] = zone_info["col"]
            writer.writerow(row)

# ================================================================================
# FILE WATCHER HELPERS
# ================================================================================
def is_file_stable(filepath, wait=1.5):
    size_before = os.path.getsize(filepath)
    time.sleep(wait)
    return size_before == os.path.getsize(filepath)

def process_file(filepath, zone_info, zone_id, folder_path, header):
    if os.path.getsize(filepath) < 20:
        print(f"  Skipping {os.path.basename(filepath)} — too small.")
        return 0

    frames  = parse_bin_file(filepath)
    samples = bucket_frames(frames)

    if not samples:
        print(f"  No valid samples in {os.path.basename(filepath)}")
        return 0

    append_samples_to_csv(samples, zone_info, zone_id, folder_path, header)
    print(f"  Wrote {len(samples)} samples  |  "
          f"frames={len(frames)}  |  "
          f"file={os.path.basename(filepath)}")
    return len(samples)

# ================================================================================
# PER-CAPTURE PROMPTS  (zone, state, posture, duration, run index)
# ================================================================================
def prompt_capture_params(zones):
    """
    Ask the operator for per-capture parameters.
    Returns (zone_input, state_key, posture_label, duration_s, run_index)
    or None if the operator wants to exit.

    Zone input meanings:
      -1  → exit the script entirely
       0  → Empty room baseline (no person)
      1-9 → Grid cell number
    """
    print("\n--- New Capture ---")
    print("Zone:  0=Empty  1-9=Grid cell  -1=Exit")

    # ---- Zone ----
    while True:
        try:
            zone_input = int(input("Enter zone number: ").strip())
        except ValueError:
            print("  Please enter an integer.")
            continue

        if zone_input == -1:
            return None                          # caller will exit

        if zone_input == 0:
            break                                # empty room — no state/posture needed

        if zone_input in zones:
            break

        print(f"  Invalid zone. Choose 0, -1, or one of {list(zones.keys())}")

    # ---- Grid state & posture (only when a person is present) ----
    state_key     = "Occupied"
    posture_label = "center"

    if zone_input != 0:
        print("\nGrid states:")
        for i, key in enumerate(GRID_STATES.keys(), 1):
            print(f"  {i}. {key}")
        while True:
            try:
                state_choice = int(input("Select state (default 1=Occupied): ").strip() or "1")
                state_key    = list(GRID_STATES.keys())[state_choice - 1]
                break
            except (ValueError, IndexError):
                print(f"  Choose 1-{len(GRID_STATES)}")

        # Posture maps naturally from state in most cases but allow override
        posture_map = {
            "Occupied": "center",
            "Standing": "standing",
            "Seated":   "seated",
            "Moving":   "moving",
        }
        posture_label = posture_map.get(state_key, "center")

    # ---- Duration ----
    while True:
        try:
            duration_s = int(input("Capture duration (seconds) [10]: ").strip() or "10")
            if duration_s > 0:
                break
            print("  Duration must be > 0.")
        except ValueError:
            print("  Please enter an integer number of seconds.")

    # ---- Run index ----
    while True:
        try:
            run_index = int(input("Run index [1]: ").strip() or "1")
            if run_index > 0:
                break
            print("  Run index must be >= 1.")
        except ValueError:
            print("  Please enter a positive integer.")

    return zone_input, state_key, posture_label, duration_s, run_index

# ================================================================================
# MAIN
# ================================================================================
if __name__ == "__main__":

    # ---- Ensure base directories exist ----
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(WATCH_DIR,       exist_ok=True)

    # ---- One-time session metadata ----
    session = SessionMeta()
    session.prompt()

    # Rebuild zone map using room dimensions entered at startup
    zones  = build_zone_map(GRID_COLS, GRID_ROWS, session.room_width, session.room_length)
    header = build_csv_header()

    print_grid(zones, GRID_COLS, GRID_ROWS, session.room_width, session.room_length)

    print("=== CSI Training Data Collector ===")
    print(f"Base output dir : {BASE_OUTPUT_DIR}")
    print(f"Watch dir       : {WATCH_DIR}")
    print(f"Bucket size     : {BUCKET_MS}ms")
    print(f"Subcarriers     : {SUBCARRIERS} per shouter")
    print(f"Shouters        : {len(SHOUTER_MACS)} active of {NUM_SHOUTERS} total")
    print(f"Features        : amp, amp_norm, phase (unwrapped), phase_diff (CFO-removed),"
          f" SNR per subcarrier + RSSI, noise_floor per shouter\n")

    # ---- Main capture loop ----
    while True:
        params = prompt_capture_params(zones)

        # -1 entered — exit cleanly
        if params is None:
            print("\nSession complete. Exiting.")
            sys.exit(0)

        zone_input, state_key, posture_label, duration_s, run_index = params

        # Build the grid state token and derive zone info
        grid_state_token = build_grid_state_token(zone_input, state_key)

        if zone_input == 0:
            # Empty room — use a synthetic zone_info so CSV columns stay consistent
            zone_info = {"label": "empty", "row": 0, "col": 0}
        else:
            zone_info = zones[zone_input]

        # ---- Auto-create output folder with doc-compliant name ----
        folder_path = build_output_folder(session, grid_state_token, duration_s, run_index)
        folder_name = os.path.basename(folder_path)

        # ---- Write metadata.json immediately ----
        write_metadata_json(folder_path, session, grid_state_token,
                            duration_s, run_index, posture_label, zone_input)

        print(f"\nOutput folder   : {folder_path}")
        print(f"Folder name     : {folder_name}")
        print(f"Grid state      : {grid_state_token}  |  Posture: {posture_label}"
              f"  |  Duration: {duration_s}s  |  Run: {run_index:02d}")
        print(f"Watching {WATCH_DIR} for new .bin files ...")
        print("Press Ctrl+C to stop this capture and start a new one.\n")

        processed    = set()
        total_samples = 0

        try:
            while True:
                bin_files        = sorted(glob.glob(os.path.join(WATCH_DIR, "*.bin")))
                files_to_process = bin_files[:-1]   # skip the file currently being written

                for filepath in files_to_process:
                    if filepath not in processed:
                        if is_file_stable(filepath):
                            n = process_file(filepath, zone_info,
                                             zone_input, folder_path, header)
                            total_samples += n
                            processed.add(filepath)

                time.sleep(2)

        except KeyboardInterrupt:
            print(f"\nCapture stopped  —  {total_samples} total samples written to:")
            print(f"  {folder_path}")
            print("Select next capture or enter -1 to exit.\n")
