"""
glasshouse.py  —  Project Glass House  |  CSI Data Collector

Zone prompt:
  -1  → exit cleanly
   0  → Empty room baseline
  1-N → Grid cell number (grid size is configurable)

Capture automatically stops after the specified duration and returns
to the zone prompt — no Ctrl+C required.

Paths (edit CONFIG block below if needed)
-----------------------------------------
  BASE_OUTPUT_DIR : folder where run subfolders are created


  To Be Fixed -> add in grid autoscaling based on input of width and length of perimeter
"""

import csv
import json
import math
import os
import serial
import struct
import sys
import time
from datetime import datetime

import numpy as np


# ============================================================
#  CONFIG
# ============================================================
BASE_OUTPUT_DIR = r"C:\Users\19124\OneDrive\Documents\Senior_Cap\GitRepo\GlassHouse\training_data"

PORT        = "COM3"        # Windows: "COM3"  |  Linux: "/dev/ttyUSB0"
BAUD        = 921600

BUCKET_MS   = 200           # time window to average frames per shouter
SUBCARRIERS = 128           # raw subcarriers from ESP32 (256 byte CSI / 2 bytes each)
MIN_FRAMES  = 1             # min shouters heard per bucket (1 = keep partial rows as NaN)

# ----------------------------------------------------------
#  NULL SUBCARRIERS — removed before CSV output
#  These indices carry no useful signal (DC leakage, zero
#  blocks, hardware artefacts) and would add noise to ML.
# ----------------------------------------------------------
NULL_SUBCARRIERS = {
    0,                                              # DC leakage / anomalous high amplitude
    1,                                              # suspicious outlier (~11 vs neighbors ~22)
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,   # zero block 1
    64,                                             # DC null
    93, 94, 95, 96, 97, 98, 99,                    # zero block 2
}

# Ordered list of subcarrier indices that are actually written to CSV
VALID_SC = [sc for sc in range(SUBCARRIERS) if sc not in NULL_SUBCARRIERS]
N_VALID  = len(VALID_SC)   # 108 with the defaults above


# ============================================================
#  SHOUTER MACs  — add / uncomment as hardware is brought up
# ============================================================
SHOUTER_MACS = {
    "68:FE:71:90:60:A0": 1,
    "68:FE:71:90:68:14": 2,
    "68:FE:71:90:6B:90": 3,
    "20:E7:C8:EC:F5:DC": 4,
}

ACTIVE_SHOUTER_IDS = sorted(SHOUTER_MACS.values())


# ============================================================
#  FRAME FORMAT  (must match ListenerAP.ino exactly)
#  magic(2)  ver(1)  flags(1)  ms(4)  rssi(1)  nf(1)  mac(6)  csi_len(2)
#  HEADER_SIZE = 16 bytes  (after the 2-byte magic)
# ============================================================
HEADER_SIZE = 16
MAGIC_0     = 0xAA
MAGIC_1     = 0x55

# ----------------------------------------------------------
#  ACTIONS — what the subject(s) are doing during capture.
#  Used as the state_key in folder names and metadata.
# ----------------------------------------------------------
ACTIONS = [
    "Standing",         # upright, stationary
    "Seated",           # sitting in chair / on floor
    "Walking",          # moving around the zone
    "Covered",        # arm/hand movements while stationary
    "Lying",            # on the floor / couch
    "Occupied",         # generic / unknown posture (default)
]

ACTION_POSTURE = {
    "Standing":  "standing",
    "Seated":    "seated",
    "Walking":   "moving",
    "Covered": "covered",
    "Lying":     "lying",
    "Occupied":  "center",
}


# ============================================================
#  SESSION METADATA
# ============================================================
class SessionMeta:
    def __init__(self):
        self.operator    = "Unknown"
        self.subject_id  = "Subject_A"
        self.room_width  = 24.0
        self.room_length = 24.0
        self.grid_cols   = 3
        self.grid_rows   = 3
        self.date        = datetime.now().strftime("%Y-%m-%d")

    def prompt(self):
        print("\n=== Session Setup ===")

        op = input("Operator name               : ").strip()
        if op:
            self.operator = op

        sid = input("Subject ID                  : ").strip()
        if sid:
            self.subject_id = sid

        w = input(f"Room width  (ft) [{self.room_width:.0f}]      : ").strip()
        h = input(f"Room length (ft) [{self.room_length:.0f}]      : ").strip()
        if w:
            self.room_width  = float(w)
        if h:
            self.room_length = float(h)

        gc = input(f"Grid columns    [{self.grid_cols}]          : ").strip()
        gr = input(f"Grid rows       [{self.grid_rows}]          : ").strip()
        if gc:
            self.grid_cols = int(gc)
        if gr:
            self.grid_rows = int(gr)

        print(f"\nSession ready  —  Operator: {self.operator}  "
              f"Subject: {self.subject_id}  "
              f"Room: {self.room_width:.0f}x{self.room_length:.0f}ft  "
              f"Grid: {self.grid_cols}x{self.grid_rows}  "
              f"Date: {self.date}\n")


# ============================================================
#  ZONE / FOLDER HELPERS
# ============================================================
def build_zone_map(cols, rows, width_ft, length_ft):
    zone_w = width_ft  / cols
    zone_h = length_ft / rows
    zones  = {}
    for r in range(rows):
        for c in range(cols):
            zid = r * cols + c + 1
            zones[zid] = {
                "label":   f"zone_{zid}",
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
    print("   0: [  Empty room — no person present  ]\n")
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

def build_grid_state_token(zone_input, action, n_subjects):
    if zone_input == 0:
        return "Empty"
    people = f"{n_subjects}P" if n_subjects > 1 else ""
    return f"Grid{zone_input}{action}{people}"

def build_folder_name(session, grid_state_token, duration_s, run_index):
    room = f"{session.room_width:.0f}x{session.room_length:.0f}Room"
    run  = f"Run{run_index:02d}"
    return f"{room}_{grid_state_token}_{duration_s}Seconds_{run}"

def create_run_folder(session, grid_state_token, duration_s, run_index):
    name = build_folder_name(session, grid_state_token, duration_s, run_index)
    path = os.path.join(BASE_OUTPUT_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path

def write_metadata_json(folder_path, session, grid_state_token,
                        duration_s, run_index, posture, zone_id, n_subjects):
    meta = {
        "room_width_ft":    session.room_width,
        "room_length_ft":   session.room_length,
        "grid_cols":        session.grid_cols,
        "grid_rows":        session.grid_rows,
        "grid_state":       grid_state_token,
        "duration_seconds": duration_s,
        "run_index":        run_index,
        "date":             session.date,
        "operator":         session.operator,
        "subject_id":       session.subject_id,
        "n_subjects":       n_subjects,
        "posture":          posture,
        "zone_id":          zone_id,
        "shouters":         [f"ESP32_S{s}" for s in ACTIVE_SHOUTER_IDS],
        "valid_subcarriers":VALID_SC,
        "null_subcarriers": sorted(NULL_SUBCARRIERS),
        "notes":            "",
    }
    with open(os.path.join(folder_path, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ============================================================
#  CAPTURE PROMPTS
# ============================================================
def prompt_capture_params(zones):
    """
    Returns (zone_input, action, posture, duration_s, run_index, n_subjects)
    or None if -1 entered (exit).
    """
    max_zone = max(zones.keys())
    print("\n--- New Capture ---")
    print(f"Zone:  0=Empty  1-{max_zone}=Grid cell  -1=Exit")

    while True:
        try:
            zone_input = int(input("Enter zone number: ").strip())
        except ValueError:
            print("  Please enter an integer.")
            continue
        if zone_input == -1:
            return None
        if zone_input == 0:
            break
        if zone_input in zones:
            break
        print(f"  Invalid. Choose -1, 0, or one of {sorted(zones.keys())}")

    action  = "Occupied"
    posture = "center"

    if zone_input != 0:

        # --- number of subjects ---
        while True:
            try:
                n_subjects = int(input("Number of subjects in zone [1]: ").strip() or "1")
                if n_subjects >= 1:
                    break
                print("  Must be >= 1.")
            except ValueError:
                print("  Enter a positive integer.")

        # --- action ---
        print("\nActions:")
        for i, a in enumerate(ACTIONS, 1):
            print(f"  {i}. {a}")
        while True:
            try:
                choice = int(input(f"Select action (default 1): ").strip() or "1")
                action = ACTIONS[choice - 1]
                break
            except (ValueError, IndexError):
                print(f"  Choose 1-{len(ACTIONS)}")
        posture = ACTION_POSTURE.get(action, "center")

    else:
        n_subjects = 0

    while True:
        try:
            duration_s = int(input("Capture duration seconds [10]: ").strip() or "10")
            if duration_s > 0:
                break
            print("  Must be > 0.")
        except ValueError:
            print("  Enter an integer.")

    while True:
        try:
            run_index = int(input("Run index [1]: ").strip() or "1")
            if run_index > 0:
                break
            print("  Must be >= 1.")
        except ValueError:
            print("  Enter a positive integer.")

    return zone_input, action, posture, duration_s, run_index, n_subjects


# ============================================================
#  CSI FEATURE EXTRACTION
# ============================================================
def _parse_csi_bytes(csi_bytes):
    csi = []
    for j in range(0, len(csi_bytes) - 1, 2):
        imag = struct.unpack('b', bytes([csi_bytes[j]]))[0]
        real = struct.unpack('b', bytes([csi_bytes[j + 1]]))[0]
        csi.append(complex(real, imag))
    return csi

def _compute_snr(amplitudes, noise_floor_dbm):
    noise = float(noise_floor_dbm)
    return [round(20.0 * math.log10(max(a, 1e-6)) - noise, 4) for a in amplitudes]

def _normalize_amplitude(amplitudes):
    arr = np.array(amplitudes, dtype=float)
    rng = arr.max() - arr.min()
    if rng < 1e-9:
        return list(np.zeros_like(arr))
    return list((arr - arr.min()) / rng)

def _extract_features(csi_complex, rssi, noise_floor):
    """
    Extracts features from all 128 subcarriers first, then filters
    to VALID_SC before returning.  phase_diff is computed on the
    full unwrapped phase array and then filtered to valid indices
    (dropping any diff that spans a null subcarrier boundary).
    """
    amplitudes_all = [abs(c)                     for c in csi_complex]
    phases_all     = [math.atan2(c.imag, c.real) for c in csi_complex]
    unwrapped_all  = list(np.unwrap(phases_all))
    amp_norm_all   = _normalize_amplitude(amplitudes_all)
    snr_all        = _compute_snr(amplitudes_all, noise_floor)

    # phase_diff on full array — index i of pdiff corresponds to
    # the difference between subcarrier i+1 and subcarrier i.
    # We keep pdiff[i] only when BOTH i and i+1 are valid.
    pdiff_all = list(np.diff(unwrapped_all))
    valid_pdiff_idx = [i for i in range(SUBCARRIERS - 1)
                       if i not in NULL_SUBCARRIERS and (i + 1) not in NULL_SUBCARRIERS]

    return {
        "amplitudes":     [amplitudes_all[sc] for sc in VALID_SC],
        "phases":         [unwrapped_all[sc]  for sc in VALID_SC],
        "phase_diff":     [pdiff_all[i]       for i in valid_pdiff_idx],
        "amp_normalized": [amp_norm_all[sc]   for sc in VALID_SC],
        "snr":            [snr_all[sc]        for sc in VALID_SC],
        "rssi":           rssi,
        "noise_floor":    noise_floor,
        # store valid_pdiff_idx length so CSV builder knows column count
        "_n_pdiff":       len(valid_pdiff_idx),
    }

# Compute once at import time so build_csv_header() can use it
_VALID_PDIFF_IDX = [i for i in range(SUBCARRIERS - 1)
                    if i not in NULL_SUBCARRIERS and (i + 1) not in NULL_SUBCARRIERS]
N_VALID_PDIFF    = len(_VALID_PDIFF_IDX)


# ============================================================
#  CSV
# ============================================================
def build_csv_header():
    header = ["timestamp_ms", "label", "zone_id", "grid_row", "grid_col", "n_subjects"]
    for s in ACTIVE_SHOUTER_IDS:
        px = f"s{s}"
        for sc in VALID_SC:
            header.append(f"{px}_amp_{sc}")
        for sc in VALID_SC:
            header.append(f"{px}_amp_norm_{sc}")
        for sc in VALID_SC:
            header.append(f"{px}_phase_{sc}")
        for i in _VALID_PDIFF_IDX:
            header.append(f"{px}_pdiff_{i}")
        for sc in VALID_SC:
            header.append(f"{px}_snr_{sc}")
        header.append(f"{px}_rssi")
        header.append(f"{px}_noise_floor")
    return header

def open_csv(folder_path, header):
    csv_path    = os.path.join(folder_path, "data.csv")
    file_exists = os.path.exists(csv_path)
    f           = open(csv_path, "a", newline="")
    writer      = csv.DictWriter(f, fieldnames=header)
    if not file_exists:
        writer.writeheader()
    return f, writer

def write_bucket_to_csv(writer, bucket_data, zone_info, zone_id, n_subjects, header):
    """
    bucket_data : dict  {shouter_id: [frame, ...], "t_bucket": ms}
    Averages frames per shouter, fills NaN for any shouter not heard.
    """
    active = sum(1 for sid in ACTIVE_SHOUTER_IDS if bucket_data[sid])

    if active < MIN_FRAMES:
        return 0

    row = {k: float("nan") for k in header}
    row["timestamp_ms"] = bucket_data["t_bucket"]
    row["label"]        = zone_info["label"]
    row["zone_id"]      = zone_id
    row["grid_row"]     = zone_info["row"]
    row["grid_col"]     = zone_info["col"]
    row["n_subjects"]   = n_subjects

    for sid in ACTIVE_SHOUTER_IDS:
        px        = f"s{sid}"
        frames_in = bucket_data[sid]

        if not frames_in:
            continue   # columns stay NaN

        avg_amp        = np.mean([f["amplitudes"]     for f in frames_in], axis=0)
        avg_phase      = np.mean([f["phases"]         for f in frames_in], axis=0)
        avg_phase_diff = np.mean([f["phase_diff"]     for f in frames_in], axis=0)
        avg_amp_norm   = np.mean([f["amp_normalized"] for f in frames_in], axis=0)
        avg_snr        = np.mean([f["snr"]            for f in frames_in], axis=0)
        avg_rssi       = float(np.mean([f["rssi"]        for f in frames_in]))
        avg_nf         = float(np.mean([f["noise_floor"] for f in frames_in]))

        for idx, sc in enumerate(VALID_SC):
            row[f"{px}_amp_{sc}"]      = round(float(avg_amp[idx]),      4)
            row[f"{px}_amp_norm_{sc}"] = round(float(avg_amp_norm[idx]), 4)
            row[f"{px}_phase_{sc}"]    = round(float(avg_phase[idx]),    4)
            row[f"{px}_snr_{sc}"]      = round(float(avg_snr[idx]),      4)
        for idx, i in enumerate(_VALID_PDIFF_IDX):
            row[f"{px}_pdiff_{i}"]     = round(float(avg_phase_diff[idx]), 4)
        row[f"{px}_rssi"]        = round(avg_rssi, 2)
        row[f"{px}_noise_floor"] = round(avg_nf,   2)

    writer.writerow(row)
    return 1


# ============================================================
#  SERIAL  —  open / ready-wait
# ============================================================
def open_serial():
    ser          = serial.Serial()
    ser.port     = PORT
    ser.baudrate = BAUD
    ser.timeout  = 0.05     # short timeout keeps the capture loop responsive
    ser.dtr      = False
    ser.rts      = False
    ser.open()
    time.sleep(0.5)
    ser.reset_input_buffer()
    return ser

def wait_for_ready(ser):
    print("  Waiting for ESP32 LISTENER_AP_READY signal...")
    deadline = time.time() + 5.0
    while time.time() < deadline:
        line = ser.readline()
        if b"LISTENER_AP_READY" in line:
            print("  ESP32 ready.\n")
            return
    print("  No ready signal — ESP32 likely already running. Continuing.\n")


# ============================================================
#  CAPTURE LOOP  —  serial → parse → bucket → CSV
# ============================================================
def run_capture(ser, folder_path, zone_info, zone_id, n_subjects, duration_s, csv_header):
    """
    Reads the serial port for duration_s seconds.
    Parses frames on the fly, buckets them, writes rows to CSV.
    Returns total number of CSV rows written.
    """
    csv_file, csv_writer = open_csv(folder_path, csv_header)

    buf          = bytearray()
    buckets      = {}       # bid -> {"t_bucket": ms, sid: [frames], ...}
    t_start_ms   = None     # ESP32 timestamp of first frame in this capture
    rows_written = 0

    deadline = time.time() + duration_s

    try:
        while time.time() < deadline:

            # countdown display
            remaining = int(deadline - time.time())
            print(f"\r  {remaining:3d}s remaining  |  rows written: {rows_written:5d}  ",
                  end="", flush=True)

            # read serial
            chunk = ser.read(4096)
            if chunk:
                buf.extend(chunk)

            # parse all complete frames out of buf
            i = 0
            n = len(buf)

            while i < n - 1:

                # skip comment lines emitted by firmware
                if buf[i] == ord('#'):
                    j = buf.find(b'\n', i)
                    if j == -1:
                        break   # incomplete line — wait for more bytes
                    i = j + 1
                    continue

                # look for magic bytes
                if not (buf[i] == MAGIC_0 and buf[i + 1] == MAGIC_1):
                    i += 1
                    continue

                offset = i + 2
                if offset + HEADER_SIZE > n:
                    break       # header incomplete

                csi_len       = struct.unpack_from('<H', buf, offset + 14)[0]
                payload_start = offset + HEADER_SIZE
                frame_end     = payload_start + csi_len

                if frame_end > n:
                    break       # payload incomplete

                # parse header fields
                timestamp   = struct.unpack_from('<I', buf, offset + 2)[0]
                rssi        = struct.unpack_from('<b', buf, offset + 6)[0]
                noise_floor = struct.unpack_from('<b', buf, offset + 7)[0]
                mac         = ':'.join(f'{b:02X}' for b in buf[offset + 8: offset + 14])

                i = frame_end

                if mac not in SHOUTER_MACS:
                    continue

                # extract features (null subcarriers removed inside)
                csi_bytes   = buf[payload_start:frame_end]
                csi_complex = _parse_csi_bytes(csi_bytes)
                features    = _extract_features(csi_complex, rssi, noise_floor)
                frame       = {"timestamp_ms": timestamp,
                               "shouter_id":   SHOUTER_MACS[mac],
                               **features}

                # assign to bucket
                if t_start_ms is None:
                    t_start_ms = timestamp

                bid = (timestamp - t_start_ms) // BUCKET_MS

                if bid not in buckets:
                    # flush buckets that are now closed (older than bid-1)
                    for cb in sorted(b for b in list(buckets) if b < bid - 1):
                        rows_written += write_bucket_to_csv(
                            csv_writer, buckets[cb], zone_info, zone_id,
                            n_subjects, csv_header)
                        del buckets[cb]

                    buckets[bid] = {sid: [] for sid in ACTIVE_SHOUTER_IDS}
                    buckets[bid]["t_bucket"] = t_start_ms + bid * BUCKET_MS

                buckets[bid][frame["shouter_id"]].append(frame)

            buf = buf[i:]   # keep only unconsumed bytes

    except KeyboardInterrupt:
        pass    # early stop — fall through to flush

    finally:
        # flush all remaining open buckets
        for cb in sorted(buckets):
            rows_written += write_bucket_to_csv(
                csv_writer, buckets[cb], zone_info, zone_id, n_subjects, csv_header)
        csv_file.flush()
        csv_file.close()

    print()   # newline after countdown
    return rows_written


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    session    = SessionMeta()
    session.prompt()

    zones      = build_zone_map(session.grid_cols, session.grid_rows,
                                session.room_width, session.room_length)
    csv_header = build_csv_header()

    print_grid(zones, session.grid_cols, session.grid_rows,
               session.room_width, session.room_length)

    print("=== Glass House — CSI Data Collector ===")
    print(f"  Serial port         : {PORT}  @  {BAUD} baud")
    print(f"  Training output     : {BASE_OUTPUT_DIR}")
    print(f"  Bucket size         : {BUCKET_MS}ms")
    print(f"  Subcarriers (raw)   : {SUBCARRIERS}")
    print(f"  Null subcarriers    : {len(NULL_SUBCARRIERS)} removed  →  {N_VALID} used")
    print(f"  Shouters active     : {len(ACTIVE_SHOUTER_IDS)}  (IDs: {ACTIVE_SHOUTER_IDS})")
    print(f"  Min shouters/bucket : {MIN_FRAMES}")
    print(f"  Grid                : {session.grid_cols}x{session.grid_rows}"
          f"  ({session.grid_cols * session.grid_rows} zones)\n")

    # open serial once — stays open for the entire session
    try:
        ser = open_serial()
    except serial.SerialException as e:
        print(f"\nERROR opening {PORT}: {e}")
        sys.exit(1)

    wait_for_ready(ser)

    try:
        while True:
            params = prompt_capture_params(zones)

            if params is None:
                print("\nSession complete. Goodbye.")
                break

            zone_input, action, posture, duration_s, run_index, n_subjects = params
            grid_state_token = build_grid_state_token(zone_input, action, n_subjects)

            zone_info = {"label": "empty", "row": 0, "col": 0} \
                        if zone_input == 0 else zones[zone_input]

            folder_path = create_run_folder(session, grid_state_token, duration_s, run_index)
            write_metadata_json(folder_path, session, grid_state_token,
                                duration_s, run_index, posture, zone_input, n_subjects)

            subj_str = f"{n_subjects} subject{'s' if n_subjects != 1 else ''}"
            print(f"\n  Run folder  : {folder_path}")
            print(f"  Grid state  : {grid_state_token}  |  Action: {action}"
                  f"  |  Subjects: {subj_str}  |  Duration: {duration_s}s  |  Run: {run_index:02d}")
            print(f"  Capturing   — auto-stops in {duration_s}s  (Ctrl+C to stop early)\n")

            rows = run_capture(ser, folder_path, zone_info, zone_input,
                               n_subjects, duration_s, csv_header)

            print(f"  Capture complete  →  {os.path.basename(folder_path)}  ({rows} rows written)")
            print("  Select next capture or enter -1 to exit.\n")

    finally:
        ser.dtr = False
        ser.rts = False
        ser.close()
        print("  Serial port closed.")