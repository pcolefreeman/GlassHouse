"""
glasshouse.py  —  Project Glass House  |  Combined CSI Writer + Processor
==========================================================================
Thread 1  (SerialWriter)  : Reads ESP32 serial stream, buffers complete
                            frames, writes them to timestamped .bin files
                            in BIN_DIR.  Frames are NEVER split across files.
                            Files are ALWAYS written regardless of capture state.

Thread 2  (BinProcessor)  : Watches BIN_DIR for stable .bin files, parses
                            frames, buckets them, and appends rows to the
                            run's data.csv.  Only processes files created
                            AFTER a capture session starts (by timestamp).

Usage
-----
  python glasshouse.py

Zone prompt:
  -1  → exit cleanly
   0  → Empty room baseline
  1-9 → Grid cell number

Capture automatically stops after the specified duration and returns
to the zone prompt — no Ctrl+C required.

Paths (edit the CONFIG block below if needed)
-----
  BIN_DIR         :  C:\\GlassHouse\\CSI_data          (incoming .bin files)
  BASE_OUTPUT_DIR :  C:\\GlassHouse\\training_data     (run folders / CSV / JSON)
"""

import csv
import glob
import json
import math
import os
import serial
import struct
import sys
import threading
import time
from datetime import datetime

import numpy as np

# ============================================================
#  CONFIG  — edit these if paths or serial port change
# ============================================================
BIN_DIR         = r"C:\Users\19124\OneDrive\Documents\Senior_Cap\GitRepo\GlassHouse\CSI_data"
BASE_OUTPUT_DIR = r"C:\Users\19124\OneDrive\Documents\Senior_Cap\GitRepo\GlassHouse\training_data"

PORT             = "COM3"       # Windows: "COM3"  |  Linux: "/dev/ttyUSB0"
BAUD             = 921600
INTERVAL_SECONDS = 5            # seconds of data per .bin file

BUCKET_MS    = 50
NUM_SHOUTERS = 4
SUBCARRIERS  = 128              # 256 byte CSI / 2 bytes per complex sample

# ============================================================
#  SHOUTER MACs  — add / uncomment as hardware is brought up
# ============================================================
SHOUTER_MACS = {
    "68:FE:71:90:60:A0": 1,
    "68:FE:71:90:68:14": 2,
    "68:FE:71:90:6B:90": 3,
    # "XX:XX:XX:XX:XX:XX": 4,
}

# MIN_FRAMES: minimum number of shouters that must be heard in a bucket
# for it to be written to CSV. Set to 1 so partial data is never dropped —
# missing shouters are filled with NaN and can be filtered in EDA.
MIN_FRAMES = 1

# ============================================================
#  THREAD-SAFE PRINTING
# ============================================================
import queue as _queue
_print_queue = _queue.Queue()

def tprint(msg):
    """Called by background threads instead of print()."""
    _print_queue.put(msg)

def flush_print_queue():
    """Called by main thread between prompts to drain queued messages."""
    while not _print_queue.empty():
        try:
            print(_print_queue.get_nowait())
        except _queue.Empty:
            break

# ============================================================
#  FRAME FORMAT  (must match ListenerAP.ino exactly)
#  magic(2)  ver(1)  flags(1)  ms(4)  rssi(1)  nf(1)  mac(6)  csi_len(2)
#  HEADER_SIZE = 16 bytes  (after the 2-byte magic)
# ============================================================
HEADER_SIZE = 16
MAGIC_0     = 0xAA
MAGIC_1     = 0x55

# Grid state descriptors
GRID_STATES = ["Occupied", "Standing", "Seated", "Moving"]

# ============================================================
#  GLOBAL CAPTURE STATE  — shared between threads
# ============================================================
_capture_lock      = threading.Lock()
_current_folder    = None      # absolute path of the active run folder
_current_zone_info = None      # dict from zones map
_current_zone_id   = None      # int
_capture_start_ts  = None      # time.time() when capture session started
_capture_end_ts    = None      # time.time() when capture session should end
_csv_header        = None      # built once at startup
_processed_files   = set()     # .bin files already handled by processor thread
_stop_event        = threading.Event()


# ============================================================
#  SESSION METADATA
# ============================================================
class SessionMeta:
    def __init__(self):
        self.operator    = "Unknown"
        self.subject_id  = "Subject_A"
        self.room_width  = 24.0
        self.room_length = 24.0
        self.date        = datetime.now().strftime("%Y-%m-%d")

    def prompt(self):
        print("\n=== Session Setup ===")
        op = input("Operator name        : ").strip()
        if op:
            self.operator = op
        sid = input("Subject ID           : ").strip()
        if sid:
            self.subject_id = sid
        w = input(f"Room width  (ft) [{self.room_width:.0f}] : ").strip()
        h = input(f"Room length (ft) [{self.room_length:.0f}] : ").strip()
        if w:
            self.room_width  = float(w)
        if h:
            self.room_length = float(h)
        print(f"\nSession ready  —  Operator: {self.operator}  "
              f"Subject: {self.subject_id}  "
              f"Room: {self.room_width:.0f}x{self.room_length:.0f}ft  "
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

def build_grid_state_token(zone_input, state_key):
    if zone_input == 0:
        return "Empty"
    return f"Grid{zone_input}{state_key}"

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
                        duration_s, run_index, posture, zone_id):
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
        "shouters":         [f"ESP32_S{s}" for s in range(1, NUM_SHOUTERS + 1)],
        "notes":            "",
    }
    with open(os.path.join(folder_path, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ============================================================
#  CAPTURE PROMPTS
# ============================================================
def prompt_capture_params(zones):
    """
    Returns (zone_input, state_key, posture, duration_s, run_index)
    or None if -1 entered (exit).
    """
    print("\n--- New Capture ---")
    print("Zone:  0=Empty  1-9=Grid cell  -1=Exit")

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
        print(f"  Invalid. Choose -1, 0, or one of {list(zones.keys())}")

    state_key = "Occupied"
    posture   = "center"
    if zone_input != 0:
        print("\nGrid states:")
        for i, s in enumerate(GRID_STATES, 1):
            print(f"  {i}. {s}")
        while True:
            try:
                choice    = int(input("Select state (default 1): ").strip() or "1")
                state_key = GRID_STATES[choice - 1]
                break
            except (ValueError, IndexError):
                print(f"  Choose 1-{len(GRID_STATES)}")
        posture_map = {"Occupied": "center", "Standing": "standing",
                       "Seated":   "seated",  "Moving":   "moving"}
        posture = posture_map.get(state_key, "center")

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

    return zone_input, state_key, posture, duration_s, run_index


# ============================================================
#  FRAME PARSER
# ============================================================
def parse_bin_file(filepath):
    with open(filepath, "rb") as f:
        raw = f.read()

    frames = []
    i      = 0
    n      = len(raw)

    while i < n - 1:

        if raw[i] == ord('#'):
            while i < n and raw[i] != ord('\n'):
                i += 1
            i += 1
            continue

        if not (raw[i] == MAGIC_0 and raw[i + 1] == MAGIC_1):
            i += 1
            continue

        offset = i + 2

        if offset + HEADER_SIZE > n:
            break

        timestamp     = struct.unpack_from('<I', raw, offset + 2)[0]
        rssi          = struct.unpack_from('<b', raw, offset + 6)[0]
        noise_floor   = struct.unpack_from('<b', raw, offset + 7)[0]
        mac           = ':'.join(f'{b:02X}' for b in raw[offset + 8: offset + 14])
        csi_len       = struct.unpack_from('<H', raw, offset + 14)[0]
        payload_start = offset + HEADER_SIZE

        if payload_start + csi_len > n:
            break

        i = payload_start + csi_len

        if mac not in SHOUTER_MACS:
            continue

        csi_bytes   = raw[payload_start: payload_start + csi_len]
        csi_complex = _parse_csi_bytes(csi_bytes)
        features    = _extract_features(csi_complex, rssi, noise_floor)

        frames.append({
            "timestamp_ms": timestamp,
            "mac":          mac,
            "shouter_id":   SHOUTER_MACS[mac],
            **features,
        })

    return frames


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
    amplitudes = [abs(c)                     for c in csi_complex]
    phases     = [math.atan2(c.imag, c.real) for c in csi_complex]
    unwrapped  = list(np.unwrap(phases))
    return {
        "amplitudes":     amplitudes,
        "phases":         unwrapped,
        "phase_diff":     list(np.diff(unwrapped)),
        "amp_normalized": _normalize_amplitude(amplitudes),
        "snr":            _compute_snr(amplitudes, noise_floor),
        "rssi":           rssi,
        "noise_floor":    noise_floor,
    }


# ============================================================
#  BUCKETING
# ============================================================
def bucket_frames(frames):
    if not frames:
        return []

    frames  = sorted(frames, key=lambda f: f["timestamp_ms"])
    t_start = frames[0]["timestamp_ms"]
    buckets = {}

    for frame in frames:
        bid = (frame["timestamp_ms"] - t_start) // BUCKET_MS
        buckets.setdefault(bid, {s: [] for s in range(1, NUM_SHOUTERS + 1)})
        buckets[bid][frame["shouter_id"]].append(frame)

    samples = []
    for bid in sorted(buckets):
        bucket   = buckets[bid]
        t_bucket = t_start + bid * BUCKET_MS
        active   = sum(1 for s in bucket if bucket[s])

        if active < MIN_FRAMES:
            continue

        sample = {"timestamp_ms": t_bucket}

        for sid in range(1, NUM_SHOUTERS + 1):
            px        = f"s{sid}"
            frames_in = bucket[sid]

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
                for sc in range(len(avg_phase_diff)):
                    sample[f"{px}_pdiff_{sc}"] = round(float(avg_phase_diff[sc]), 4)
                sample[f"{px}_rssi"]        = round(float(avg_rssi), 2)
                sample[f"{px}_noise_floor"] = round(float(avg_nf),   2)

            else:
                for sc in range(SUBCARRIERS):
                    sample[f"{px}_amp_{sc}"]      = float("nan")
                    sample[f"{px}_amp_norm_{sc}"] = float("nan")
                    sample[f"{px}_phase_{sc}"]    = float("nan")
                    sample[f"{px}_snr_{sc}"]      = float("nan")
                for sc in range(SUBCARRIERS - 1):
                    sample[f"{px}_pdiff_{sc}"] = float("nan")
                sample[f"{px}_rssi"]        = float("nan")
                sample[f"{px}_noise_floor"] = float("nan")

        samples.append(sample)

    return samples


# ============================================================
#  CSV
# ============================================================
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
        for sc in range(SUBCARRIERS - 1):
            header.append(f"{px}_pdiff_{sc}")
        for sc in range(SUBCARRIERS):
            header.append(f"{px}_snr_{sc}")
        header.append(f"{px}_rssi")
        header.append(f"{px}_noise_floor")
    return header

def append_samples_to_csv(samples, zone_info, zone_id, folder_path, header):
    csv_path    = os.path.join(folder_path, "data.csv")
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


# ============================================================
#  THREAD 1 — SERIAL WRITER
#  Always writes .bin files regardless of capture state.
#  The processor thread decides whether to use each file.
# ============================================================
class SerialWriter(threading.Thread):
    def __init__(self):
        super().__init__(name="SerialWriter", daemon=True)
        self._ser    = None
        self._buf    = bytearray()
        self._frames = bytearray()

    def _open_serial(self):
        ser          = serial.Serial()
        ser.port     = PORT
        ser.baudrate = BAUD
        ser.timeout  = 1
        ser.dtr      = False
        ser.rts      = False
        ser.open()
        time.sleep(0.5)
        ser.reset_input_buffer()
        return ser

    def _wait_for_ready(self, ser):
        tprint("  [Writer] Waiting for ESP32 LISTENER_AP_READY signal...")
        deadline = time.time() + 5.0
        while time.time() < deadline:
            line = ser.readline()
            if b"LISTENER_AP_READY" in line:
                tprint("  [Writer] ESP32 ready.")
                return
        tprint("  [Writer] No ready signal — ESP32 likely already running. Continuing.")

    def _extract_complete_frames(self, raw_chunk):
        self._buf.extend(raw_chunk)
        i = 0
        n = len(self._buf)

        while i < n - 1:

            if self._buf[i] == ord('#'):
                j = i
                while j < n and self._buf[j] != ord('\n'):
                    j += 1
                if j >= n:
                    break
                j += 1
                self._frames.extend(self._buf[i:j])
                i = j
                continue

            if not (self._buf[i] == MAGIC_0 and self._buf[i + 1] == MAGIC_1):
                i += 1
                continue

            offset = i + 2
            if offset + HEADER_SIZE > n:
                break

            csi_len       = struct.unpack_from('<H', self._buf, offset + 14)[0]
            payload_start = offset + HEADER_SIZE
            frame_end     = payload_start + csi_len

            if frame_end > n:
                break

            self._frames.extend(self._buf[i:frame_end])
            i = frame_end

        self._buf = self._buf[i:]

    def _flush_to_file(self):
        if not self._frames:
            return
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(BIN_DIR, f"csi_{ts}.bin")
        with open(filepath, "wb") as f:
            f.write(self._frames)
        tprint(f"  [Writer] Saved {os.path.basename(filepath)}  "
               f"({len(self._frames):,} bytes)")
        self._frames = bytearray()

    def run(self):
        try:
            self._ser = self._open_serial()
        except serial.SerialException as e:
            tprint(f"  [Writer] ERROR opening {PORT}: {e}")
            _stop_event.set()
            return

        self._wait_for_ready(self._ser)
        tprint(f"  [Writer] Recording {INTERVAL_SECONDS}s intervals → {BIN_DIR}\n")

        try:
            while not _stop_event.is_set():
                self._frames = bytearray()
                interval_end = time.time() + INTERVAL_SECONDS

                while time.time() < interval_end and not _stop_event.is_set():
                    try:
                        chunk = self._ser.read(512)
                        if chunk:
                            self._extract_complete_frames(chunk)
                    except serial.SerialException as e:
                        tprint(f"  [Writer] Serial error: {e}")
                        _stop_event.set()
                        break

                self._flush_to_file()

        finally:
            self._flush_to_file()
            self._ser.dtr = False
            self._ser.rts = False
            self._ser.close()
            tprint("  [Writer] Serial port closed.")


# ============================================================
#  THREAD 2 — BIN PROCESSOR
#  Only processes .bin files created after the current capture
#  session started (checked via file mtime vs _capture_start_ts).
#  Stops writing to CSV once _capture_end_ts has passed.
# ============================================================
class BinProcessor(threading.Thread):
    def __init__(self):
        super().__init__(name="BinProcessor", daemon=True)

    def _is_stable(self, filepath, wait=1.5):
        size_before = os.path.getsize(filepath)
        time.sleep(wait)
        return size_before == os.path.getsize(filepath)

    def _process_file(self, filepath):
        if os.path.getsize(filepath) < 20:
            tprint(f"  [Processor] Skipping {os.path.basename(filepath)} — too small.")
            return 0

        frames  = parse_bin_file(filepath)
        samples = bucket_frames(frames)

        if not samples:
            tprint(f"  [Processor] No valid samples in {os.path.basename(filepath)}"
                   f"  (raw frames={len(frames)}, shouters needed={MIN_FRAMES})")
            return 0

        with _capture_lock:
            folder    = _current_folder
            zone_info = _current_zone_info
            zone_id   = _current_zone_id
            header    = _csv_header

        if folder is None:
            tprint(f"  [Processor] No active session — skipping {os.path.basename(filepath)}")
            return 0

        append_samples_to_csv(samples, zone_info, zone_id, folder, header)
        tprint(f"  [Processor] {os.path.basename(filepath)}"
               f"  →  {len(samples)} samples  (frames={len(frames)})"
               f"  →  {os.path.basename(folder)}")
        return len(samples)

    def run(self):
        tprint(f"  [Processor] Watching {BIN_DIR} for .bin files...")

        while not _stop_event.is_set():
            with _capture_lock:
                start_ts  = _capture_start_ts
                end_ts    = _capture_end_ts
                processed = set(_processed_files)

            bin_files      = sorted(glob.glob(os.path.join(BIN_DIR, "*.bin")))
            files_to_check = bin_files[:-1] if len(bin_files) > 1 else []

            for filepath in files_to_check:
                if filepath in processed:
                    continue

                # No active session — skip
                if start_ts is None:
                    continue

                file_mtime = os.path.getmtime(filepath)

                # File predates this session — mark and skip
                if file_mtime < start_ts:
                    with _capture_lock:
                        _processed_files.add(filepath)
                    continue

                # File was created after capture ended — skip and mark
                if end_ts is not None and file_mtime > end_ts:
                    with _capture_lock:
                        _processed_files.add(filepath)
                    continue

                if self._is_stable(filepath):
                    self._process_file(filepath)
                    with _capture_lock:
                        _processed_files.add(filepath)

            time.sleep(2)


# ============================================================
#  CLEANUP
# ============================================================
def _wipe_bin_dir():
    """Delete all .bin files from BIN_DIR on exit."""
    bin_files = glob.glob(os.path.join(BIN_DIR, "*.bin"))
    deleted   = 0
    for f in bin_files:
        try:
            os.remove(f)
            deleted += 1
        except OSError as e:
            print(f"  [Cleanup] Could not delete {os.path.basename(f)}: {e}")
    print(f"  [Cleanup] Deleted {deleted} .bin file(s) from {BIN_DIR}")


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":

    os.makedirs(BIN_DIR,         exist_ok=True)
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    session     = SessionMeta()
    session.prompt()

    zones       = build_zone_map(3, 3, session.room_width, session.room_length)
    _csv_header = build_csv_header()

    print_grid(zones, 3, 3, session.room_width, session.room_length)

    print("=== Glass House — CSI Data Collector ===")
    print(f"  Serial port     : {PORT}  @  {BAUD} baud")
    print(f"  Bin interval    : {INTERVAL_SECONDS}s per file  →  {BIN_DIR}")
    print(f"  Training output : {BASE_OUTPUT_DIR}")
    print(f"  Bucket size     : {BUCKET_MS}ms")
    print(f"  Subcarriers     : {SUBCARRIERS} per shouter (raw)")
    print(f"  Shouters active : {len(SHOUTER_MACS)} of {NUM_SHOUTERS}")
    print(f"  Min shouters/bucket : {MIN_FRAMES}\n")

    writer    = SerialWriter()
    processor = BinProcessor()
    writer.start()
    processor.start()

    time.sleep(0.3)
    flush_print_queue()

    while True:
        flush_print_queue()
        params = prompt_capture_params(zones)
        flush_print_queue()

        # -1 entered — stop threads and exit
        if params is None:
            print("\nStopping threads...")
            _stop_event.set()
            writer.join(timeout=INTERVAL_SECONDS + 2)
            processor.join(timeout=4)
            flush_print_queue()
            _wipe_bin_dir()
            print("Session complete. Goodbye.")
            sys.exit(0)

        zone_input, state_key, posture, duration_s, run_index = params
        grid_state_token = build_grid_state_token(zone_input, state_key)

        if zone_input == 0:
            zone_info = {"label": "empty", "row": 0, "col": 0}
        else:
            zone_info = zones[zone_input]

        folder_path = create_run_folder(session, grid_state_token, duration_s, run_index)
        write_metadata_json(folder_path, session, grid_state_token,
                            duration_s, run_index, posture, zone_input)

        now = time.time()

        # Atomically update capture state
        with _capture_lock:
            _current_folder    = folder_path
            _current_zone_info = zone_info
            _current_zone_id   = zone_input
            _capture_start_ts  = now
            _capture_end_ts    = now + duration_s
            _processed_files   = set()

        folder_name = os.path.basename(folder_path)
        print(f"\n  Run folder  : {folder_path}")
        print(f"  Grid state  : {grid_state_token}  |  Posture: {posture}"
              f"  |  Duration: {duration_s}s  |  Run: {run_index:02d}")
        print(f"  Capturing   — auto-stops in {duration_s}s ...\n")

        # Count down, flushing thread messages every second
        # Ctrl+C still works as an early stop
        try:
            for remaining in range(duration_s, 0, -1):
                flush_print_queue()
                print(f"  \r  {remaining:3d}s remaining ...", end="", flush=True)
                time.sleep(1)
        except KeyboardInterrupt:
            pass

        print()  # newline after countdown

        # Close the capture window — processor will ignore any later files
        with _capture_lock:
            _capture_end_ts    = time.time()
            _current_folder    = None
            _current_zone_info = None
            _current_zone_id   = None
            _capture_start_ts  = None
            _capture_end_ts    = None

        flush_print_queue()
        print(f"  Capture complete  →  {folder_name}")
        print("  Select next capture or enter -1 to exit.\n")