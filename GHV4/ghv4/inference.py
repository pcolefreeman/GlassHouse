"""inference.py — GHV4 live inference with spacing features.

Reads bidirectional CSI frames from Serial, extracts features, appends
6 shouter-spacing features from spacing.json, runs trained model.
Run in dry-run mode (no --model arg) to verify the pipeline without a trained model.

Usage:
    python inference.py --port COM3 --model model.pkl --spacing spacing.json
    python inference.py --port COM3                    # dry-run
"""
import argparse
import json
import logging
import math
import os
import serial
import time

from ghv4 import csi_parser
from ghv4.config import PAIR_KEYS, BAUD_RATE, SPACING_FEATURE_NAMES

_log = logging.getLogger(__name__)

SPACING_REFRESH_S = 5


def load_preprocessor(processed_dir: str):
    """Load feature_names.txt and scaler.pkl from the processed data directory.

    Returns (trained_feature_names, scaler_dict) or (None, None) if files missing.
    trained_feature_names: list of column names the model was trained on (post-drop).
    scaler_dict: {'amp_cols', 'amp_mean', 'amp_std', 'rssi_cols', 'rssi_mean', 'rssi_std'}.
    """
    fn_path = os.path.join(processed_dir, "feature_names.txt")
    sc_path = os.path.join(processed_dir, "scaler.pkl")
    try:
        with open(fn_path) as f:
            trained_names = [line.strip() for line in f if line.strip()]
        import joblib
        scaler = joblib.load(sc_path)
        return trained_names, scaler
    except FileNotFoundError as e:
        _log.warning("Preprocessor files not found (%s) — skipping scaling", e)
        return None, None


def apply_preprocessing(raw_features: list, raw_names: list,
                        trained_names: list, scaler: dict) -> list:
    """Apply the same column-drop + scaling that preprocess.py applies.

    raw_features: full feature vector from extract_feature_vector (aligned to raw_names).
    raw_names: column names from build_feature_names().
    trained_names: post-drop column names from feature_names.txt (excludes spacing).
    scaler: dict with amp/rssi mean/std arrays.

    Returns a list of scaled floats aligned to trained_names.
    """
    # Build lookup: raw column name → value
    lookup = {name: val for name, val in zip(raw_names, raw_features)}

    # Select only trained columns, apply scaling
    amp_col_set = set(scaler.get('amp_cols', []))
    rssi_col_set = set(scaler.get('rssi_cols', []))
    amp_mean = scaler.get('amp_mean', [])
    amp_std = scaler.get('amp_std', [])
    rssi_mean = scaler.get('rssi_mean', [])
    rssi_std = scaler.get('rssi_std', [])

    # Build ordered indices for amp/rssi columns within trained_names
    amp_idx_map = {}  # trained_name → index into amp_mean/std arrays
    rssi_idx_map = {}
    ai = 0
    ri = 0
    for col in trained_names:
        if col in amp_col_set:
            amp_idx_map[col] = ai
            ai += 1
        elif col in rssi_col_set:
            rssi_idx_map[col] = ri
            ri += 1

    result = []
    for col in trained_names:
        # Skip spacing features — they're appended separately
        if col in SPACING_FEATURE_NAMES:
            continue
        val = lookup.get(col, float('nan'))
        if math.isnan(val):
            result.append(0.0)  # NaN fill matches preprocess.py
            continue
        if col in amp_idx_map:
            idx = amp_idx_map[col]
            if idx < len(amp_mean):
                std = amp_std[idx] if amp_std[idx] != 0 else 1.0
                val = (val - amp_mean[idx]) / std
        elif col in rssi_idx_map:
            idx = rssi_idx_map[col]
            if idx < len(rssi_mean):
                std = rssi_std[idx] if rssi_std[idx] != 0 else 1.0
                val = (val - rssi_mean[idx]) / std
        elif '_phase_' in col or '_pdiff_' in col:
            val = val / math.pi
        result.append(val)
    return result


def load_spacing(path: str) -> list:
    """Load spacing.json; return 6 floats ordered by PAIR_KEYS. Zeros if absent."""
    try:
        with open(path) as f:
            data = json.load(f)
        pairs = data.get("pairs", {})
        return [float(pairs.get(k, {}).get("distance_m", 0.0)) for k in PAIR_KEYS]
    except FileNotFoundError:
        return [0.0] * 6


def load_model(path: str):
    """Load a model file. Supports joblib (sklearn) by default.
    Replace with tflite/onnx loading as needed.
    """
    import joblib
    return joblib.load(path)


def run_calibration(
    ser,
    model_dir: str,
    spacing_path: str,
    window_s: float = None,
) -> dict:
    """Run calibration phase: collect snaps via serial, predict distances, write spacing.

    Creates a DistanceCalibrator, reads [0xEE][0xFF] snap frames from `ser`
    for `window_s` seconds, then predicts per-pair distances and writes
    spacing.json. Extends the window if any pair has insufficient data.

    Args:
        ser: Open serial port object (or BytesIO for testing).
        model_dir: Path to distance_models/ directory.
        spacing_path: Where to write spacing.json.
        window_s: Collection window in seconds (default from config).
                  Set to 0.0 for testing with pre-filled BytesIO buffers.

    Returns:
        Dict mapping pair_id → predicted distance in meters.
    """
    import queue
    import threading
    from ghv4.config import (
        CALIBRATION_WINDOW_S,
        CALIBRATION_EXTENSION_S,
        CALIBRATION_MAX_EXTENSIONS,
        PAIR_KEYS,
    )
    from ghv4.distance_inference import DistanceCalibrator
    from ghv4.serial_io import SerialReader

    if window_s is None:
        window_s = CALIBRATION_WINDOW_S

    cal = DistanceCalibrator(model_dir)

    if not cal._models:
        _log.warning("No distance models found in %s, skipping calibration", model_dir)
        return {}

    _log.info("Calibration: collecting snaps for %.0f seconds...", window_s)

    # Create a SerialReader with snap_callback that feeds the calibrator
    fq = queue.Queue()
    reader = SerialReader(
        ser, fq,
        snap_callback=cal.feed_snap,
    )

    # Run reader in a background thread for the calibration window
    stop_event = threading.Event()

    def _timed_run():
        while not stop_event.is_set():
            try:
                reader._read_one_frame()
            except Exception:
                break

    reader_thread = threading.Thread(target=_timed_run, daemon=True)
    reader_thread.start()

    # Wait for collection window
    if window_s > 0:
        time.sleep(window_s)
    else:
        # For testing: let reader drain the buffer
        reader_thread.join(timeout=2.0)

    stop_event.set()
    reader_thread.join(timeout=2.0)

    # Predict distances
    distances = cal.predict_distances()

    # Extension logic: extend if any pair with a model is missing
    extensions = 0
    while extensions < CALIBRATION_MAX_EXTENSIONS:
        missing = [p for p in PAIR_KEYS if p in cal._models and p not in distances]
        if not missing:
            break
        extensions += 1
        _log.info("Extending calibration +%ds for pairs: %s",
                  CALIBRATION_EXTENSION_S, missing)

        stop_event.clear()
        reader_thread = threading.Thread(target=_timed_run, daemon=True)
        reader_thread.start()
        time.sleep(CALIBRATION_EXTENSION_S)
        stop_event.set()
        reader_thread.join(timeout=2.0)

        distances = cal.predict_distances()

    cal.write_spacing(spacing_path, distances)
    return distances


def main():
    parser = argparse.ArgumentParser(description="GHV4 live inference")
    parser.add_argument('--port',    required=True,             help="Serial port")
    parser.add_argument('--model',   default=None,              help="Trained model file (.pkl)")
    parser.add_argument('--baud',    type=int, default=BAUD_RATE)
    parser.add_argument('--spacing', default='spacing.json',
                        help="Path to spacing.json (default: spacing.json in cwd)")
    parser.add_argument('--calibrate', action='store_true',
                        help="Run distance calibration phase and write spacing.json, then exit")
    parser.add_argument('--model-dir', default='distance_models/',
                        help="Path to distance_models/ directory (used with --calibrate)")
    parser.add_argument('--processed-dir', default='data/processed/',
                        help="Path to processed data dir (feature_names.txt + scaler.pkl)")
    args = parser.parse_args()

    if args.calibrate:
        ser = None
        try:
            ser = serial.Serial(args.port, args.baud, timeout=1)
            print(f"[CAL] Calibrating on {args.port} — writing {args.spacing}")
            distances = run_calibration(ser, args.model_dir, args.spacing)
            if distances:
                for pair, dist in sorted(distances.items()):
                    print(f"  {pair}: {dist:.2f} m")
            else:
                print("[CAL] No distance models found or no data collected.")
        except KeyboardInterrupt:
            print("\n[CAL] Interrupted.")
        finally:
            if ser:
                ser.close()
        return

    model = None
    if args.model:
        try:
            model = load_model(args.model)
            print(f"[INF] Loaded model: {args.model}")
        except Exception as e:
            print(f"[WARN] Could not load model ({e}) — dry-run mode")
    else:
        print("[INF] No model specified — dry-run mode")

    raw_feature_names = csi_parser.build_feature_names()
    spacing_vals  = load_spacing(args.spacing)
    last_spacing_load = time.time()

    # Load preprocessing transforms (column drop + scaling)
    trained_names, scaler = load_preprocessor(args.processed_dir)
    if model is not None and trained_names is None:
        print("[WARN] No feature_names.txt/scaler.pkl found — predictions will use raw features")
    elif trained_names is not None:
        print(f"[INF] Loaded preprocessor: {len(trained_names)} trained features")

    # NOTE: ser is initialised to None so the finally block can always call
    # ser.close() safely, even if Serial() raises serial.SerialException.
    ser = None
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
        print(f"[INF] Listening on {args.port} — Ctrl+C to stop\n")
        eof_count = 0
        while True:
            # Refresh spacing periodically
            if time.time() - last_spacing_load >= SPACING_REFRESH_S:
                spacing_vals = load_spacing(args.spacing)
                last_spacing_load = time.time()

            lf, sf = csi_parser.collect_one_exchange(ser)
            if lf is None:
                # collect_one_exchange returns (None, None) on EOF / serial timeout.
                # Back off briefly to avoid busy-looping on a disconnected port.
                eof_count += 1
                if eof_count >= 5:
                    print("[INF] Serial EOF — stopping.")
                    break
                time.sleep(0.1)
                continue
            eof_count = 0
            # collect_one_exchange always returns matched (lf, sf) pairs here.
            # extract_feature_vector handles sf=None defensively if called elsewhere.
            raw_features = csi_parser.extract_feature_vector(lf, sf, raw_feature_names)

            if model is not None:
                if any(math.isnan(v) for v in raw_features if isinstance(v, float)):
                    print(f"[WARN] Skipping prediction — NaN features (MISS frame)")
                    continue
                if trained_names is not None and scaler is not None:
                    features = apply_preprocessing(
                        raw_features, raw_feature_names, trained_names, scaler
                    )
                else:
                    features = raw_features
                features = features + spacing_vals  # append 6 spacing features
                prediction = model.predict([features])
                print(f"Zone: {prediction[0]}")
            else:
                # Dry-run sanity output
                tx = sf['poll_rssi'] if sf else 'N/A'
                print(f"[DRY] mac={lf.get('mac','?')}  poll_seq={lf.get('poll_seq','?')}"
                      f"  rssi={lf.get('rssi','?')}  tx_rssi={tx}")

    except KeyboardInterrupt:
        print("\n[INF] Stopped.")
    finally:
        if ser:
            ser.close()


if __name__ == '__main__':
    main()
