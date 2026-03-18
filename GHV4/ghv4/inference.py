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
import math
import serial
import time

from ghv4 import csi_parser
from ghv4.config import PAIR_KEYS, BAUD_RATE

SPACING_REFRESH_S = 5


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


def main():
    parser = argparse.ArgumentParser(description="GHV4 live inference")
    parser.add_argument('--port',    required=True,             help="Serial port")
    parser.add_argument('--model',   default=None,              help="Trained model file (.pkl)")
    parser.add_argument('--baud',    type=int, default=BAUD_RATE)
    parser.add_argument('--spacing', default='spacing.json',
                        help="Path to spacing.json (default: spacing.json in cwd)")
    args = parser.parse_args()

    model = None
    if args.model:
        try:
            model = load_model(args.model)
            print(f"[INF] Loaded model: {args.model}")
        except Exception as e:
            print(f"[WARN] Could not load model ({e}) — dry-run mode")
    else:
        print("[INF] No model specified — dry-run mode")

    feature_names = csi_parser.build_feature_names()
    spacing_vals  = load_spacing(args.spacing)
    last_spacing_load = time.time()

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
            features = csi_parser.extract_feature_vector(lf, sf, feature_names)
            features = features + spacing_vals  # append 6 spacing features

            if model is not None:
                if any(math.isnan(v) for v in features if isinstance(v, float)):
                    print(f"[WARN] Skipping prediction — NaN features (MISS frame)")
                    continue
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
