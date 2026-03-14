"""InferenceV2.py — GHV2 live inference skeleton.

Reads bidirectional CSI frames from Serial, extracts features, runs trained model.
Run in dry-run mode (no --model arg) to verify the pipeline without a trained model.

Usage:
    python InferenceV2.py --port COM3 --model model.pkl
    python InferenceV2.py --port COM3                    # dry-run
"""
import argparse
import serial
import time
import csi_parser


def load_model(path: str):
    """Load a model file. Supports joblib (sklearn) by default.
    Replace with tflite/onnx loading as needed.
    """
    import joblib
    return joblib.load(path)


def main():
    parser = argparse.ArgumentParser(description="GHV2 live inference")
    parser.add_argument('--port',  required=True,             help="Serial port")
    parser.add_argument('--model', default=None,              help="Trained model file (.pkl)")
    parser.add_argument('--baud',  type=int, default=921600)
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

    # NOTE: ser is initialised to None so the finally block can always call
    # ser.close() safely, even if Serial() raises serial.SerialException.
    ser = None
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
        print(f"[INF] Listening on {args.port} — Ctrl+C to stop\n")
        eof_count = 0
        while True:
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
            # sf may be None if only a listener frame was received (no shouter
            # response). extract_feature_vector handles sf=None by filling NaN.
            features = csi_parser.extract_feature_vector(lf, sf, feature_names)

            if model is not None:
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
