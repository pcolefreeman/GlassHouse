"""Run GHV4 SAR breathing detection."""
import argparse
import json
import logging
import queue
import sys
import time

import numpy as np

from ghv4.breathing import BreathingDetector, reconstruct_csi_from_csv_row
from ghv4.config import (
    BAUD_RATE,
    BREATHING_SLIDE_N,
    BREATHING_WINDOW_N,
    BREATHING_PATH_MAP,
    BUCKET_MS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
_log = logging.getLogger("run_sar")


def _print_console(scores: dict, path_conf: dict):
    """Print a 3x3 grid to the console."""
    ts = time.strftime("%H:%M:%S")
    lines = [
        f"\n=== SAR Breathing Detection ===",
        f"Window: {BREATHING_WINDOW_N * BUCKET_MS / 1000:.0f}s | Updated: {ts}",
        "",
        "      C0     C1     C2",
    ]
    for r in range(3):
        row_str = f"R{r} "
        for c in range(3):
            cell = f"r{r}c{c}"
            val = scores.get(cell)
            if val is None:
                row_str += "[  -- ] "
            else:
                row_str += f"[{val:4.0f}%] "
        lines.append(row_str)
    lines.append("")
    parts = []
    for sid in sorted(path_conf):
        parts.append(f"S{sid}={path_conf[sid]*100:.0f}%")
    lines.append(f"Path confidence: {' '.join(parts)}")
    detected = [f"S{sid}" for sid, c in path_conf.items() if c > 0.3]
    if detected:
        lines.append(f"Status: BREATHING DETECTED ({', '.join(detected)})")
    else:
        lines.append("Status: No breathing detected")
    print("\n".join(lines))


def run_replay(csv_path: str, detector: BreathingDetector, shouter_ids: list):
    """Replay a CSV file through the breathing detector."""
    import pandas as pd
    _log.info("Replaying %s", csv_path)
    df = pd.read_csv(csv_path)
    frames_fed = 0
    for idx, row in df.iterrows():
        for sid in shouter_ids:
            amp_col = f"s{sid}_amp_norm_0"
            if amp_col not in df.columns:
                continue
            csi = reconstruct_csi_from_csv_row(row, shouter_id=sid)
            # Convert complex array to bytes for feed_frame
            import struct
            csi_bytes = b''
            for c in csi:
                csi_bytes += struct.pack('<hh',
                    int(np.real(c) * 1000), int(np.imag(c) * 1000))
            detector.feed_frame('shouter', {
                'shouter_id': sid,
                'csi_bytes': csi_bytes,
            })
        frames_fed += 1
        if frames_fed >= BREATHING_WINDOW_N and frames_fed % BREATHING_SLIDE_N == 0:
            if detector.is_ready():
                scores = detector.get_grid_scores()
                # Compute raw path confidences for display
                path_conf = {}
                for sid_inner in shouter_ids:
                    buf = detector._buffers.get(sid_inner)
                    if buf and buf.is_full():
                        window = buf.get_window()
                        ratio = detector._extractor.extract(window)
                        path_conf[sid_inner] = detector._analyzer.analyze(ratio)
                _print_console(scores, path_conf)
    _log.info("Replay complete: %d rows processed", frames_fed)


def run_live(port: str, detector: BreathingDetector):
    """Run live serial breathing detection."""
    import serial as pyserial
    from ghv4.serial_io import SerialReader

    frame_queue = queue.Queue()
    ser = pyserial.Serial(port, BAUD_RATE, timeout=1.0)
    reader = SerialReader(ser, frame_queue)
    reader.start()
    _log.info("Live mode on %s — waiting for %d frames...", port, BREATHING_WINDOW_N)

    frames_since_update = 0
    try:
        while True:
            try:
                frame_type, frame_dict = frame_queue.get(timeout=1.0)
            except (queue.Empty, ValueError):
                continue
            detector.feed_frame(frame_type, frame_dict)
            frames_since_update += 1
            if frames_since_update >= BREATHING_SLIDE_N and detector.is_ready():
                frames_since_update = 0
                scores = detector.get_grid_scores()
                path_conf = {}
                for sid in detector._buffers:
                    buf = detector._buffers[sid]
                    if buf.is_full():
                        window = buf.get_window()
                        ratio = detector._extractor.extract(window)
                        path_conf[sid] = detector._analyzer.analyze(ratio)
                _print_console(scores, path_conf)
    except KeyboardInterrupt:
        _log.info("Stopped.")
    finally:
        reader.stop()
        ser.close()


def main():
    parser = argparse.ArgumentParser(
        description="GHV4 SAR Breathing Detection — zero-calibration human presence detection"
    )
    parser.add_argument("--port", help="Serial port for live mode (e.g., COM3)")
    parser.add_argument("--replay", help="Path to CSV file for replay mode")
    parser.add_argument("--display", choices=["console", "pygame"], default="console",
                        help="Output display mode (default: console)")
    parser.add_argument("--layout", help="Path to JSON file overriding BREATHING_PATH_MAP")
    args = parser.parse_args()

    if not args.port and not args.replay:
        parser.error("Provide --port for live mode or --replay for CSV replay")

    # Load custom path map if provided
    path_map = None
    if args.layout:
        with open(args.layout) as f:
            raw = json.load(f)
        path_map = {int(k): v for k, v in raw.items()}

    detector = BreathingDetector(path_map=path_map)

    if args.replay:
        shouter_ids = list(detector._buffers.keys())
        run_replay(args.replay, detector, shouter_ids)
    else:
        run_live(args.port, detector)


if __name__ == "__main__":
    main()
