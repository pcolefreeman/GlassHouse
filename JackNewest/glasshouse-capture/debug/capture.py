"""CSI capture tool for GlassHouse v2.

Logs every packet at full rate to a JSON-lines file.

Usage (run from the glasshouse-capture folder):
    python -m debug.capture --port COM9 --seconds 30 --label empty
    python -m debug.capture --port COM9 --seconds 30 --label occupied_q1 --delay 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from python.frame_decoder import parse_packet
from python.serial_receiver import SerialReceiver


def capture(port: str, baud: int, seconds: float, label: str,
            delay: float = 0) -> Path:

    out_path = Path(f"captures/capture_{label}.jsonl")
    out_path.parent.mkdir(exist_ok=True)

    receiver = SerialReceiver(port=port, baud=baud)

    if delay > 0:
        print(f"Get into position for '{label}' capture...")
        for remaining in range(int(delay), 0, -1):
            print(f"  Starting in {remaining}s...", end="\r")
            time.sleep(1)
        print(f"  GO -- capturing now!          ")

    print(f"Capturing to {out_path} for {seconds}s  (label={label})")
    print("Press Ctrl+C to stop early.\n")

    pkt_count = 0
    counts: dict[str, int] = {}
    start = time.monotonic()

    with open(out_path, "w") as f:
        try:
            for packet in receiver.read_packets():
                elapsed = time.monotonic() - start
                if elapsed > seconds:
                    break

                record = {"t": round(elapsed, 4), "label": label}
                record.update(parse_packet(packet))
                counts[record["type"]] = counts.get(record["type"], 0) + 1

                pkt_count += 1
                try:
                    f.write(json.dumps(record) + "\n")
                except (ValueError, TypeError) as exc:
                    print(f"\n  [warn] skipping malformed record: {exc}",
                          file=sys.stderr)

                # Live progress every ~2s
                if pkt_count % 50 == 0:
                    print(f"\r  {elapsed:.1f}s  pkts={pkt_count} "
                          f"links={counts.get('link', 0)} "
                          f"vitals={counts.get('vitals', 0)} "
                          f"iq={counts.get('iq', 0)} "
                          f"csi={counts.get('csi', 0)} "
                          f"hb={counts.get('heartbeat', 0)} "
                          f"unk={counts.get('unknown', 0)}", end="")

        except KeyboardInterrupt:
            pass
        finally:
            receiver.close()

    print(f"\n\nDone: {pkt_count} packets -> {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GlassHouse CSI Capture")
    parser.add_argument("--port", default="COM9",
                        help="Serial port (default: COM9)")
    parser.add_argument("--baud", type=int, default=921600,
                        help="Baud rate (default: 921600)")
    parser.add_argument("--seconds", type=float, default=30,
                        help="Capture duration in seconds (default: 30)")
    parser.add_argument("--label", default="test",
                        help="Label for this capture (e.g. 'empty', 'occupied_q1')")
    parser.add_argument("--delay", type=float, default=0,
                        help="Countdown before capture starts (seconds)")
    args = parser.parse_args()

    capture(args.port, args.baud, args.seconds, args.label, delay=args.delay)
