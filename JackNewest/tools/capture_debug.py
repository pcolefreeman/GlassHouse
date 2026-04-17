#!/usr/bin/env python3
"""Dump raw link reports from coordinator serial for diagnostics."""
import sys
import struct
import time

import serial
from cobs import cobs as cobs_codec

sys.path.insert(0, "python")

# All 6 expected pairwise links (4 perimeter nodes)
EXPECTED_LINKS = {"12", "13", "14", "23", "24", "34"}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Capture and display link reports")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--duration", type=int, default=30)
    args = parser.parse_args()

    # Open without toggling DTR/RTS — on ESP32-S3 USB-Serial/JTAG,
    # DTR transitions can reset the chip, killing the coordinator mid-run.
    ser = serial.Serial()
    ser.port = args.port
    ser.baudrate = args.baud
    ser.timeout = 0.5
    ser.dtr = False
    ser.rts = False
    ser.open()
    # Flush any boot log garbage already in the buffer
    time.sleep(0.5)
    ser.reset_input_buffer()
    start = time.time()
    deadline = start + args.duration
    count = 0
    seen_links: set[str] = set()
    buf = bytearray()

    print(f"Listening on {args.port} for {args.duration}s...")

    try:
        while time.time() < deadline:
            chunk = ser.read(256)
            if chunk:
                buf.extend(chunk)
            while b'\x00' in buf:
                idx = buf.index(b'\x00')
                frame = bytes(buf[:idx])
                buf = buf[idx + 1:]
                if len(frame) == 0:
                    continue
                try:
                    pkt = cobs_codec.decode(frame)
                except Exception:
                    continue
                if len(pkt) == 10 and pkt[0] == 0x01:
                    _, node, partner, var, state, samples = struct.unpack(
                        "<BBBfBH", pkt[:10]
                    )
                    if not (1 <= node <= 4 and 1 <= partner <= 4 and node != partner):
                        continue  # Skip malformed report
                    link_id = f"{min(node, partner)}{max(node, partner)}"
                    seen_links.add(link_id)
                    state_str = "MOTION" if state else "IDLE"
                    elapsed = time.time() - start
                    print(
                        f"[{elapsed:6.1f}s] link {link_id}: "
                        f"var={var:.6f} {state_str} n={samples}"
                    )
                    count += 1
                elif len(pkt) >= 4 and pkt[:4] == b"\xC5\x11\x00\x02":
                    presence = pkt[8] if len(pkt) > 8 else 0
                    print(
                        f"[{time.time() - start:6.1f}s] VITALS: "
                        f"presence={'YES' if presence else 'NO'}"
                    )
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()

    # Audit SR-1: Link coverage summary
    missing = EXPECTED_LINKS - seen_links
    print(f"\n{'=' * 40}")
    print(f"{count} link reports in {time.time() - start:.1f}s")
    print(f"Links seen: {sorted(seen_links)} ({len(seen_links)}/6)")
    if missing:
        print(f"Links MISSING: {sorted(missing)}")
    else:
        print("All 6 links detected!")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
