#!/usr/bin/env python3
"""
Serial CSI Reader — parses ESP32 CSI CSV output and computes
per-subcarrier amplitude in real time.

Expected serial line format (from csi_receiver.ino):
  CSI_DATA,<seq>,<MAC>,<rssi>,<data_len>,<b0> <b1> <b2> ...

Bytes are signed int8 values representing [imaginary, real] pairs
per OFDM subcarrier.  Amplitude = sqrt(imag² + real²).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Optional


def parse_csi_line(line: str) -> Optional[dict]:
    """Parse a single CSV line from the CSI receiver firmware.

    Supports two formats:

    **S01 format (6 fields):**
        ``CSI_DATA,<seq>,<mac>,<rssi>,<data_len>,<bytes...>``
        Returns: ``{seq, mac, rssi, data_len, raw_bytes}``

    **S02 multi-link format (8 fields):**
        ``CSI_DATA,<seq>,<tx_node>,<rx_node>,<link_id>,<rssi>,<data_len>,<bytes...>``
        Returns: ``{seq, tx_node, rx_node, link_id, rssi, data_len, raw_bytes}``

    Format detection: split with maxsplit=7 first.  If 8 fields result
    and field[2] is a single uppercase letter A-D, parse as S02.
    Otherwise fall back to S01 (re-split with maxsplit=5).

    Returns None if the line doesn't match any CSI_DATA format.
    """
    line = line.strip()
    if not line.startswith("CSI_DATA,"):
        return None

    # Try S02 (8-field) format first
    parts8 = line.split(",", 7)
    if len(parts8) == 8 and len(parts8[2]) == 1 and parts8[2].isalpha() and parts8[2].isupper():
        # S02 multi-link format
        try:
            seq = int(parts8[1])
            tx_node = parts8[2]
            rx_node = parts8[3]
            link_id = parts8[4]
            rssi = int(parts8[5])
            data_len = int(parts8[6])
        except (ValueError, IndexError):
            return None

        byte_str = parts8[7].strip()
        if not byte_str:
            raw_bytes: list[int] = []
        else:
            try:
                raw_bytes = [int(b) for b in byte_str.split()]
            except ValueError:
                return None

        return {
            "seq": seq,
            "tx_node": tx_node,
            "rx_node": rx_node,
            "link_id": link_id,
            "rssi": rssi,
            "data_len": data_len,
            "raw_bytes": raw_bytes,
        }

    # Fall back to S01 (6-field) format
    parts = line.split(",", 5)  # CSI_DATA, seq, mac, rssi, data_len, bytes...
    if len(parts) < 6:
        return None

    try:
        seq = int(parts[1])
        mac = parts[2]
        rssi = int(parts[3])
        data_len = int(parts[4])
    except (ValueError, IndexError):
        return None

    # Parse space-separated signed byte values
    byte_str = parts[5].strip()
    if not byte_str:
        raw_bytes = []
    else:
        try:
            raw_bytes = [int(b) for b in byte_str.split()]
        except ValueError:
            return None

    return {
        "seq": seq,
        "mac": mac,
        "rssi": rssi,
        "data_len": data_len,
        "raw_bytes": raw_bytes,
    }


def compute_amplitudes(
    raw_bytes: list[int], skip_first_word: bool = False
) -> list[float]:
    """Compute per-subcarrier amplitude from raw CSI byte pairs.

    Each subcarrier is represented by two consecutive bytes:
    [imaginary, real] as signed int8 values.

    Args:
        raw_bytes: List of signed integer byte values from CSI data.
        skip_first_word: If True, skip the first 4 bytes (useful when
            feeding raw CSI data that includes the invalid first word).
            Default False because the ESP32 firmware already skips them
            before printing.

    Returns:
        List of amplitude values (one per subcarrier).
    """
    start = 4 if skip_first_word else 0
    data = raw_bytes[start:]

    # Need pairs of bytes — truncate if odd
    pair_count = len(data) // 2
    amplitudes: list[float] = []

    for i in range(pair_count):
        imag = _to_signed8(data[2 * i])
        real = _to_signed8(data[2 * i + 1])
        amplitudes.append(math.sqrt(imag * imag + real * real))

    return amplitudes


def _to_signed8(val: int) -> int:
    """Convert a value to signed 8-bit integer range [-128, 127].

    Values already in range are passed through.  Values > 127 are
    interpreted as unsigned and converted (e.g., 255 → -1).
    """
    val = val & 0xFF
    return val - 256 if val > 127 else val


def format_amplitude_summary(
    amplitudes: list[float], max_display: int = 10
) -> str:
    """Format amplitudes as a compact one-line summary.

    Shows first N values, total count, mean, and max.
    """
    if not amplitudes:
        return "[no subcarriers]"

    shown = amplitudes[:max_display]
    vals_str = " ".join(f"{a:.1f}" for a in shown)
    suffix = f" ... ({len(amplitudes)} total)" if len(amplitudes) > max_display else ""

    mean_val = sum(amplitudes) / len(amplitudes)
    max_val = max(amplitudes)

    return f"[{vals_str}{suffix}]  mean={mean_val:.1f}  max={max_val:.1f}"


def main() -> None:
    """Entry point — read serial CSI data and print amplitude summaries."""
    parser = argparse.ArgumentParser(
        description="Read CSI data from ESP32 serial port and display amplitudes"
    )
    parser.add_argument(
        "--port", required=True, help="Serial port (e.g., COM3, /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--baud", type=int, default=921600, help="Baud rate (default: 921600)"
    )
    args = parser.parse_args()

    # Import here so tests can import the module without pyserial installed
    try:
        import serial  # type: ignore[import-untyped]
    except ImportError:
        print(
            "ERROR: pyserial not installed. Run: pip install pyserial",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Opening {args.port} at {args.baud} baud...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as e:
        print(f"ERROR: Could not open {args.port}: {e}", file=sys.stderr)
        sys.exit(1)

    print("Reading CSI data (Ctrl+C to stop)...\n")

    csi_count = 0
    parse_failures = 0

    try:
        while True:
            raw = ser.readline()
            if not raw:
                continue

            try:
                line = raw.decode("utf-8", errors="replace").strip()
            except Exception:
                parse_failures += 1
                continue

            if not line:
                continue

            # Skip non-CSI lines silently (boot messages, diagnostics, etc.)
            if not line.startswith("CSI_DATA,"):
                continue

            result = parse_csi_line(line)
            if result is None:
                parse_failures += 1
                continue

            amplitudes = compute_amplitudes(result["raw_bytes"])
            summary = format_amplitude_summary(amplitudes)

            ts = time.strftime("%H:%M:%S")
            if "link_id" in result:
                print(
                    f"[{ts}] link={result['link_id']}  "
                    f"seq={result['seq']:>5d}  "
                    f"rssi={result['rssi']:>4d}  "
                    f"subs={len(amplitudes):>3d}  {summary}"
                )
            else:
                print(
                    f"[{ts}] seq={result['seq']:>5d}  "
                    f"rssi={result['rssi']:>4d}  "
                    f"subs={len(amplitudes):>3d}  {summary}"
                )
            csi_count += 1

    except KeyboardInterrupt:
        print(f"\n--- Stopped. Received {csi_count} CSI frames, "
              f"{parse_failures} parse failures ---")
    finally:
        ser.close()
        print("Serial port closed.")


if __name__ == "__main__":
    main()
