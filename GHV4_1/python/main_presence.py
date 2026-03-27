#!/usr/bin/env python3
"""
Presence Detection CLI — wires serial CSI reader → parser → feature
extraction → presence engine → console output.

Prints OCCUPIED or EMPTY to console on state changes only, with a
per-link detail line showing which links are disturbed.

Usage:
    python main_presence.py --port COM3
    python main_presence.py --port /dev/ttyUSB0 --threshold 0.01 --window 30
"""

from __future__ import annotations

import argparse
import sys
import time

from serial_csi_reader import parse_csi_line, compute_amplitudes
from csi_features import select_subcarriers, compute_turbulence
from presence_detector import PresenceEngine, RoomState


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Real-time CSI presence detection — prints OCCUPIED or EMPTY"
    )
    parser.add_argument(
        "--port", required=True, help="Serial port (e.g., COM3, /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--baud", type=int, default=921600, help="Baud rate (default: 921600)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.005,
        help="Variance threshold for motion detection (default: 0.005)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Moving variance window size in samples (default: 20)",
    )
    return parser


def format_link_detail(engine: PresenceEngine) -> str:
    """Format a per-link status line for display on state change.

    Example: ``  Links: AB=MOTION(0.0142) AC=IDLE(0.0003) ...``
    """
    parts: list[str] = []
    for link_id, status in engine.get_link_states().items():
        state = status["state"]
        variance = status["variance"]
        parts.append(f"{link_id}={state}({variance:.4f})")
    return "  Links: " + " ".join(parts)


def process_line(
    line: str,
    engine: PresenceEngine,
    prev_state: RoomState,
) -> tuple[RoomState, bool]:
    """Process one serial line through the full detection pipeline.

    Returns:
        Tuple of (current_room_state, was_csi_line).
        ``was_csi_line`` is False if the line was not a CSI_DATA line
        or failed to parse.
    """
    parsed = parse_csi_line(line)
    if parsed is None:
        return prev_state, False

    # S02 format has link_id; S01 format doesn't — skip S01 lines
    link_id = parsed.get("link_id")
    if link_id is None:
        return prev_state, False

    amplitudes = compute_amplitudes(parsed["raw_bytes"])
    selected = select_subcarriers(amplitudes)
    turb = compute_turbulence(selected)

    try:
        new_state = engine.update(link_id, turb)
    except KeyError:
        # Unrecognized link_id — skip silently
        return prev_state, False

    return new_state, True


def main() -> None:
    """Entry point — read serial CSI data and print presence state changes."""
    parser = build_parser()
    args = parser.parse_args()

    # Lazy import so tests can import this module without pyserial
    try:
        import serial  # type: ignore[import-untyped]
    except ImportError:
        print(
            "ERROR: pyserial not installed. Run: pip install pyserial",
            file=sys.stderr,
        )
        sys.exit(1)

    engine = PresenceEngine(
        window_size=args.window,
        threshold=args.threshold,
    )

    print(f"Opening {args.port} at {args.baud} baud...")
    print(f"Detection: window={args.window} threshold={args.threshold}")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as e:
        print(f"ERROR: Could not open {args.port}: {e}", file=sys.stderr)
        sys.exit(1)

    print("Monitoring for presence (Ctrl+C to stop)...\n")

    csi_count = 0
    parse_failures = 0
    prev_state = RoomState.EMPTY

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

            # Skip non-CSI lines silently
            if not line.startswith("CSI_DATA,"):
                continue

            new_state, was_csi = process_line(line, engine, prev_state)

            if not was_csi:
                parse_failures += 1
                continue

            csi_count += 1

            if new_state != prev_state:
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] {new_state.value}")
                print(format_link_detail(engine))
                prev_state = new_state

    except KeyboardInterrupt:
        print(
            f"\n--- Stopped. Processed {csi_count} CSI frames, "
            f"{parse_failures} parse failures ---"
        )
    finally:
        ser.close()
        print("Serial port closed.")


if __name__ == "__main__":
    main()
