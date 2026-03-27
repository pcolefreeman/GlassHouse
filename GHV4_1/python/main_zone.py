#!/usr/bin/env python3
"""
Zone Detection CLI — extends the presence pipeline with room quadrant
estimation.

Pipes serial CSI data through:
    parse → amplitudes → subcarriers → turbulence → PresenceEngine →
    ZoneDetector → console output

Prints OCCUPIED/EMPTY state changes with zone estimation detail showing
which quadrant (Q1-Q4) the person is likely in.

Usage:
    python main_zone.py --port COM3
    python main_zone.py --port /dev/ttyUSB0 --threshold 0.01 --window 30
"""

from __future__ import annotations

import argparse
import sys
import time

from serial_csi_reader import parse_csi_line, compute_amplitudes
from csi_features import select_subcarriers, compute_turbulence
from presence_detector import PresenceEngine, RoomState
from zone_detector import ZoneDetector, Zone


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Real-time CSI zone detection — prints OCCUPIED/EMPTY with quadrant estimation"
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


def format_zone_detail(detector: ZoneDetector) -> str:
    """Format a zone estimation line for display on state/zone change.

    Example::

        Zone: Q1 (confidence: 2.3x) | Q1=0.0420 Q2=0.0180 Q3=0.0050 Q4=0.0030

    When no zone is estimated (room EMPTY or windows not full), shows::

        Zone: -- (no estimate) | Q1=0.0000 Q2=0.0000 Q3=0.0000 Q4=0.0000
    """
    result = detector.estimate()
    score_parts = " ".join(
        f"{z.value}={result.scores.get(z, 0.0):.4f}" for z in Zone
    )

    if result.zone is None:
        return f"  Zone: -- (no estimate) | {score_parts}"

    if result.confidence == float("inf"):
        conf_str = "inf"
    else:
        conf_str = f"{result.confidence:.1f}x"

    return f"  Zone: {result.zone.value} (confidence: {conf_str}) | {score_parts}"


def process_line_zone(
    line: str,
    engine: PresenceEngine,
    detector: ZoneDetector,
    prev_state: RoomState,
    prev_zone: Zone | None,
) -> tuple[RoomState, Zone | None, bool]:
    """Process one serial line through the full zone detection pipeline.

    Returns:
        Tuple of (current_room_state, current_zone, was_csi_line).
        ``was_csi_line`` is False if the line was not a valid S02 CSI_DATA
        line or failed to parse.
    """
    parsed = parse_csi_line(line)
    if parsed is None:
        return prev_state, prev_zone, False

    # S02 format has link_id; S01 format doesn't — skip S01 lines
    link_id = parsed.get("link_id")
    if link_id is None:
        return prev_state, prev_zone, False

    amplitudes = compute_amplitudes(parsed["raw_bytes"])
    selected = select_subcarriers(amplitudes)
    turb = compute_turbulence(selected)

    try:
        new_state = engine.update(link_id, turb)
    except KeyError:
        # Unrecognized link_id — skip silently
        return prev_state, prev_zone, False

    # Zone estimation: only meaningful when room is OCCUPIED
    zone_result = detector.estimate()
    if new_state == RoomState.OCCUPIED and zone_result.zone is not None:
        current_zone = zone_result.zone
    else:
        current_zone = None

    return new_state, current_zone, True


def main() -> None:
    """Entry point — read serial CSI data and print zone detection output."""
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
    detector = ZoneDetector(engine)

    print(f"Opening {args.port} at {args.baud} baud...")
    print(f"Detection: window={args.window} threshold={args.threshold}")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as e:
        print(f"ERROR: Could not open {args.port}: {e}", file=sys.stderr)
        sys.exit(1)

    print("Monitoring for presence + zone (Ctrl+C to stop)...\n")

    csi_count = 0
    parse_failures = 0
    prev_state = RoomState.EMPTY
    prev_zone: Zone | None = None

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

            new_state, new_zone, was_csi = process_line_zone(
                line, engine, detector, prev_state, prev_zone,
            )

            if not was_csi:
                parse_failures += 1
                continue

            csi_count += 1

            # Print on state change OR zone change
            state_changed = new_state != prev_state
            zone_changed = new_zone != prev_zone

            if state_changed or zone_changed:
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] {new_state.value}")
                from main_presence import format_link_detail
                print(format_link_detail(engine))
                print(format_zone_detail(detector))
                prev_state = new_state
                prev_zone = new_zone

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
