#!/usr/bin/env python3
"""
Zone Detection GUI — real-time visualization of CSI presence detection
and zone localization.

Pipes serial CSI data through:
    parse → amplitudes → subcarriers → turbulence → PresenceEngine →
    ZoneDetector → ZoneDisplay (pygame GUI)

Usage:
    python main_gui.py --port COM3
    python main_gui.py --port /dev/ttyUSB0 --threshold 0.01 --window 30
"""

from __future__ import annotations

import argparse
import sys
import time

from serial_csi_reader import parse_csi_line, compute_amplitudes
from csi_features import select_subcarriers, compute_turbulence
from presence_detector import PresenceEngine, RoomState
from zone_detector import ZoneDetector, Zone, ZoneResult


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Real-time CSI zone detection GUI — "
        "displays presence status and zone estimation"
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


def process_frame_gui(
    line: str,
    engine: PresenceEngine,
    detector: ZoneDetector,
) -> dict | None:
    """Process one serial line and return GUI-ready state dict.

    This is a pure function (no pygame or serial dependency) for easy
    testing.  Returns None if the line is not a valid S02 CSI_DATA line.

    Returns:
        Dict with keys:
            - room_state: RoomState
            - zone_result: ZoneResult from detector.estimate()
            - link_states: dict from engine.get_link_states()
            - link_id: str — which link this frame was for
            - turbulence: float — turbulence value for this frame
        Or None if the line was not processable.
    """
    parsed = parse_csi_line(line)
    if parsed is None:
        return None

    # S02 format has link_id; S01 format doesn't — skip S01 lines
    link_id = parsed.get("link_id")
    if link_id is None:
        return None

    amplitudes = compute_amplitudes(parsed["raw_bytes"])
    selected = select_subcarriers(amplitudes)
    turb = compute_turbulence(selected)

    try:
        new_state = engine.update(link_id, turb)
    except KeyError:
        # Unrecognized link_id — skip silently
        return None

    zone_result = detector.estimate()
    link_states = engine.get_link_states()

    return {
        "room_state": new_state,
        "zone_result": zone_result,
        "link_states": link_states,
        "link_id": link_id,
        "turbulence": turb,
    }


def main() -> None:
    """Entry point — read serial CSI data and render on pygame GUI."""
    parser = build_parser()
    args = parser.parse_args()

    # Lazy imports so tests can import this module without these packages
    try:
        import serial  # type: ignore[import-untyped]
    except ImportError:
        print(
            "ERROR: pyserial not installed. Run: pip install pyserial",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from gui_zone import ZoneDisplay
    except ImportError:
        print(
            "ERROR: pygame not installed. Run: pip install pygame-ce",
            file=sys.stderr,
        )
        sys.exit(1)

    engine = PresenceEngine(
        window_size=args.window,
        threshold=args.threshold,
    )
    detector = ZoneDetector(engine)
    display = ZoneDisplay()

    print(f"Opening {args.port} at {args.baud} baud...")
    print(f"Detection: window={args.window} threshold={args.threshold}")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.01)
    except serial.SerialException as e:
        print(f"ERROR: Could not open {args.port}: {e}", file=sys.stderr)
        display.close()
        sys.exit(1)

    print("GUI started — monitoring for presence + zone (close window or Ctrl+C to stop)...\n")

    csi_count = 0
    parse_failures = 0
    prev_state = RoomState.EMPTY
    prev_zone: Zone | None = None
    fps_timer = time.monotonic()
    fps_count = 0
    fps_display = 0.0

    try:
        running = True
        while running:
            # Read available serial data (non-blocking with short timeout)
            raw = ser.readline()
            if raw:
                try:
                    line = raw.decode("utf-8", errors="replace").strip()
                except Exception:
                    parse_failures += 1
                    continue

                if line and line.startswith("CSI_DATA,"):
                    result = process_frame_gui(line, engine, detector)
                    if result is None:
                        parse_failures += 1
                    else:
                        csi_count += 1
                        fps_count += 1

                        # Print on state/zone change
                        new_state = result["room_state"]
                        zone_result = result["zone_result"]
                        new_zone = (
                            zone_result.zone
                            if new_state == RoomState.OCCUPIED
                            else None
                        )

                        if new_state != prev_state or new_zone != prev_zone:
                            ts = time.strftime("%H:%M:%S")
                            zone_str = (
                                new_zone.value if new_zone else "--"
                            )
                            print(
                                f"[{ts}] {new_state.value} | Zone: {zone_str}"
                            )
                            prev_state = new_state
                            prev_zone = new_zone

            # Compute FPS every second
            now = time.monotonic()
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                fps_display = fps_count / elapsed
                fps_count = 0
                fps_timer = now

            # Update GUI (processes pygame events too)
            zone_result = detector.estimate()
            link_states = engine.get_link_states()
            running = display.update(
                engine.room_state,
                zone_result,
                link_states,
                fps=fps_display,
            )

    except KeyboardInterrupt:
        print(
            f"\n--- Stopped. Processed {csi_count} CSI frames, "
            f"{parse_failures} parse failures ---"
        )
    finally:
        display.close()
        ser.close()
        print("GUI and serial port closed.")


if __name__ == "__main__":
    main()
