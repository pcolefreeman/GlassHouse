"""Run GHV4 SAR breathing detection."""
import argparse
import json
import logging
import queue
import sys
import threading
import time

from ghv4.breathing import BreathingDetector, GridProjector
from ghv4.config import (
    BAUD_RATE,
    BREATHING_SLIDE_N,
    BREATHING_WINDOW_S,
    BREATHING_PATH_MAP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
_log = logging.getLogger("run_sar")


def _print_console(scores: dict, path_conf: dict):
    """Print a 3x3 grid to the console."""
    ts = time.strftime("%H:%M:%S")
    lines = [
        f"\n=== SAR Breathing Detection ===",
        f"Window: {BREATHING_WINDOW_S:.0f}s | Updated: {ts}",
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
    for k in sorted(path_conf):
        parts.append(f"S{k[0]}↔S{k[1]}={path_conf[k]*100:.0f}%")
    lines.append(f"Path confidence: {' '.join(parts)}")
    detected = [f"S{k[0]}↔S{k[1]}" for k, c in path_conf.items() if c > 0.3]
    if detected:
        lines.append(f"Status: BREATHING DETECTED ({', '.join(detected)})")
    else:
        lines.append("Status: No breathing detected")
    print("\n".join(lines))


def run_live(port: str, detector: BreathingDetector, display_mode: str,
             fullscreen: bool):
    """Run live serial breathing detection."""
    if display_mode == 'pygame':
        _run_pygame_loop(port, detector, fullscreen, demo=False)
    else:
        _run_console_loop(port, detector)


def _run_console_loop(port: str, detector: BreathingDetector):
    """Console-only live mode."""
    import serial as pyserial
    from ghv4.serial_io import SerialReader
    from ghv4.config import BREATHING_WINDOW_N

    frame_queue = queue.Queue()
    ser = pyserial.Serial(port, BAUD_RATE, timeout=1.0)
    reader = SerialReader(ser, frame_queue)
    reader.start()
    _log.info("Live console mode on %s — waiting for %d frames...", port, BREATHING_WINDOW_N)

    frames_since_update = 0
    try:
        while True:
            try:
                frame_type, frame_dict = frame_queue.get(timeout=1.0)
            except (queue.Empty, ValueError):
                continue
            detector.feed_frame(frame_type, frame_dict)
            if frame_type == 'csi_snap':
                frames_since_update += 1
            if frames_since_update >= BREATHING_SLIDE_N and detector.is_ready():
                frames_since_update = 0
                scores = detector.get_grid_scores()
                path_conf = {}
                for key, buf in detector._buffers.items():
                    if buf.is_full():
                        window = buf.get_window()
                        ratio = detector._extractor.extract(window)
                        path_conf[key] = detector._analyzer.analyze(ratio)
                _print_console(scores, path_conf)
    except KeyboardInterrupt:
        _log.info("Stopped.")
    finally:
        reader.stop()
        ser.close()


def _run_pygame_loop(port, detector, fullscreen, demo):
    """Pygame display loop (main thread). Spawns BreathingThread or SARDemoThread."""
    from ghv4.breathing import BreathingDisplay, BreathingThread, SARDemoThread
    if BreathingDisplay is None:
        print("ERROR: pygame is required for --display pygame. Install with: pip install pygame")
        sys.exit(1)

    from ghv4.config import PI_DISPLAY_FPS
    import pygame

    result_queue = queue.Queue()
    stop_event = threading.Event()
    display = BreathingDisplay(fullscreen=fullscreen)

    if demo:
        thread = SARDemoThread(result_queue, stop_event)
    else:
        thread = BreathingThread(port, BAUD_RATE, detector, result_queue, stop_event)
    thread.start()

    clock = pygame.time.Clock()
    running = True
    try:
        while running:
            running = display.handle_events()
            # Drain queue
            latest_scores = None
            try:
                while True:
                    item = result_queue.get_nowait()
                    if item.get("type") == "scores":
                        latest_scores = item
                    elif item.get("type") == "status":
                        display.set_status(item["msg"])
            except queue.Empty:
                pass
            if latest_scores:
                display.update(latest_scores["grid"], latest_scores["path_conf"])
            display.render()
            pygame.display.flip()
            clock.tick(PI_DISPLAY_FPS)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        thread.join(timeout=2.0)
        display.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="GHV4 SAR Breathing Detection — zero-calibration human presence detection"
    )
    parser.add_argument("--port", help="Serial port for live mode (e.g., COM3)")
    parser.add_argument("--display", choices=["console", "pygame"], default="console",
                        help="Output display mode (default: console)")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode: synthetic breathing without hardware")
    parser.add_argument("--fullscreen", action="store_true",
                        help="Run pygame display in fullscreen mode")
    parser.add_argument("--layout", help="Path to JSON file overriding BREATHING_PATH_MAP")
    args = parser.parse_args()

    if not args.port and not args.demo:
        parser.error("Provide --port for live mode or --demo for demo mode")

    # Load custom path map if provided
    path_map = None
    if args.layout:
        with open(args.layout) as f:
            raw = json.load(f)
        # JSON keys are strings like "1,2"; convert to tuples
        path_map = {tuple(map(int, k.split(','))): v for k, v in raw.items()}

    detector = BreathingDetector(path_map=path_map)

    if args.demo:
        if args.display == 'console':
            print("Demo mode requires --display pygame")
            sys.exit(1)
        _run_pygame_loop(None, detector, args.fullscreen, demo=True)
    else:
        run_live(args.port, detector, args.display, args.fullscreen)


if __name__ == "__main__":
    main()
