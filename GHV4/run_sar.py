"""Run GHV4 SAR breathing detection."""
import argparse
import json
import logging
import queue
import sys
import threading
import time

from ghv4.breathing import BreathingDetector
from ghv4.config import (
    BAUD_RATE,
    BREATHING_SLIDE_N,
    BREATHING_WINDOW_S,
    BREATHING_WINDOW_N,
    BREATHING_CONFIDENCE_THRESHOLD,
    BREATHING_MIN_PATHS_TOTAL,
    PI_DISPLAY_FPS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
_log = logging.getLogger("run_sar")


def _print_console(scores: dict, path_conf: dict, hr_conf: dict | None = None):
    """Print SAR vital sign status table to console."""
    ts = time.strftime("%H:%M:%S")
    hr_conf = hr_conf or {}
    lines = [
        f"\n=== GHV4-SAR: Vital Sign Detector === ({ts})",
        "",
        f"{'Path':<12}{'Presence':>10}{'Breathing':>16}{'Heart Rate':>16}{'Status':>14}",
    ]
    for key in sorted(path_conf):
        label = f"S{key[0]}↔S{key[1]}"
        presence = path_conf[key] * 100
        breathing = path_conf[key] * 100
        hr_c, hr_bpm = hr_conf.get(key, (0.0, 0.0))
        hr_pct = hr_c * 100

        br_str = f"{breathing:.0f}%"
        hr_str = f"{hr_pct:.0f}%"
        if hr_bpm > 0:
            hr_str += f" ({hr_bpm:.0f} BPM)"

        vital = max(breathing, hr_pct)
        if vital > BREATHING_CONFIDENCE_THRESHOLD * 100:
            status = "■ VITAL SIGNS" if hr_pct > 20 else "■ BREATHING"
        else:
            status = "· quiet"

        lines.append(f"{label:<12}{presence:>9.0f}%{br_str:>15}{hr_str:>15}  {status}")

    detected = [f"S{k[0]}↔S{k[1]}" for k, v in path_conf.items()
                if v > BREATHING_CONFIDENCE_THRESHOLD]
    if detected:
        lines.append(f"\n>> SURVIVOR DETECTED — paths {', '.join(detected)}")
    else:
        lines.append("\n>> No vital signs detected")
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

    frame_queue = queue.Queue()
    try:
        ser = pyserial.Serial(port, BAUD_RATE, timeout=1.0)
    except pyserial.SerialException as exc:
        _log.error("Cannot open serial port %s: %s", port, exc)
        sys.exit(1)

    reader = SerialReader(ser, frame_queue)
    reader.start()
    _log.info("Live console mode on %s — waiting for %d frames...", port, BREATHING_WINDOW_N)

    frames_since_update = 0
    last_fill_log = time.time()
    try:
        while True:
            try:
                frame_type, frame_dict = frame_queue.get(timeout=1.0)
            except (queue.Empty, ValueError):
                # Log fill status periodically even with no data
                now = time.time()
                if now - last_fill_log >= 5.0:
                    last_fill_log = now
                    fill = detector.get_buffer_fill()
                    parts = [f"S{k[0]}↔S{k[1]}={v*100:.0f}%"
                             for k, v in sorted(fill.items())]
                    _log.info("Buffer fill: %s", " ".join(parts))
                # Check if serial port is still open
                if not ser.is_open:
                    _log.error("Serial port %s disconnected.", port)
                    break
                continue

            detector.feed_frame(frame_type, frame_dict)
            if frame_type == 'csi_snap':
                frames_since_update += 1

            # Log fill status every 5s
            now = time.time()
            if now - last_fill_log >= 5.0:
                last_fill_log = now
                fill = detector.get_buffer_fill()
                parts = [f"S{k[0]}↔S{k[1]}={v*100:.0f}%"
                         for k, v in sorted(fill.items())]
                _log.info("Buffer fill: %s", " ".join(parts))

            if frames_since_update >= BREATHING_SLIDE_N and detector.is_ready():
                frames_since_update = 0
                scores = detector.get_grid_scores()
                path_conf = detector._last_path_conf
                _print_console(scores, path_conf,
                               getattr(detector, '_last_hr_conf', {}))
    except KeyboardInterrupt:
        _log.info("Stopped.")
    except OSError as exc:
        _log.error("Serial I/O error on %s: %s", port, exc)
    finally:
        reader.stop()
        try:
            ser.close()
        except Exception:
            pass


def _run_pygame_loop(port, detector, fullscreen, demo):
    """Pygame display loop (main thread). Spawns BreathingThread or SARDemoThread."""
    from ghv4.breathing import BreathingDisplay, BreathingThread, SARDemoThread
    if BreathingDisplay is None:
        print("ERROR: pygame is required for --display pygame. Install with: pip install pygame")
        sys.exit(1)

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
                    elif item.get("type") == "fill":
                        display.update_fill(item["fill"])
            except queue.Empty:
                pass
            if latest_scores:
                display.update(latest_scores["grid"], latest_scores["path_conf"])
                if hasattr(display, 'update_hr') and "hr_conf" in latest_scores:
                    display.update_hr(latest_scores["hr_conf"])
            display.render()
            pygame.display.flip()
            clock.tick(PI_DISPLAY_FPS)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        thread.join(timeout=2.0)
        display.cleanup()


def _run_demo_console(detector):
    """Console demo mode: synthetic vital signs cycling across paths."""
    from ghv4.breathing import SARDemoThread
    result_queue = queue.Queue()
    stop_event = threading.Event()
    thread = SARDemoThread(result_queue, stop_event)
    thread.start()
    try:
        while True:
            try:
                item = result_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item.get("type") == "scores":
                _print_console(item["grid"], item["path_conf"],
                               item.get("hr_conf", {}))
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        thread.join(timeout=2.0)


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
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"],
                        help="Logging level (default: INFO)")
    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if not args.port and not args.demo:
        parser.error("Provide --port for live mode or --demo for demo mode")

    # Load custom path map if provided
    path_map = None
    if args.layout:
        try:
            with open(args.layout) as f:
                raw = json.load(f)
            # JSON keys are strings like "1,2"; convert to tuples
            path_map = {tuple(map(int, k.split(','))): v for k, v in raw.items()}
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            _log.error("Failed to load layout file %s: %s", args.layout, exc)
            sys.exit(1)

    detector = BreathingDetector(path_map=path_map)

    if args.demo:
        if args.display == 'pygame':
            _run_pygame_loop(None, detector, args.fullscreen, demo=True)
        else:
            _run_demo_console(detector)
    else:
        run_live(args.port, detector, args.display, args.fullscreen)


if __name__ == "__main__":
    main()
