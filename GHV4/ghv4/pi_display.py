"""pi_display.py — Lightweight pygame grid display for Raspberry Pi inference.

Shows a 3×3 zone grid with the predicted cell highlighted in real time.
Inference runs in a background thread; pygame renders in the main thread.

Usage:
    python run_pi_display.py --port /dev/ttyUSB0 --model models/model.pkl --fullscreen
    python run_pi_display.py --demo          # cycle cells without hardware
"""
import argparse
import math
import queue
import random
import threading
import time

import pygame

from ghv4 import csi_parser
from ghv4.config import (
    BAUD_RATE,
    CELL_LABELS,
    PAIR_KEYS,
    SPACING_FEATURE_NAMES,
    PI_DISPLAY_FPS,
    PI_DISPLAY_BG,
    PI_CELL_ACTIVE,
    PI_CELL_INACTIVE,
    PI_CELL_BORDER,
    PI_TEXT_ACTIVE,
    PI_TEXT_INACTIVE,
    PI_SCREEN_SIZE,
)
from ghv4.inference import (
    load_model,
    load_spacing,
    load_preprocessor,
    apply_preprocessing,
)

# Shouter corner positions relative to grid (row, col) — matches viz.py
_SHOUTER_CORNERS = {
    1: (2, 0),   # bottom-left  (row 2, col 0)
    2: (0, 0),   # top-left     (row 0, col 0)
    3: (0, 2),   # top-right    (row 0, col 2)
    4: (2, 2),   # bottom-right (row 2, col 2)
}

SPACING_REFRESH_S = 5


# ---------------------------------------------------------------------------
# Inference thread
# ---------------------------------------------------------------------------
class InferenceThread(threading.Thread):
    """Background thread: serial → parse → predict → queue."""

    def __init__(self, port, baud, model_path, spacing_path, processed_dir,
                 result_queue, stop_event):
        super().__init__(daemon=True)
        self._port = port
        self._baud = baud
        self._model_path = model_path
        self._spacing_path = spacing_path
        self._processed_dir = processed_dir
        self._q = result_queue
        self._stop = stop_event

    def run(self):
        import serial as pyserial

        model = None
        if self._model_path:
            try:
                model = load_model(self._model_path)
                self._q.put({"type": "status", "msg": f"Model loaded: {self._model_path}"})
            except Exception as e:
                self._q.put({"type": "status", "msg": f"Model load failed: {e}"})

        raw_names = csi_parser.build_feature_names()
        spacing_vals = load_spacing(self._spacing_path)
        last_spacing = time.time()

        trained_names, scaler = load_preprocessor(self._processed_dir)

        ser = None
        try:
            ser = pyserial.Serial(self._port, self._baud, timeout=1)
            self._q.put({"type": "status", "msg": f"Connected: {self._port}"})

            eof_count = 0
            while not self._stop.is_set():
                # Refresh spacing periodically
                if time.time() - last_spacing >= SPACING_REFRESH_S:
                    spacing_vals = load_spacing(self._spacing_path)
                    last_spacing = time.time()

                lf, sf = csi_parser.collect_one_exchange(ser)
                if lf is None:
                    eof_count += 1
                    if eof_count >= 5:
                        self._q.put({"type": "status", "msg": "Serial EOF"})
                        break
                    time.sleep(0.1)
                    continue
                eof_count = 0

                raw_features = csi_parser.extract_feature_vector(lf, sf, raw_names)

                if model is None:
                    # Dry-run: no prediction
                    continue

                if any(math.isnan(v) for v in raw_features if isinstance(v, float)):
                    continue

                if trained_names is not None and scaler is not None:
                    features = apply_preprocessing(
                        raw_features, raw_names, trained_names, scaler
                    )
                else:
                    features = raw_features
                features = features + spacing_vals

                # Predict with confidence if available
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba([features])[0]
                    cell_idx = int(proba.argmax())
                    cell = CELL_LABELS[cell_idx]
                    confidence = float(proba[cell_idx])
                else:
                    prediction = model.predict([features])
                    cell = prediction[0]
                    confidence = 1.0

                self._q.put({
                    "type": "prediction",
                    "cell": cell,
                    "confidence": confidence,
                    "timestamp": time.time(),
                })

        except Exception as e:
            self._q.put({"type": "status", "msg": f"Error: {e}"})
        finally:
            if ser:
                ser.close()


# ---------------------------------------------------------------------------
# Demo thread (no hardware needed)
# ---------------------------------------------------------------------------
class DemoThread(threading.Thread):
    """Cycles through cells with random confidence for testing/demos."""

    def __init__(self, result_queue, stop_event):
        super().__init__(daemon=True)
        self._q = result_queue
        self._stop = stop_event

    def run(self):
        idx = 0
        self._q.put({"type": "status", "msg": "Demo mode"})
        while not self._stop.is_set():
            cell = CELL_LABELS[idx % 9]
            confidence = round(random.uniform(0.6, 1.0), 2)
            self._q.put({
                "type": "prediction",
                "cell": cell,
                "confidence": confidence,
                "timestamp": time.time(),
            })
            idx += 1
            # Wait ~2s, checking stop every 100ms
            for _ in range(20):
                if self._stop.is_set():
                    return
                time.sleep(0.1)


# ---------------------------------------------------------------------------
# Grid display
# ---------------------------------------------------------------------------
class GridDisplay:
    """Pygame-based 3×3 zone grid for operator display."""

    TITLE_H = 44
    STATUS_H = 40
    GRID_PAD = 24
    CELL_GAP = 4

    def __init__(self, screen_size=PI_SCREEN_SIZE, fullscreen=False):
        self._screen_size = screen_size
        self._fullscreen = fullscreen
        self._current_cell = None
        self._confidence = 0.0
        self._last_update_time = None
        self._status_msg = "Waiting..."
        self._cell_rects = {}  # (row, col) -> pygame.Rect
        self._shouter_positions = {}  # sid -> (x, y) screen coords

        self._init_pygame()
        self._compute_layout()

    def _init_pygame(self):
        pygame.init()
        flags = pygame.FULLSCREEN if self._fullscreen else 0
        self._screen = pygame.display.set_mode(self._screen_size, flags)
        pygame.display.set_caption("GlassHouse V4 — Zone Tracker")

        # Fonts: try monospace, fall back to pygame default
        try:
            self._font_cell = pygame.font.SysFont("monospace", 32, bold=True)
            self._font_conf = pygame.font.SysFont("monospace", 22)
            self._font_title = pygame.font.SysFont("monospace", 24, bold=True)
            self._font_status = pygame.font.SysFont("monospace", 18)
            self._font_shouter = pygame.font.SysFont("monospace", 14, bold=True)
        except Exception:
            self._font_cell = pygame.font.Font(None, 36)
            self._font_conf = pygame.font.Font(None, 26)
            self._font_title = pygame.font.Font(None, 28)
            self._font_status = pygame.font.Font(None, 22)
            self._font_shouter = pygame.font.Font(None, 18)

    def _compute_layout(self):
        """Compute cell rectangles and shouter marker positions."""
        w, h = self._screen_size

        # Available area for the grid
        grid_top = self.TITLE_H + self.GRID_PAD
        grid_bottom = h - self.STATUS_H - self.GRID_PAD
        grid_h = grid_bottom - grid_top
        grid_w = min(grid_h, w - 2 * self.GRID_PAD)  # keep square-ish
        grid_left = (w - grid_w) // 2

        cell_w = (grid_w - 2 * self.CELL_GAP) // 3
        cell_h = (grid_h - 2 * self.CELL_GAP) // 3

        for row in range(3):
            for col in range(3):
                x = grid_left + col * (cell_w + self.CELL_GAP)
                y = grid_top + row * (cell_h + self.CELL_GAP)
                self._cell_rects[(row, col)] = pygame.Rect(x, y, cell_w, cell_h)

        # Shouter corner markers — just outside grid corners
        margin = 14
        self._shouter_positions = {
            2: (grid_left - margin, grid_top - margin),                       # S2 top-left
            3: (grid_left + grid_w + margin, grid_top - margin),              # S3 top-right
            1: (grid_left - margin, grid_top + grid_h + margin),              # S1 bottom-left
            4: (grid_left + grid_w + margin, grid_top + grid_h + margin),     # S4 bottom-right
        }

        # Store grid bounds for reference
        self._grid_rect = pygame.Rect(grid_left, grid_top, grid_w, grid_h)

    def update(self, cell, confidence):
        """Set the currently active cell and confidence."""
        self._current_cell = cell
        self._confidence = confidence
        self._last_update_time = time.time()

    def set_status(self, msg):
        """Set the status bar message."""
        self._status_msg = msg

    def render(self):
        """Draw the full display: title, grid, status bar."""
        self._screen.fill(PI_DISPLAY_BG)
        self._draw_title()
        self._draw_grid()
        self._draw_shouter_markers()
        self._draw_status_bar()

    def _draw_title(self):
        w = self._screen_size[0]
        text = self._font_title.render("GlassHouse V4 — Zone Tracker",
                                       True, PI_TEXT_ACTIVE)
        rect = text.get_rect(center=(w // 2, self.TITLE_H // 2))
        self._screen.blit(text, rect)

        # Separator line
        y = self.TITLE_H - 1
        pygame.draw.line(self._screen, PI_CELL_BORDER, (0, y), (w, y))

    def _draw_grid(self):
        for (row, col), rect in self._cell_rects.items():
            label = f"r{row}c{col}"
            is_active = (label == self._current_cell)

            # Cell fill
            fill = PI_CELL_ACTIVE if is_active else PI_CELL_INACTIVE
            pygame.draw.rect(self._screen, fill, rect, border_radius=6)

            # Cell border
            pygame.draw.rect(self._screen, PI_CELL_BORDER, rect, width=2,
                             border_radius=6)

            # Cell label
            text_color = PI_TEXT_ACTIVE if is_active else PI_TEXT_INACTIVE
            label_surf = self._font_cell.render(label, True, text_color)

            if is_active and self._confidence < 1.0:
                # Show label above center, confidence below
                label_rect = label_surf.get_rect(
                    center=(rect.centerx, rect.centery - 14))
                self._screen.blit(label_surf, label_rect)

                conf_text = f"{self._confidence:.0%}"
                conf_surf = self._font_conf.render(conf_text, True, text_color)
                conf_rect = conf_surf.get_rect(
                    center=(rect.centerx, rect.centery + 18))
                self._screen.blit(conf_surf, conf_rect)
            else:
                # Center label
                label_rect = label_surf.get_rect(center=rect.center)
                self._screen.blit(label_surf, label_rect)

    def _draw_shouter_markers(self):
        cyan = (0, 200, 200)
        for sid, (x, y) in self._shouter_positions.items():
            pygame.draw.circle(self._screen, cyan, (x, y), 8)
            label = self._font_shouter.render(f"S{sid}", True, cyan)
            label_rect = label.get_rect(center=(x, y - 16))
            self._screen.blit(label, label_rect)

    def _draw_status_bar(self):
        w, h = self._screen_size
        bar_y = h - self.STATUS_H

        # Separator line
        pygame.draw.line(self._screen, PI_CELL_BORDER, (0, bar_y), (w, bar_y))

        parts = [self._status_msg]
        if self._current_cell and self._last_update_time:
            ts = time.strftime("%H:%M:%S", time.localtime(self._last_update_time))
            parts.append(f"Last: {self._current_cell} @ {ts}")

        status_text = "  |  ".join(parts)
        surf = self._font_status.render(status_text, True, PI_TEXT_INACTIVE)
        rect = surf.get_rect(midleft=(12, bar_y + self.STATUS_H // 2))
        self._screen.blit(surf, rect)

    def handle_events(self):
        """Process pygame events. Returns False if the display should close."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
        return True

    def cleanup(self):
        """Shut down pygame."""
        pygame.quit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="GHV4 Pi LCD inference display")
    parser.add_argument('--port', default=None,
                        help="Serial port (required unless --demo)")
    parser.add_argument('--model', default=None,
                        help="Trained model .pkl file")
    parser.add_argument('--baud', type=int, default=BAUD_RATE)
    parser.add_argument('--spacing', default='spacing.json',
                        help="Path to spacing.json")
    parser.add_argument('--processed-dir', default='data/processed/',
                        help="Path to processed data dir")
    parser.add_argument('--fullscreen', action='store_true',
                        help="Run in fullscreen mode")
    parser.add_argument('--demo', action='store_true',
                        help="Demo mode: cycle cells without hardware")
    args = parser.parse_args()

    if not args.demo and not args.port:
        parser.error("--port is required unless --demo is set")

    result_queue = queue.Queue()
    stop_event = threading.Event()

    display = GridDisplay(fullscreen=args.fullscreen)

    if args.demo:
        thread = DemoThread(result_queue, stop_event)
    else:
        thread = InferenceThread(
            args.port, args.baud, args.model,
            args.spacing, args.processed_dir,
            result_queue, stop_event,
        )
    thread.start()

    clock = pygame.time.Clock()
    running = True

    try:
        while running:
            running = display.handle_events()

            # Drain queue — keep only latest prediction
            latest = None
            try:
                while True:
                    item = result_queue.get_nowait()
                    if item.get("type") == "prediction":
                        latest = item
                    elif item.get("type") == "status":
                        display.set_status(item["msg"])
            except queue.Empty:
                pass

            if latest:
                display.update(latest["cell"], latest.get("confidence", 1.0))

            display.render()
            pygame.display.flip()
            clock.tick(PI_DISPLAY_FPS)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        thread.join(timeout=2.0)
        display.cleanup()


if __name__ == "__main__":
    main()
