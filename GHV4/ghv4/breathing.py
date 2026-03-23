"""breathing.py — CSI breathing/micro-motion detection pipeline.

Zero-calibration human presence detection using WiFi CSI signals.
Uses CSI Ratio (conjugate multiply between subcarrier pairs) to cancel
CFO/clock drift, then FFT to detect breathing-band (0.1-0.5 Hz) power.
"""
import logging

import numpy as np

from ghv4.config import (
    SUBCARRIERS,
    NULL_SUBCARRIER_INDICES,
    BREATHING_WINDOW_N,
    BREATHING_BAND_HZ,
    BREATHING_NPAIRS,
    BREATHING_CONFIDENCE_THRESHOLD,
    BREATHING_PATH_MAP,
    BREATHING_SNAP_HZ,
    CELL_LABELS,
)
from ghv4.csi_parser import parse_csi_bytes

_log = logging.getLogger(__name__)


class CSIRingBuffer:
    """Per-path circular buffer of complex CSI arrays.

    Stores the most recent `capacity` frames of complex CSI data.
    Returns None from get_window() until the buffer is full.
    """

    def __init__(self, capacity: int = BREATHING_WINDOW_N,
                 n_subcarriers: int = SUBCARRIERS):
        self._capacity = capacity
        self._n_sub = n_subcarriers
        self._buf = np.zeros((capacity, n_subcarriers), dtype=np.complex64)
        self._head = 0       # next write position
        self._count = 0      # total frames pushed (capped at capacity for is_full)

    @property
    def count(self) -> int:
        return min(self._count, self._capacity)

    def is_full(self) -> bool:
        return self._count >= self._capacity

    def push(self, csi_complex: np.ndarray) -> None:
        """Add one frame of complex CSI data to the buffer."""
        self._buf[self._head] = csi_complex[:self._n_sub]
        self._head = (self._head + 1) % self._capacity
        self._count += 1

    def get_window(self) -> np.ndarray | None:
        """Return (capacity, n_subcarriers) array in FIFO order, or None if not full."""
        if not self.is_full():
            return None
        # Roll so oldest is row 0
        return np.roll(self._buf, -self._head, axis=0).copy()


class CSIRatioExtractor:
    """Select subcarrier pairs and compute CSI ratio phase.

    CSI Ratio: R(t) = H(t, k1) * conj(H(t, k2))
    The conjugate multiply cancels CFO/clock drift (common-mode between subcarriers).
    Only differential phase from physical motion (breathing) remains.
    """

    def __init__(self, n_subcarriers: int = SUBCARRIERS,
                 n_pairs: int = BREATHING_NPAIRS,
                 null_indices: frozenset = NULL_SUBCARRIER_INDICES):
        valid = sorted(set(range(n_subcarriers)) - set(null_indices))
        # Select n_pairs+1 evenly spaced subcarriers, then pair adjacent ones
        # to get exactly n_pairs pairs
        n_select = n_pairs + 1
        step = max(1, len(valid) // (n_select + 1))
        selected = [valid[step * (i + 1)] for i in range(n_select)
                    if step * (i + 1) < len(valid)]
        # Pair adjacent selected subcarriers
        self.pair_indices = [(selected[i], selected[i + 1])
                            for i in range(min(n_pairs, len(selected) - 1))]

    def extract(self, window: np.ndarray) -> np.ndarray:
        """Compute CSI ratio phase for each time step and pair.

        Args:
            window: (n_time, n_subcarriers) complex64 array.

        Returns:
            (n_time, n_pairs) float32 array of ratio phases in radians.
        """
        n_time = window.shape[0]
        n_pairs = len(self.pair_indices)
        result = np.empty((n_time, n_pairs), dtype=np.float32)
        for j, (k1, k2) in enumerate(self.pair_indices):
            ratio = window[:, k1] * np.conj(window[:, k2])
            result[:, j] = np.angle(ratio)
        return result


class BreathingAnalyzer:
    """Detrend + Hanning window + FFT + breathing band power.

    Analyzes ratio phase time series to detect breathing-band (0.1-0.5 Hz) energy.
    Returns a confidence score (0.0-1.0) representing the fraction of spectral
    energy in the breathing band vs total energy.
    """

    def __init__(self, sample_rate_hz: float = BREATHING_SNAP_HZ,
                 band_hz: tuple = BREATHING_BAND_HZ):
        self._fs = sample_rate_hz
        self._band_hz = band_hz

    def analyze(self, ratio_phases: np.ndarray) -> float:
        """Compute breathing confidence from ratio phase time series.

        Args:
            ratio_phases: (n_time, n_pairs) float array of CSI ratio phases.

        Returns:
            Confidence score 0.0-1.0 (max breathing power ratio across pairs).
        """
        n_time, n_pairs = ratio_phases.shape
        freq_resolution = self._fs / n_time
        # Bin indices for breathing band
        bin_lo = max(1, int(np.ceil(self._band_hz[0] / freq_resolution)))
        bin_hi = min(n_time // 2, int(np.floor(self._band_hz[1] / freq_resolution)))

        pair_ratios = []
        for j in range(n_pairs):
            signal = ratio_phases[:, j].astype(np.float64)
            # Detrend: subtract linear fit
            x = np.arange(n_time, dtype=np.float64)
            coeffs = np.polyfit(x, signal, 1)
            signal -= np.polyval(coeffs, x)
            # Hanning window
            signal *= np.hanning(n_time)
            # FFT
            spectrum = np.fft.rfft(signal)
            power = np.abs(spectrum) ** 2
            # Breathing band power ratio (exclude DC bin 0)
            total_power = np.sum(power[1:])
            if total_power < 1e-12:
                pair_ratios.append(0.0)
                continue
            breathing_power = np.sum(power[bin_lo:bin_hi + 1])
            pair_ratios.append(float(breathing_power / total_power))

        return float(np.median(pair_ratios))


class GridProjector:
    """Project per-path breathing confidence onto a 3x3 grid.

    Uses a static path-to-cell mapping. Each cell's score is the max confidence
    of all paths crossing it. Cells not covered by any active path report None.
    Scores are normalized to 0-100%.
    """

    def __init__(self, path_map: dict | None = None):
        self.path_map = path_map if path_map is not None else BREATHING_PATH_MAP

    def project(self, path_confidences: dict[int, float]) -> dict[str, float | None]:
        """Project path confidences onto grid cells.

        Args:
            path_confidences: {shouter_id: confidence_0_to_1} for active paths.

        Returns:
            {cell_label: score_0_to_100_or_None} for all 9 cells.
        """
        scores: dict[str, float | None] = {cell: None for cell in CELL_LABELS}
        for sid, conf in path_confidences.items():
            if sid not in self.path_map:
                continue
            for cell in self.path_map[sid]:
                current = scores[cell]
                value = conf * 100.0
                if current is None or value > current:
                    scores[cell] = value
        return scores


class BreathingDetector:
    """Orchestrator: feeds csi_snap frames into ring buffers, runs analysis pipeline.

    Usage:
        det = BreathingDetector()
        det.feed_frame('csi_snap', frame_dict)  # call for each serial frame
        if det.is_ready():
            scores = det.get_grid_scores()       # {cell_label: 0-100 or None}
    """

    def __init__(self, path_map: dict | None = None):
        self._path_map = path_map if path_map is not None else BREATHING_PATH_MAP
        self._buffers: dict[tuple, CSIRingBuffer] = {
            key: CSIRingBuffer() for key in self._path_map
        }
        self._extractor = CSIRatioExtractor()
        self._analyzer = BreathingAnalyzer()
        self._projector = GridProjector(path_map=self._path_map)

    def feed_frame(self, frame_type: str, frame_dict: dict) -> None:
        """Feed a parsed frame into the detector.

        Args:
            frame_type: 'csi_snap' (other types are ignored)
            frame_dict: parsed frame dict with 'reporter_id', 'peer_id', 'csi'
        """
        if frame_type != 'csi_snap':
            return
        reporter = frame_dict.get('reporter_id')
        peer = frame_dict.get('peer_id')
        if reporter is None or peer is None:
            return
        key = (min(reporter, peer), max(reporter, peer))
        if key not in self._buffers:
            return
        csi_raw = frame_dict.get('csi', b'')
        if not csi_raw:
            return
        csi_complex = parse_csi_bytes(csi_raw)
        csi_array = np.array(csi_complex, dtype=np.complex64)
        # Pad/truncate to SUBCARRIERS
        if len(csi_array) < SUBCARRIERS:
            csi_array = np.pad(csi_array, (0, SUBCARRIERS - len(csi_array)))
        else:
            csi_array = csi_array[:SUBCARRIERS]
        self._buffers[key].push(csi_array)

    def is_ready(self) -> bool:
        """True if at least one path has a full buffer."""
        return any(buf.is_full() for buf in self._buffers.values())

    def get_buffer_fill(self) -> dict[tuple, float]:
        """Return fill fraction (0.0-1.0) for each path buffer."""
        return {key: buf.count / BREATHING_WINDOW_N
                for key, buf in self._buffers.items()}

    def get_grid_scores(self) -> dict[str, float | None]:
        """Run analysis on all ready paths and project onto grid.

        Confidence is purely FFT breathing-band power ratio (0.1–0.5 Hz).
        No variance term — variance has no meaning without a calibrated
        reference, making it unsuitable for SAR in unknown environments.
        """
        path_confidences = {}
        for key, buf in self._buffers.items():
            if not buf.is_full():
                continue
            window = buf.get_window()
            ratio_phases = self._extractor.extract(window)
            confidence = self._analyzer.analyze(ratio_phases)
            _log.info("Path S%d↔S%d fft=%.3f", key[0], key[1], confidence)
            path_confidences[key] = confidence
        return self._projector.project(path_confidences)


def reconstruct_csi_from_csv_row(row, shouter_id: int,
                                  n_subcarriers: int = SUBCARRIERS) -> np.ndarray:
    """Reconstruct complex CSI array from a CSV row's amp_norm + phase columns.

    Args:
        row: pandas Series or dict with columns s{id}_amp_norm_{sc} and s{id}_phase_{sc}.
        shouter_id: which shouter's columns to read.
        n_subcarriers: number of subcarriers (default 128).

    Returns:
        (n_subcarriers,) complex64 array.
    """
    csi = np.zeros(n_subcarriers, dtype=np.complex64)
    prefix = f"s{shouter_id}"
    for sc in range(n_subcarriers):
        amp_col = f"{prefix}_amp_norm_{sc}"
        phase_col = f"{prefix}_phase_{sc}"
        amp = row.get(amp_col, 0.0) if hasattr(row, 'get') else row[amp_col]
        phase = row.get(phase_col, 0.0) if hasattr(row, 'get') else row[phase_col]
        if np.isnan(amp) or np.isnan(phase):
            continue
        csi[sc] = amp * np.exp(1j * phase)
    return csi


# ---------------------------------------------------------------------------
# SAR breathing threads (for run_sar.py)
# ---------------------------------------------------------------------------
import queue as _queue
import threading
import time as _time


class BreathingThread(threading.Thread):
    """Daemon thread: serial → feed_frame() → periodic get_grid_scores() → result queue."""

    def __init__(self, port, baud, detector, result_queue, stop_event):
        super().__init__(daemon=True)
        self._port = port
        self._baud = baud
        self._detector = detector
        self._q = result_queue
        self._stop = stop_event

    def run(self):
        import serial as pyserial
        from ghv4.serial_io import SerialReader
        from ghv4.config import BREATHING_SLIDE_N

        frame_queue = _queue.Queue()
        ser = pyserial.Serial(self._port, self._baud, timeout=1.0)
        reader = SerialReader(ser, frame_queue)
        reader.start()
        self._q.put({"type": "status", "msg": f"Connected: {self._port}"})

        frames_since_update = 0
        last_fill_report = _time.time()
        snap_count_by_path = {}  # track per-path frame rate
        last_rate_time = _time.time()
        try:
            while not self._stop.is_set():
                try:
                    item = frame_queue.get(timeout=0.5)
                except _queue.Empty:
                    # Even on empty, send fill status every 2s so display updates
                    now = _time.time()
                    if now - last_fill_report >= 2.0:
                        last_fill_report = now
                        fill = self._detector.get_buffer_fill()
                        self._q.put({"type": "fill", "fill": fill})
                    continue
                frame_type, frame_dict = item
                self._detector.feed_frame(frame_type, frame_dict)
                if frame_type == 'csi_snap':
                    frames_since_update += 1
                    # Track per-path counts for rate logging
                    r = frame_dict.get('reporter_id')
                    p = frame_dict.get('peer_id')
                    if r is not None and p is not None:
                        key = (min(r, p), max(r, p))
                        snap_count_by_path[key] = snap_count_by_path.get(key, 0) + 1

                # Log per-path rates every 5s
                now = _time.time()
                if now - last_rate_time >= 5.0:
                    elapsed = now - last_rate_time
                    parts = []
                    for k in sorted(snap_count_by_path):
                        rate = snap_count_by_path[k] / elapsed
                        parts.append(f"S{k[0]}↔S{k[1]}={rate:.1f}/s")
                    if parts:
                        _log.info("Snap rates: %s", " ".join(parts))
                    snap_count_by_path.clear()
                    last_rate_time = now

                # Send fill status every 2s
                if now - last_fill_report >= 2.0:
                    last_fill_report = now
                    fill = self._detector.get_buffer_fill()
                    self._q.put({"type": "fill", "fill": fill})

                if frames_since_update >= BREATHING_SLIDE_N and self._detector.is_ready():
                    frames_since_update = 0
                    scores = self._detector.get_grid_scores()
                    path_conf = {}
                    for key, buf in self._detector._buffers.items():
                        if buf.is_full():
                            window = buf.get_window()
                            ratio = self._detector._extractor.extract(window)
                            path_conf[key] = self._detector._analyzer.analyze(ratio)
                    self._q.put({"type": "scores", "grid": scores, "path_conf": path_conf})
        except Exception as e:
            self._q.put({"type": "status", "msg": f"Error: {e}"})
        finally:
            reader.stop()
            ser.close()


class SARDemoThread(threading.Thread):
    """Synthetic 0.25 Hz breathing signal cycling across paths for --demo mode."""

    def __init__(self, result_queue, stop_event):
        super().__init__(daemon=True)
        self._q = result_queue
        self._stop = stop_event

    def run(self):
        self._q.put({"type": "status", "msg": "Demo mode — synthetic breathing"})
        path_keys = list(BREATHING_PATH_MAP.keys())
        projector = GridProjector()
        step = 0
        while not self._stop.is_set():
            # Rotate which path has highest confidence
            path_conf = {}
            for i, key in enumerate(path_keys):
                # Sinusoidal confidence cycling with phase offset per path
                t = step * 0.05
                phase_offset = i * (2 * np.pi / len(path_keys))
                conf = 0.5 + 0.4 * np.sin(2 * np.pi * 0.25 * t + phase_offset)
                path_conf[key] = float(conf)
            grid = projector.project(path_conf)
            self._q.put({"type": "scores", "grid": grid, "path_conf": path_conf})
            step += 1
            # ~1 Hz update rate
            for _ in range(10):
                if self._stop.is_set():
                    return
                _time.sleep(0.1)


# ---------------------------------------------------------------------------
# Pygame heatmap display (lazy import — pygame may not be installed)
# ---------------------------------------------------------------------------
try:
    import pygame as _pygame

    class BreathingDisplay:
        """Pygame heatmap display for SAR breathing detection."""

        TITLE_H = 44
        STATUS_H = 40
        GRID_PAD = 24
        CELL_GAP = 4

        def __init__(self, screen_size=None, fullscreen=False):
            from ghv4.config import PI_SCREEN_SIZE
            self._screen_size = screen_size or PI_SCREEN_SIZE
            self._fullscreen = fullscreen
            self._grid_scores = {cell: None for cell in CELL_LABELS}
            self._path_conf = {}
            self._path_fill = {}
            self._status_msg = "Waiting..."
            self._cell_rects = {}
            self._shouter_positions = {}

            self._init_pygame()
            self._compute_layout()

        def _init_pygame(self):
            _pygame.init()
            flags = _pygame.FULLSCREEN if self._fullscreen else 0
            self._screen = _pygame.display.set_mode(self._screen_size, flags)
            _pygame.display.set_caption("GlassHouse V4 — SAR Breathing Detection")
            try:
                self._font_cell = _pygame.font.SysFont("monospace", 28, bold=True)
                self._font_conf = _pygame.font.SysFont("monospace", 20)
                self._font_title = _pygame.font.SysFont("monospace", 24, bold=True)
                self._font_status = _pygame.font.SysFont("monospace", 16)
                self._font_shouter = _pygame.font.SysFont("monospace", 14, bold=True)
            except Exception:
                self._font_cell = _pygame.font.Font(None, 32)
                self._font_conf = _pygame.font.Font(None, 24)
                self._font_title = _pygame.font.Font(None, 28)
                self._font_status = _pygame.font.Font(None, 20)
                self._font_shouter = _pygame.font.Font(None, 18)

        def _compute_layout(self):
            from ghv4.config import PI_CELL_BORDER
            w, h = self._screen_size
            grid_top = self.TITLE_H + self.GRID_PAD
            grid_bottom = h - self.STATUS_H - self.GRID_PAD
            grid_h = grid_bottom - grid_top
            grid_w = min(grid_h, w - 2 * self.GRID_PAD)
            grid_left = (w - grid_w) // 2

            cell_w = (grid_w - 2 * self.CELL_GAP) // 3
            cell_h = (grid_h - 2 * self.CELL_GAP) // 3

            for row in range(3):
                for col in range(3):
                    x = grid_left + col * (cell_w + self.CELL_GAP)
                    y = grid_top + row * (cell_h + self.CELL_GAP)
                    self._cell_rects[(row, col)] = _pygame.Rect(x, y, cell_w, cell_h)

            margin = 14
            self._shouter_positions = {
                2: (grid_left - margin, grid_top - margin),
                3: (grid_left + grid_w + margin, grid_top - margin),
                1: (grid_left - margin, grid_top + grid_h + margin),
                4: (grid_left + grid_w + margin, grid_top + grid_h + margin),
            }
            self._grid_rect = _pygame.Rect(grid_left, grid_top, grid_w, grid_h)

        @staticmethod
        def _cell_color(score):
            """Interpolate PI_CELL_INACTIVE -> PI_CELL_ACTIVE by score (0-100)."""
            from ghv4.config import PI_CELL_INACTIVE, PI_CELL_ACTIVE
            t = max(0.0, min(1.0, score / 100.0))
            return tuple(int(lo + t * (hi - lo))
                         for lo, hi in zip(PI_CELL_INACTIVE, PI_CELL_ACTIVE))

        def update(self, grid_scores, path_conf):
            self._grid_scores = grid_scores
            self._path_conf = path_conf

        def update_fill(self, fill):
            """Update per-path buffer fill fractions {(s1,s2): 0.0-1.0}."""
            self._path_fill = fill

        def set_status(self, msg):
            self._status_msg = msg

        def render(self):
            from ghv4.config import (
                PI_DISPLAY_BG, PI_CELL_BORDER, PI_CELL_INACTIVE,
                PI_TEXT_ACTIVE, PI_TEXT_INACTIVE,
            )
            self._screen.fill(PI_DISPLAY_BG)

            # Title
            w = self._screen_size[0]
            title = self._font_title.render(
                "GlassHouse V4 — SAR Breathing Detection", True, PI_TEXT_ACTIVE)
            self._screen.blit(title, title.get_rect(center=(w // 2, self.TITLE_H // 2)))
            _pygame.draw.line(self._screen, PI_CELL_BORDER,
                              (0, self.TITLE_H - 1), (w, self.TITLE_H - 1))

            # Compute per-cell max fill fraction from path_fill
            cell_fill = {}
            if self._path_fill:
                for path_key, frac in self._path_fill.items():
                    if path_key in BREATHING_PATH_MAP:
                        for cell in BREATHING_PATH_MAP[path_key]:
                            if cell not in cell_fill or frac > cell_fill[cell]:
                                cell_fill[cell] = frac

            # Grid cells
            for (row, col), rect in self._cell_rects.items():
                label = f"r{row}c{col}"
                score = self._grid_scores.get(label)
                if score is not None:
                    bg = self._cell_color(score)
                    text_color = PI_TEXT_ACTIVE
                    score_text = f"{score:.0f}%"
                else:
                    bg = PI_CELL_INACTIVE
                    text_color = PI_TEXT_INACTIVE
                    # Show fill progress instead of bare "--"
                    cfill = cell_fill.get(label, 0.0)
                    if cfill > 0:
                        score_text = f"fill {cfill*100:.0f}%"
                    else:
                        score_text = "--"
                _pygame.draw.rect(self._screen, bg, rect, border_radius=6)
                _pygame.draw.rect(self._screen, PI_CELL_BORDER, rect, width=2,
                                  border_radius=6)
                # Label
                lbl = self._font_cell.render(label, True, text_color)
                self._screen.blit(lbl, lbl.get_rect(
                    center=(rect.centerx, rect.centery - 12)))
                # Score
                sc = self._font_conf.render(score_text, True, text_color)
                self._screen.blit(sc, sc.get_rect(
                    center=(rect.centerx, rect.centery + 16)))

            # Shouter markers + path lines
            cyan = (0, 200, 200)
            for sid, (x, y) in self._shouter_positions.items():
                _pygame.draw.circle(self._screen, cyan, (x, y), 8)
                lbl = self._font_shouter.render(f"S{sid}", True, cyan)
                self._screen.blit(lbl, lbl.get_rect(center=(x, y - 16)))

            # Path lines between shouter pairs
            for key, conf in self._path_conf.items():
                s1_pos = self._shouter_positions.get(key[0])
                s2_pos = self._shouter_positions.get(key[1])
                if s1_pos and s2_pos:
                    alpha = max(0.2, min(1.0, conf))
                    color = tuple(int(c * alpha) for c in cyan)
                    _pygame.draw.line(self._screen, color, s1_pos, s2_pos, 2)

            # Status bar
            h = self._screen_size[1]
            bar_y = h - self.STATUS_H
            _pygame.draw.line(self._screen, PI_CELL_BORDER, (0, bar_y), (w, bar_y))

            parts = [self._status_msg]
            if self._path_conf:
                conf_strs = [f"S{k[0]}↔S{k[1]}={v*100:.0f}%"
                             for k, v in sorted(self._path_conf.items())]
                parts.append(" ".join(conf_strs))
            elif self._path_fill:
                # Show fill progress while buffers are filling
                fill_strs = [f"S{k[0]}↔S{k[1]}={v*100:.0f}%"
                             for k, v in sorted(self._path_fill.items()) if v > 0]
                if fill_strs:
                    parts.append("Fill: " + " ".join(fill_strs))
            detected = [f"S{k[0]}↔S{k[1]}"
                        for k, v in self._path_conf.items()
                        if v > BREATHING_CONFIDENCE_THRESHOLD]
            if detected:
                parts.append(f"DETECTED ({', '.join(detected)})")
            else:
                parts.append("No breathing detected")

            status = self._font_status.render("  |  ".join(parts), True, PI_TEXT_INACTIVE)
            self._screen.blit(status, status.get_rect(
                midleft=(12, bar_y + self.STATUS_H // 2)))

        def handle_events(self):
            for event in _pygame.event.get():
                if event.type == _pygame.QUIT:
                    return False
                if event.type == _pygame.KEYDOWN:
                    if event.key in (_pygame.K_ESCAPE, _pygame.K_q):
                        return False
            return True

        def cleanup(self):
            _pygame.quit()

except ImportError:
    BreathingDisplay = None  # pygame not installed
