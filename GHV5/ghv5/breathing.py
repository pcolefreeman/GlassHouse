"""breathing.py — CSI breathing/micro-motion detection pipeline.

Zero-calibration human presence detection using WiFi CSI signals.
Uses CSI Ratio (conjugate multiply between subcarrier pairs) to cancel
CFO/clock drift, then FFT to detect breathing-band (0.1-0.5 Hz) power.
"""
import logging

import numpy as np

from ghv5.config import (
    SUBCARRIERS,
    NULL_SUBCARRIER_INDICES,
    BREATHING_WINDOW_N,
    BREATHING_BAND_HZ,
    BREATHING_NPAIRS,
    BREATHING_CONFIDENCE_THRESHOLD,
    BREATHING_PATH_MAP,
    BREATHING_SNAP_HZ,
    BREATHING_PCA_COMPONENTS,
    CELL_LABELS,
    PRESENCE_VARIANCE_MIDPOINT,
    PRESENCE_VARIANCE_STEEPNESS,
)
from ghv5.csi_parser import parse_csi_bytes

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


class GridProjector:
    """Project per-path breathing confidence onto a 3x3 grid.

    Uses a static path-to-cell mapping. Each cell's score is the max confidence
    of all paths crossing it. Cells not covered by any active path report None.
    Scores are normalized to 0-100%.
    """

    def __init__(self, path_map: dict | None = None):
        self.path_map = path_map if path_map is not None else BREATHING_PATH_MAP

    def project(self, path_confidences: dict[tuple[int, int], float]) -> dict[str, float | None]:
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
        """Run presence analysis on all ready paths and project onto grid.

        Uses a two-pass approach: Pass 1 collects per-path mean amplitudes
        for cross-path ranking context; Pass 2 scores each path.
        """
        # Pass 1: collect mean amplitudes for cross-path ranking
        all_path_means: dict[tuple, float] = {}
        ready_windows: dict[tuple, np.ndarray] = {}
        for key, buf in self._buffers.items():
            if not buf.is_full():
                continue
            window = buf.get_window()
            ready_windows[key] = window
            valid = sorted(set(range(window.shape[1])) - set(NULL_SUBCARRIER_INDICES))
            amp = np.abs(window[:, valid]).astype(np.float64)
            all_path_means[key] = float(np.median(np.mean(amp, axis=0)))

        # Pass 2: score each path
        presence_confidences: dict[tuple, float] = {}
        for key, window in ready_windows.items():
            confidence = self._presence_score(window, all_path_means)
            _log.info("Path S%d↔S%d presence=%.3f", key[0], key[1], confidence)
            presence_confidences[key] = confidence

        return self._projector.project(presence_confidences)

    @staticmethod
    def _pca_score(window: np.ndarray, k: int = BREATHING_PCA_COMPONENTS) -> float:
        """Per-subcarrier amplitude PCA + FFT, scored as breathing-band SNR."""
        from sklearn.decomposition import PCA

        n_time, n_subs = window.shape
        freq_res = BREATHING_SNAP_HZ / n_time
        bin_lo = max(1, int(np.ceil(BREATHING_BAND_HZ[0] / freq_res)))
        bin_hi = min(n_time // 2, int(np.floor(BREATHING_BAND_HZ[1] / freq_res)))
        n_band = bin_hi - bin_lo + 1
        ref_lo = bin_hi + 1
        ref_hi = min(n_time // 2, ref_lo + n_band - 1)

        valid = sorted(set(range(n_subs)) - set(NULL_SUBCARRIER_INDICES))
        k = min(k, len(valid))
        if k == 0:
            return 0.0

        amp = np.abs(window[:, valid]).astype(np.float64)

        # Vectorised linear detrend
        x = np.arange(n_time, dtype=np.float64)
        xm = x - x.mean()
        xvar = float(np.dot(xm, xm))
        if xvar == 0:
            return 0.0
        slopes = (xm @ amp) / xvar
        intercepts = amp.mean(axis=0) - slopes * x.mean()
        amp -= x[:, None] * slopes + intercepts

        # Hanning taper
        amp *= np.hanning(n_time)[:, None]

        # PCA: project to k principal components
        components = PCA(n_components=k).fit_transform(amp)  # (n_time, k)

        # FFT power spectra
        power = np.abs(np.fft.rfft(components, axis=0)) ** 2  # (n_freq, k)

        # Per-component breathing-band SNR
        snr_vals = []
        for j in range(k):
            ref_j = float(np.mean(power[ref_lo:ref_hi + 1, j]))
            if ref_j <= 1e-12:
                continue
            breath_j = float(np.mean(power[bin_lo:bin_hi + 1, j]))
            snr_vals.append(breath_j / ref_j)

        if not snr_vals:
            return 0.0

        snr_max = float(max(snr_vals))
        _log.debug("  pca_snr_max=%.3f", snr_max)

        log_snr = np.log(max(snr_max, 1e-6))
        score = 1.0 / (1.0 + np.exp(-3.0 * (log_snr - np.log(3.0))))
        return float(score)

    @staticmethod
    def _presence_score(window: np.ndarray,
                        all_path_means: dict | None = None) -> float:
        """Zero-calibration presence score: cross-path ranking + amplitude variance.

        Signal 1 — cross-path ranking (requires 3+ paths):
          Compares this path's mean amplitude against the group median.
          A body attenuates specific paths; those paths score higher.
          rank_score = (group_median - this_mean) / group_median, clamped [0, 1].

        Signal 2 — per-path amplitude variance (always available):
          A person (even stationary) causes involuntary motion that increases
          amplitude variance vs an empty, static environment.
          Mapped through a log-sigmoid; midpoint tuned via PRESENCE_VARIANCE_MIDPOINT.

        Final score: max(rank_score, variance_score).

        Args:
            window: (n_time, n_subcarriers) complex64 CSI array.
            all_path_means: {(s1, s2): mean_amp} for all ready paths (for ranking).
                            If None or fewer than 3 entries, rank signal is skipped.

        Returns:
            Presence confidence 0.0–1.0.
        """
        n_time, n_subs = window.shape
        valid = sorted(set(range(n_subs)) - set(NULL_SUBCARRIER_INDICES))
        amp = np.abs(window[:, valid]).astype(np.float64)  # (n_time, n_valid)

        # ── Signal 1: cross-path amplitude ranking ──────────────────────────
        rank_score = 0.0
        if all_path_means is not None and len(all_path_means) >= 3:
            this_mean = float(np.median(np.mean(amp, axis=0)))
            group_median = float(np.median(list(all_path_means.values())))
            if group_median > 1e-9:
                rank_score = float(np.clip(
                    (group_median - this_mean) / group_median, 0.0, 1.0
                ))
            _log.debug("  path_mean=%.3f group_median=%.3f rank_score=%.3f",
                       this_mean, group_median, rank_score)

        # ── Signal 2: per-path amplitude variance ───────────────────────────
        var_per_sub = np.var(amp, axis=0)                   # variance over time
        path_var = float(np.percentile(var_per_sub, 75))    # 75th pct across subcarriers
        _log.debug("  path_var=%.6f", path_var)

        log_var = np.log(max(path_var, 1e-12))
        log_mid = np.log(max(PRESENCE_VARIANCE_MIDPOINT, 1e-12))
        variance_score = float(1.0 / (1.0 + np.exp(
            -PRESENCE_VARIANCE_STEEPNESS * (log_var - log_mid)
        )))
        _log.debug("  variance_score=%.3f", variance_score)

        presence = float(max(rank_score, variance_score))
        _log.debug("  presence=%.3f", presence)
        return presence

    def get_all_scores(self, k: int = BREATHING_PCA_COMPONENTS) -> dict:
        """Run presence and PCA scoring on all ready path buffers.

        Returns:
            {
              "presence":  {cell: score_0_to_100_or_None},
              "pca":       {cell: score_0_to_100_or_None},
              "path_conf": {(s1, s2): presence_confidence_0_to_1},
            }
        """
        # Pass 1: collect mean amplitudes for cross-path ranking
        all_path_means: dict[tuple, float] = {}
        ready_windows: dict[tuple, np.ndarray] = {}
        for key, buf in self._buffers.items():
            if not buf.is_full():
                continue
            window = buf.get_window()
            ready_windows[key] = window
            valid = sorted(set(range(window.shape[1])) - set(NULL_SUBCARRIER_INDICES))
            amp = np.abs(window[:, valid]).astype(np.float64)
            all_path_means[key] = float(np.median(np.mean(amp, axis=0)))

        # Pass 2: score each path
        presence_confidences: dict[tuple, float] = {}
        pca_confidences: dict[tuple, float] = {}
        for key, window in ready_windows.items():
            p_conf = self._presence_score(window, all_path_means)
            pca_conf = self._pca_score(window, k=k)
            presence_confidences[key] = p_conf
            pca_confidences[key] = pca_conf
            _log.info("Path S%d↔S%d presence=%.3f pca=%.3f",
                      key[0], key[1], p_conf, pca_conf)

        return {
            "presence":  self._projector.project(presence_confidences),
            "pca":       self._projector.project(pca_confidences),
            "path_conf": presence_confidences,
        }


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
        from ghv5.serial_io import SerialReader
        from ghv5.config import BREATHING_SLIDE_N

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
                    all_scores = self._detector.get_all_scores()
                    self._q.put({
                        "type":          "scores",
                        "presence_grid": all_scores["presence"],
                        "pca_grid":      all_scores["pca"],
                        "path_conf":     all_scores["path_conf"],
                    })
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
            self._q.put({
                "type":          "scores",
                "presence_grid": grid,
                "pca_grid":      grid,   # demo mode: duplicate presence projection
                "path_conf":     path_conf,
            })
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
            from ghv5.config import PI_SCREEN_SIZE
            self._screen_size = screen_size or PI_SCREEN_SIZE
            self._fullscreen = fullscreen
            self._presence_grid = {cell: None for cell in CELL_LABELS}
            self._pca_grid = {cell: None for cell in CELL_LABELS}
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
            _pygame.display.set_caption("GlassHouse V5 — SAR Breathing Detection")
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
            from ghv5.config import PI_CELL_BORDER
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
            from ghv5.config import PI_CELL_INACTIVE, PI_CELL_ACTIVE
            t = max(0.0, min(1.0, score / 100.0))
            return tuple(int(lo + t * (hi - lo))
                         for lo, hi in zip(PI_CELL_INACTIVE, PI_CELL_ACTIVE))

        def update(self, presence_grid, pca_grid, path_conf):
            self._presence_grid = presence_grid
            self._pca_grid = pca_grid
            self._path_conf = path_conf

        def update_fill(self, fill):
            """Update per-path buffer fill fractions {(s1,s2): 0.0-1.0}."""
            self._path_fill = fill

        def set_status(self, msg):
            self._status_msg = msg

        def render(self):
            from ghv5.config import (
                PI_DISPLAY_BG, PI_CELL_BORDER, PI_CELL_INACTIVE,
                PI_TEXT_ACTIVE, PI_TEXT_INACTIVE,
            )
            self._screen.fill(PI_DISPLAY_BG)

            # Title
            w = self._screen_size[0]
            title = self._font_title.render(
                "GlassHouse V5 — SAR Breathing Detection", True, PI_TEXT_ACTIVE)
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
                amp_score = self._presence_grid.get(label)
                pca_score = self._pca_grid.get(label)
                cfill = cell_fill.get(label, 0.0)

                if amp_score is not None:
                    bg = self._cell_color(amp_score)
                    text_color = PI_TEXT_ACTIVE
                    amp_text = f"Pr:{amp_score:.0f}%"
                elif cfill > 0:
                    bg = PI_CELL_INACTIVE
                    text_color = PI_TEXT_INACTIVE
                    amp_text = f"fill {cfill*100:.0f}%"
                else:
                    bg = PI_CELL_INACTIVE
                    text_color = PI_TEXT_INACTIVE
                    amp_text = "Pr:--"

                pca_text = f"P:{pca_score:.0f}%" if pca_score is not None else "P:--"

                _pygame.draw.rect(self._screen, bg, rect, border_radius=6)
                _pygame.draw.rect(self._screen, PI_CELL_BORDER, rect, width=2, border_radius=6)
                lbl = self._font_cell.render(label, True, text_color)
                self._screen.blit(lbl, lbl.get_rect(center=(rect.centerx, rect.centery - 18)))
                amp_surf = self._font_conf.render(amp_text, True, text_color)
                self._screen.blit(amp_surf, amp_surf.get_rect(center=(rect.centerx, rect.centery + 2)))
                pca_surf = self._font_conf.render(pca_text, True, PI_TEXT_INACTIVE)
                self._screen.blit(pca_surf, pca_surf.get_rect(center=(rect.centerx, rect.centery + 20)))

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
