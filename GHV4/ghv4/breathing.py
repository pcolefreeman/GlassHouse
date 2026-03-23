"""breathing.py — CSI breathing/micro-motion detection pipeline.

Zero-calibration human presence detection using WiFi CSI signals.
Uses CSI Ratio (conjugate multiply between subcarrier pairs) to cancel
CFO/clock drift, then FFT to detect breathing-band (0.1-0.5 Hz) power.
"""
import numpy as np

from ghv4.config import (
    SUBCARRIERS,
    NULL_SUBCARRIER_INDICES,
    BREATHING_WINDOW_N,
    BREATHING_BAND_HZ,
    BREATHING_NPAIRS,
    BREATHING_CONFIDENCE_THRESHOLD,
    BREATHING_PATH_MAP,
    BUCKET_MS,
    CELL_LABELS,
)
from ghv4.csi_parser import parse_csi_bytes


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

    def __init__(self, sample_rate_hz: float = 1000.0 / BUCKET_MS,
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
    """Orchestrator: feeds frames into ring buffers, runs analysis pipeline.

    Usage:
        det = BreathingDetector()
        det.feed_frame(frame_type, frame_dict)  # call for each serial frame
        if det.is_ready():
            scores = det.get_grid_scores()       # {cell_label: 0-100 or None}
    """

    def __init__(self, path_map: dict | None = None):
        self._path_map = path_map if path_map is not None else BREATHING_PATH_MAP
        self._buffers: dict[int, CSIRingBuffer] = {
            sid: CSIRingBuffer() for sid in self._path_map
        }
        self._extractor = CSIRatioExtractor()
        self._analyzer = BreathingAnalyzer()
        self._projector = GridProjector(path_map=self._path_map)

    def feed_frame(self, frame_type: str, frame_dict: dict) -> None:
        """Feed a parsed frame into the detector.

        Args:
            frame_type: 'listener' or 'shouter'
            frame_dict: parsed frame dict from csi_parser
        """
        if frame_type != 'shouter':
            return
        sid = frame_dict.get('shouter_id')
        if sid not in self._buffers:
            return
        csi_bytes = frame_dict.get('csi_bytes', b'')
        if not csi_bytes:
            return
        csi_complex = parse_csi_bytes(csi_bytes)
        csi_array = np.array(csi_complex, dtype=np.complex64)
        # Pad/truncate to SUBCARRIERS
        if len(csi_array) < SUBCARRIERS:
            csi_array = np.pad(csi_array, (0, SUBCARRIERS - len(csi_array)))
        else:
            csi_array = csi_array[:SUBCARRIERS]
        self._buffers[sid].push(csi_array)

    def is_ready(self) -> bool:
        """True if at least one path has a full buffer."""
        return any(buf.is_full() for buf in self._buffers.values())

    def get_grid_scores(self) -> dict[str, float | None]:
        """Run analysis on all ready paths and project onto grid.

        Returns:
            {cell_label: confidence_0_to_100_or_None} for all 9 cells.
        """
        path_confidences = {}
        for sid, buf in self._buffers.items():
            if not buf.is_full():
                continue
            window = buf.get_window()
            ratio_phases = self._extractor.extract(window)
            confidence = self._analyzer.analyze(ratio_phases)
            if confidence >= BREATHING_CONFIDENCE_THRESHOLD:
                path_confidences[sid] = confidence
            else:
                path_confidences[sid] = 0.0
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
