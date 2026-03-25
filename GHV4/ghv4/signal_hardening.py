"""signal_hardening.py — CSI signal cleaning filters for SAR vital sign detection.

Three filters that clean CSI before analysis:
1. Hampel filter — reject temporal outliers per subcarrier
2. Coherence gate — reject frames with unstable phase
3. Subcarrier selection — pick most informative subcarriers by variance

Algorithms inspired by RuView (ruvnet/RuView, MIT license).
No code copied; reimplemented in Python for GHV4.
"""
import logging

import numpy as np

from ghv4.config import (
    HAMPEL_WINDOW,
    HAMPEL_THRESHOLD,
    COHERENCE_THRESHOLD,
    SUBCARRIER_TOP_K,
    SUBCARRIER_MIN_K,
)

_log = logging.getLogger(__name__)

# MAD-to-std-dev conversion factor (1 / Phi^{-1}(3/4))
_MAD_SCALE = 1.4826


def hampel_filter(csi_amplitudes: np.ndarray,
                  window: int = HAMPEL_WINDOW,
                  threshold: float = HAMPEL_THRESHOLD) -> np.ndarray:
    """Filter outliers per subcarrier across the time axis.

    Args:
        csi_amplitudes: (n_frames, n_subcarriers) real-valued array.
        window: Sliding window size (odd recommended).
        threshold: MAD multiplier for outlier rejection.

    Returns:
        Filtered array with same shape. Outliers replaced by local median.

    Raises:
        ValueError: If input is not a 2-D array or contains non-finite values.
    """
    if csi_amplitudes.ndim != 2:
        raise ValueError(
            f"hampel_filter expects 2-D array, got shape {csi_amplitudes.shape}")
    if csi_amplitudes.size == 0:
        return csi_amplitudes.copy()

    # Replace NaN/inf with 0 and warn
    if not np.all(np.isfinite(csi_amplitudes)):
        _log.warning("hampel_filter: input contains NaN/inf — replacing with 0")
        csi_amplitudes = np.where(
            np.isfinite(csi_amplitudes), csi_amplitudes, 0.0)

    result = csi_amplitudes.copy()
    n_frames, n_subs = result.shape

    # Single frame: nothing to filter temporally
    if n_frames < 2:
        return result

    half = window // 2

    for sc in range(n_subs):
        orig_col = csi_amplitudes[:, sc]
        for i in range(n_frames):
            lo = max(0, i - half)
            hi = min(n_frames, i + half + 1)
            segment = orig_col[lo:hi]
            med = np.median(segment)
            mad = np.median(np.abs(segment - med))
            if mad < 1e-12:
                # If MAD is ~0, check absolute deviation from median instead
                if abs(orig_col[i] - med) > 1e-12:
                    result[i, sc] = med
                continue
            if abs(orig_col[i] - med) > threshold * _MAD_SCALE * mad:
                result[i, sc] = med
    return result


def coherence_score(csi_complex: np.ndarray) -> float:
    """Return coherence score 0.0 (noise) to 1.0 (clean).

    Computes circular variance of phase differences between adjacent
    subcarriers. Low variance = coherent (good), high variance = noise.

    Args:
        csi_complex: (n_subcarriers,) complex array for one frame.

    Returns:
        Score in [0, 1]. Higher = more coherent.

    Raises:
        ValueError: If input is not a 1-D array.
    """
    if csi_complex.ndim != 1:
        raise ValueError(
            f"coherence_score expects 1-D array, got shape {csi_complex.shape}")
    if len(csi_complex) < 2:
        # Need at least 2 subcarriers for a phase difference
        return 0.0

    # Replace non-finite values with 0 before computing phases
    if not np.all(np.isfinite(csi_complex)):
        _log.warning("coherence_score: input contains NaN/inf — replacing with 0")
        csi_complex = np.where(np.isfinite(csi_complex), csi_complex, 0.0 + 0j)

    # All-zero vector has no meaningful phase
    if np.all(csi_complex == 0):
        return 0.0

    phases = np.angle(csi_complex)
    diffs = np.diff(phases)
    # Circular mean resultant length (1 = all diffs identical, 0 = random)
    mean_resultant = np.abs(np.mean(np.exp(1j * diffs)))
    return float(mean_resultant)


def gate_frame(csi_complex: np.ndarray,
               threshold: float = COHERENCE_THRESHOLD) -> bool:
    """Return True if frame should be accepted (coherence above threshold)."""
    return coherence_score(csi_complex) >= threshold


def select_subcarriers(ring_buffer: np.ndarray,
                       top_k: int = SUBCARRIER_TOP_K,
                       min_k: int = SUBCARRIER_MIN_K) -> np.ndarray:
    """Return indices of top-K subcarriers ranked by variance.

    Args:
        ring_buffer: (n_frames, n_subcarriers) amplitude array.
        top_k: Maximum subcarriers to select.
        min_k: Minimum subcarriers to return.

    Returns:
        1-D array of subcarrier indices (at least min_k, at most top_k).

    Raises:
        ValueError: If input is not a 2-D array.
    """
    if ring_buffer.ndim != 2:
        raise ValueError(
            f"select_subcarriers expects 2-D array, got shape {ring_buffer.shape}")
    if ring_buffer.size == 0:
        return np.array([], dtype=np.intp)

    n_subs = ring_buffer.shape[1]

    # Replace NaN/inf before variance computation
    if not np.all(np.isfinite(ring_buffer)):
        _log.warning("select_subcarriers: input contains NaN/inf — replacing with 0")
        ring_buffer = np.where(np.isfinite(ring_buffer), ring_buffer, 0.0)

    variances = np.var(ring_buffer, axis=0)
    ranked = np.argsort(variances)[::-1]
    k = max(min_k, min(top_k, n_subs))
    return ranked[:k]
