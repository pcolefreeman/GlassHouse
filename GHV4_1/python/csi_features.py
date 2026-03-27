"""
CSI Feature Extraction — subcarrier selection and CV-normalized turbulence.

Pure functions for extracting detection-relevant features from CSI amplitude
data.  These are the foundation for presence detection: select informative
subcarriers, then compute a single turbulence scalar that quantifies how
much the wireless channel is disturbed.

Subcarrier layout (ESP32 802.11n HT20, 64-point FFT):
  - Guard bands: indices 0–10 and 53–63 (low SNR, not usable)
  - DC null:     index 32 (always zero, not usable)
  - Valid range: indices 11–51 (excluding 32)

The fixed subcarrier set [12, 14, 16, 18, 20, 24, 28, 36, 40, 44, 48, 52]
comes from ESPectre ML defaults — 12 indices spread across the valid range,
chosen for detection reliability.

Turbulence is the coefficient of variation (CV = σ/μ) of the selected
subcarrier amplitudes within a single CSI frame.  CV normalization is
required because ESP32-WROOM has no AGC gain lock — raw standard deviation
would produce false positives when the automatic gain changes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Fixed subcarrier indices selected for presence detection (ESPectre defaults).
#: Spread across the valid range (11–51 excl. DC at 32), avoiding guard bands.
SELECTED_SUBCARRIERS: list[int] = [12, 14, 16, 18, 20, 24, 28, 36, 40, 44, 48, 52]

#: Division-by-zero guard for CV computation.
_EPSILON: float = 1e-10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_subcarriers(
    amplitudes: list[float],
    indices: list[int] | None = None,
) -> NDArray[np.float64]:
    """Select a subset of subcarrier amplitudes by index.

    Args:
        amplitudes: Full list of per-subcarrier amplitudes (typically 64
            values from ``compute_amplitudes``).
        indices: Subcarrier indices to select.  Defaults to
            :data:`SELECTED_SUBCARRIERS`.  Out-of-range indices are
            silently skipped (the amplitude array may be shorter than 64
            if the firmware truncates or the frame is malformed).

    Returns:
        1-D numpy array of the selected amplitude values.  May be shorter
        than ``indices`` if some indices exceed ``len(amplitudes)``.
    """
    if indices is None:
        indices = SELECTED_SUBCARRIERS

    n = len(amplitudes)
    selected = [amplitudes[i] for i in indices if i < n]
    return np.array(selected, dtype=np.float64)


def compute_turbulence(selected_amplitudes: NDArray[np.float64]) -> float:
    """Compute CV-normalized turbulence for one CSI frame.

    Turbulence = coefficient of variation = σ / μ across the selected
    subcarrier amplitudes.  This single scalar quantifies how disturbed
    the wireless channel is in this frame.

    CV normalization makes the metric robust to ESP32 AGC gain changes
    that would cause raw standard deviation to spike without any actual
    motion.

    Args:
        selected_amplitudes: 1-D array of amplitude values from
            :func:`select_subcarriers`.

    Returns:
        Turbulence value (float).  Returns ``0.0`` if the array is empty
        or the mean is below epsilon (avoids division by zero).
    """
    if len(selected_amplitudes) == 0:
        return 0.0

    mean = float(np.mean(selected_amplitudes))
    if mean < _EPSILON:
        return 0.0

    std = float(np.std(selected_amplitudes))
    return std / mean
