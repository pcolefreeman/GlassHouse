"""Snap-frame CSI → ML feature vector for distance estimation.

Parses [0xEE][0xFF] snap CSI bytes (int8 I/Q) into a 484-element feature
vector (121 amp_norm + 121 phase per direction, forward + reverse).

Used by: distance_preprocess.py, distance_train.py, distance_inference.py
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from ghv4.config import (
    NULL_SUBCARRIER_INDICES,
    SUBCARRIERS,
    VALID_SUBCARRIER_COUNT,
    DISTANCE_FEATURE_COUNT,
)

# Ordered indices of valid (non-null) subcarriers
_VALID_INDICES = sorted(set(range(SUBCARRIERS)) - NULL_SUBCARRIER_INDICES)

# Pre-built feature name list (484 entries)
_directions = ("fwd", "rev")
_feat_types = ("amp_norm", "phase")
FEATURE_NAMES: List[str] = []
for d in _directions:
    for ft in _feat_types:
        for idx in _VALID_INDICES:
            FEATURE_NAMES.append(f"{d}_{ft}_{idx}")

assert len(FEATURE_NAMES) == DISTANCE_FEATURE_COUNT


def snap_csi_to_complex(csi_bytes: bytes) -> Optional[np.ndarray]:
    """Parse raw snap CSI bytes into a 121-element complex array.

    ESP32 format: int8 pairs (imag, real) per subcarrier, 128 subcarriers.
    Null subcarriers are dropped.

    Returns None if buffer is too short.
    """
    if len(csi_bytes) < SUBCARRIERS * 2:
        return None

    raw = np.frombuffer(csi_bytes[: SUBCARRIERS * 2], dtype=np.int8)
    imag = raw[0::2].astype(np.float64)
    real = raw[1::2].astype(np.float64)
    full = real + 1j * imag  # (128,)
    return full[_VALID_INDICES]  # (121,)


def extract_snap_features(csi_complex: np.ndarray) -> List[float]:
    """Extract 242 features from one direction's 121-element complex CSI.

    Returns [amp_norm(121), phase(121)] where:
    - amp_norm: magnitudes normalized to [0, 1] (per-frame min-max)
    - phase: atan2 phase scaled by pi to [-1, 1]
    """
    amp = np.abs(csi_complex)
    amp_max = amp.max()
    if amp_max > 0:
        amp_norm = amp / amp_max
    else:
        amp_norm = np.zeros_like(amp)

    phase = np.angle(csi_complex) / math.pi  # [-1, 1]

    return amp_norm.tolist() + phase.tolist()


def pair_features(
    fwd_complex: np.ndarray, rev_complex: np.ndarray
) -> List[float]:
    """Combine forward + reverse snap CSI into a 484-element feature vector.

    Args:
        fwd_complex: 121-element complex array (reporter→peer direction)
        rev_complex: 121-element complex array (peer→reporter direction)

    Returns:
        484-element list: [fwd_amp_norm(121), fwd_phase(121),
                           rev_amp_norm(121), rev_phase(121)]
    """
    fwd_feats = extract_snap_features(fwd_complex)
    rev_feats = extract_snap_features(rev_complex)
    return fwd_feats + rev_feats
