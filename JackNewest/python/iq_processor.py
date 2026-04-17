"""FFT-based vitals estimation from raw I/Q CSI packets."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.signal import welch

logger = logging.getLogger(__name__)

_VALID_NODE_IDS = frozenset(range(1, 5))  # nodes 1-4 only


class IQProcessor:
    """Accumulates I/Q frames per node, estimates breathing/heart BPM via Welch PSD."""

    def __init__(self, sample_rate: float = 0.6, window_sec: float = 30.0) -> None:
        self._fs = sample_rate
        self._max_frames = int(sample_rate * window_sec)  # 18 at 0.6 Hz
        self._buffers: dict[int, list[np.ndarray]] = {}  # node_id -> list of phase arrays

    def feed(self, node_id: int, channel: int, iq_data: bytes) -> None:
        """Ingest one I/Q frame for a given node."""
        if node_id not in _VALID_NODE_IDS:
            return
        if len(iq_data) < 2:
            return

        # Parse interleaved int8 I/Q pairs
        raw = np.frombuffer(iq_data, dtype=np.uint8)
        signed = raw.astype(np.int8)
        i_vals = signed[0::2].astype(np.float32)
        q_vals = signed[1::2].astype(np.float32)

        phase = np.arctan2(q_vals, i_vals)

        if node_id not in self._buffers:
            self._buffers[node_id] = []
        self._buffers[node_id].append(phase)

        # Trim to sliding window
        if len(self._buffers[node_id]) > self._max_frames:
            self._buffers[node_id] = self._buffers[node_id][-self._max_frames:]

    def get_vitals(self) -> dict:
        """Estimate breathing and heart BPM from accumulated I/Q data."""
        zeros = {
            "breathing_bpm": 0.0,
            "breathing_confidence": 0.0,
            "heart_bpm": 0.0,
            "heart_confidence": 0.0,
        }
        try:
            return self._compute_vitals()
        except Exception:
            logger.warning("IQProcessor.get_vitals() failed; returning zeros", exc_info=True)
            return zeros

    def _compute_vitals(self) -> dict:
        min_frames = self._max_frames // 2  # need at least 30s of data

        breathing_estimates: list[tuple[float, float]] = []
        heart_estimates: list[tuple[float, float]] = []

        for node_id, frames in self._buffers.items():
            if len(frames) < min_frames:
                continue

            # Stack into matrix (N_frames, N_subcarriers)
            matrix = np.stack(frames, axis=0)
            # Unwrap along time axis (axis=0)
            unwrapped = np.unwrap(matrix, axis=0)
            # Average across subcarriers
            phase_series = unwrapped.mean(axis=1)

            br_bpm, br_snr = self._estimate_bpm(phase_series, (0.1, 0.5), self._fs)
            breathing_estimates.append((br_bpm, br_snr))

            hr_bpm, hr_snr = self._estimate_bpm(phase_series, (0.8, 2.0), self._fs)
            heart_estimates.append((hr_bpm, hr_snr))

        fused_br_bpm, fused_br_conf = self._fuse_estimates(breathing_estimates)
        fused_hr_bpm, fused_hr_conf = self._fuse_estimates(heart_estimates)

        return {
            "breathing_bpm": fused_br_bpm,
            "breathing_confidence": fused_br_conf,
            "heart_bpm": fused_hr_bpm,
            "heart_confidence": fused_hr_conf,
        }

    @staticmethod
    def _estimate_bpm(
        phase_series: np.ndarray, band_hz: tuple[float, float], fs: float
    ) -> tuple[float, float]:
        """Welch PSD peak detection within a frequency band. Returns (bpm, snr)."""
        freqs, psd = welch(phase_series, fs=fs, nperseg=min(256, len(phase_series)))
        mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
        if not np.any(mask):
            return 0.0, 0.0

        psd_band = psd[mask]
        freqs_band = freqs[mask]
        peak_idx = np.argmax(psd_band)
        peak_freq = freqs_band[peak_idx]
        peak_power = psd_band[peak_idx]
        snr = peak_power / max(float(np.median(psd_band)), 1e-10)
        return float(peak_freq * 60.0), float(snr)

    @staticmethod
    def _fuse_estimates(
        estimates: list[tuple[float, float]],
    ) -> tuple[float, float]:
        """SNR-weighted fusion across nodes. Filters low-SNR estimates."""
        good = [(bpm, snr) for bpm, snr in estimates if snr > 3.0]
        if not good:
            return 0.0, 0.0
        total_snr = sum(snr for _, snr in good)
        fused_bpm = sum(bpm * snr for bpm, snr in good) / total_snr
        confidence = min(total_snr / 40.0, 1.0)
        return fused_bpm, confidence
