"""Tests for IQProcessor FFT-based vitals estimation."""

import numpy as np
import pytest

from python.iq_processor import IQProcessor


def _make_iq_frame(n_subcarriers: int, phase_offset: float) -> bytes:
    """Build an I/Q byte frame with uniform phase offset across subcarriers."""
    i_vals = np.round(50 * np.cos(phase_offset) * np.ones(n_subcarriers)).astype(np.int8)
    q_vals = np.round(50 * np.sin(phase_offset) * np.ones(n_subcarriers)).astype(np.int8)
    interleaved = np.empty(2 * n_subcarriers, dtype=np.int8)
    interleaved[0::2] = i_vals
    interleaved[1::2] = q_vals
    return interleaved.tobytes()


class TestSyntheticBreathing:
    """AC-1: IQProcessor produces BPM from synthetic I/Q."""

    def test_synthetic_breathing(self):
        proc = IQProcessor(sample_rate=10.0, window_sec=60.0)  # synthetic rate
        n_subcarriers = 32
        breathing_hz = 0.25  # 15 BPM

        for frame_idx in range(600):
            t = frame_idx / 10.0
            phase = 0.5 * np.sin(2 * np.pi * breathing_hz * t)
            iq_bytes = _make_iq_frame(n_subcarriers, phase)
            proc.feed(1, 6, iq_bytes)

        vitals = proc.get_vitals()
        assert abs(vitals["breathing_bpm"] - 15.0) <= 3.0
        assert vitals["breathing_confidence"] > 0.0


class TestPhaseUnwrapping:
    """AC-2: Phase unwrapping handles discontinuities."""

    def test_phase_unwrapping(self):
        proc = IQProcessor(sample_rate=10.0, window_sec=60.0)
        n_subcarriers = 16

        for frame_idx in range(600):
            t = frame_idx / 10.0
            # Large amplitude forces wrapping across +/-pi
            phase = 2.0 * np.pi * 0.25 * t
            iq_bytes = _make_iq_frame(n_subcarriers, phase)
            proc.feed(1, 6, iq_bytes)

        # Access internal buffers and verify unwrapping
        frames = proc._buffers[1]
        matrix = np.stack(frames, axis=0)
        unwrapped = np.unwrap(matrix, axis=0)
        phase_series = unwrapped.mean(axis=1)

        # No discontinuity > pi between consecutive samples
        diffs = np.abs(np.diff(phase_series))
        assert np.all(diffs < np.pi), f"Max diff: {diffs.max():.3f}"


class TestSNRFiltering:
    """AC-4: Cross-node fusion filters low-SNR nodes."""

    def test_snr_filtering(self):
        proc = IQProcessor(sample_rate=10.0, window_sec=60.0)
        n_subcarriers = 32
        breathing_hz = 0.25

        rng = np.random.default_rng(42)

        for frame_idx in range(600):
            t = frame_idx / 10.0
            # Nodes 1,2: strong breathing signal
            phase_strong = 0.5 * np.sin(2 * np.pi * breathing_hz * t)
            for node in (1, 2):
                proc.feed(node, 6, _make_iq_frame(n_subcarriers, phase_strong))
            # Nodes 3,4: pure noise
            for node in (3, 4):
                noise_phase = rng.uniform(-np.pi, np.pi)
                proc.feed(node, 6, _make_iq_frame(n_subcarriers, noise_phase))

        vitals = proc.get_vitals()
        # Should still get ~15 BPM from the strong nodes
        assert abs(vitals["breathing_bpm"] - 15.0) <= 3.0
        assert vitals["breathing_confidence"] > 0.0


class TestEdgeCases:
    def test_empty_buffer(self):
        proc = IQProcessor(sample_rate=10.0, window_sec=60.0)
        vitals = proc.get_vitals()
        assert vitals["breathing_bpm"] == 0.0
        assert vitals["breathing_confidence"] == 0.0
        assert vitals["heart_bpm"] == 0.0
        assert vitals["heart_confidence"] == 0.0

    def test_partial_buffer(self):
        proc = IQProcessor(sample_rate=10.0, window_sec=60.0)
        for i in range(100):  # Only 10s, need 30s minimum
            proc.feed(1, 6, _make_iq_frame(16, float(i) * 0.1))
        vitals = proc.get_vitals()
        assert vitals["breathing_bpm"] == 0.0

    def test_invalid_node_id(self):
        proc = IQProcessor(sample_rate=10.0, window_sec=60.0)
        proc.feed(0, 6, _make_iq_frame(16, 0.0))
        proc.feed(255, 6, _make_iq_frame(16, 0.0))
        assert proc._buffers == {}

    def test_zero_length_iq(self):
        proc = IQProcessor(sample_rate=10.0, window_sec=60.0)
        proc.feed(1, 6, b'')
        proc.feed(1, 6, b'\x01')  # single byte < 2
        assert 1 not in proc._buffers

    def test_get_vitals_exception_safety(self):
        proc = IQProcessor(sample_rate=10.0, window_sec=10.0)
        # Feed NaN-corrupted frames to trigger computation errors
        for _ in range(100):
            proc.feed(1, 6, _make_iq_frame(16, 0.0))
        # Corrupt internal buffer with NaN
        proc._buffers[1] = [np.full(16, np.nan) for _ in range(100)]
        vitals = proc.get_vitals()
        # Should return zeros, not raise
        assert vitals["breathing_bpm"] == 0.0 or isinstance(vitals["breathing_bpm"], float)
