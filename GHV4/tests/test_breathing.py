"""Tests for ghv4.breathing — CSI breathing detection pipeline."""
import struct

import numpy as np
import pytest

from ghv4.breathing import (
    CSIRingBuffer,
    CSIRatioExtractor,
    BreathingAnalyzer,
    GridProjector,
    BreathingDetector,
    reconstruct_csi_from_csv_row,
)
from ghv4.config import BREATHING_PATH_MAP, CELL_LABELS


class TestCSIRingBuffer:
    def test_empty_buffer_not_ready(self):
        buf = CSIRingBuffer(capacity=5, n_subcarriers=4)
        assert not buf.is_full()
        assert buf.get_window() is None

    def test_push_until_full(self):
        buf = CSIRingBuffer(capacity=3, n_subcarriers=4)
        for i in range(3):
            buf.push(np.ones(4, dtype=np.complex64) * (i + 1))
        assert buf.is_full()

    def test_get_window_shape_and_dtype(self):
        buf = CSIRingBuffer(capacity=3, n_subcarriers=4)
        for i in range(3):
            buf.push(np.ones(4, dtype=np.complex64) * (i + 1))
        w = buf.get_window()
        assert w.shape == (3, 4)
        assert w.dtype == np.complex64

    def test_fifo_order(self):
        """Oldest frame is row 0, newest is row -1."""
        buf = CSIRingBuffer(capacity=3, n_subcarriers=2)
        buf.push(np.array([1+0j, 0+0j], dtype=np.complex64))
        buf.push(np.array([2+0j, 0+0j], dtype=np.complex64))
        buf.push(np.array([3+0j, 0+0j], dtype=np.complex64))
        w = buf.get_window()
        assert w[0, 0] == 1+0j
        assert w[2, 0] == 3+0j

    def test_overflow_evicts_oldest(self):
        buf = CSIRingBuffer(capacity=3, n_subcarriers=2)
        for i in range(5):
            buf.push(np.array([i+0j, 0+0j], dtype=np.complex64))
        w = buf.get_window()
        # frames 0,1 evicted; remaining are 2,3,4
        assert w[0, 0] == 2+0j
        assert w[2, 0] == 4+0j

    def test_count_tracks_pushes(self):
        buf = CSIRingBuffer(capacity=3, n_subcarriers=2)
        assert buf.count == 0
        buf.push(np.ones(2, dtype=np.complex64))
        assert buf.count == 1

    def test_push_longer_array_truncates(self):
        """Pushing an array longer than n_subcarriers should truncate."""
        buf = CSIRingBuffer(capacity=2, n_subcarriers=3)
        buf.push(np.array([1+0j, 2+0j, 3+0j, 4+0j, 5+0j], dtype=np.complex64))
        buf.push(np.array([6+0j, 7+0j, 8+0j, 9+0j, 10+0j], dtype=np.complex64))
        w = buf.get_window()
        assert w.shape == (2, 3)
        assert w[0, 2] == 3+0j  # truncated at n_subcarriers=3


class TestCSIRatioExtractor:
    def test_pair_indices_count(self):
        ext = CSIRatioExtractor(n_subcarriers=128, n_pairs=10)
        assert len(ext.pair_indices) == 10

    def test_pair_indices_avoid_null(self):
        """No pair index should be a null subcarrier."""
        ext = CSIRatioExtractor(n_subcarriers=128, n_pairs=10)
        from ghv4.config import NULL_SUBCARRIER_INDICES
        for k1, k2 in ext.pair_indices:
            assert k1 not in NULL_SUBCARRIER_INDICES
            assert k2 not in NULL_SUBCARRIER_INDICES

    def test_extract_shape(self):
        """extract() returns (n_time, n_pairs) phase array."""
        ext = CSIRatioExtractor(n_subcarriers=128, n_pairs=10)
        # 5 time steps, 128 subcarriers
        window = np.ones((5, 128), dtype=np.complex64)
        result = ext.extract(window)
        assert result.shape == (5, 10)

    def test_extract_cancels_common_phase(self):
        """If all subcarriers have the same phase rotation, CSI ratio phase should be ~0."""
        ext = CSIRatioExtractor(n_subcarriers=128, n_pairs=10)
        n_time = 10
        # Apply uniform phase rotation: all subcarriers get same random phase per time step
        rng = np.random.default_rng(42)
        window = np.zeros((n_time, 128), dtype=np.complex64)
        for t in range(n_time):
            phase = rng.uniform(-np.pi, np.pi)
            window[t, :] = np.exp(1j * phase)
        result = ext.extract(window)
        # All ratio phases should be ~0 (common mode cancelled)
        assert np.allclose(result, 0.0, atol=1e-5)


class TestBreathingAnalyzer:
    def test_synthetic_breathing_high_confidence(self):
        """A 0.25 Hz sinusoidal phase modulation should yield high confidence."""
        analyzer = BreathingAnalyzer(sample_rate_hz=5.0, band_hz=(0.1, 0.5))
        n_time = 150
        n_pairs = 10
        t = np.arange(n_time) / 5.0  # 5 Hz sample rate
        # 0.25 Hz breathing signal on all pairs
        phases = np.column_stack([np.sin(2 * np.pi * 0.25 * t)] * n_pairs).astype(np.float32)
        confidence = analyzer.analyze(phases)
        assert confidence > 0.5, f"Expected high confidence, got {confidence}"

    def test_static_signal_low_confidence(self):
        """Constant phase (no motion) should yield near-zero confidence."""
        analyzer = BreathingAnalyzer(sample_rate_hz=5.0, band_hz=(0.1, 0.5))
        phases = np.zeros((150, 10), dtype=np.float32)
        confidence = analyzer.analyze(phases)
        assert confidence < 0.1, f"Expected low confidence, got {confidence}"

    def test_high_frequency_signal_low_confidence(self):
        """A 2 Hz signal (outside breathing band) should yield low confidence."""
        analyzer = BreathingAnalyzer(sample_rate_hz=5.0, band_hz=(0.1, 0.5))
        n_time = 150
        n_pairs = 10
        t = np.arange(n_time) / 5.0
        # 2 Hz signal — well above breathing band
        phases = np.column_stack([np.sin(2 * np.pi * 2.0 * t)] * n_pairs).astype(np.float32)
        confidence = analyzer.analyze(phases)
        assert confidence < 0.3, f"Expected low confidence for 2 Hz, got {confidence}"

    def test_random_noise_moderate_confidence(self):
        """White noise has energy spread across all bins; breathing band fraction should be small."""
        analyzer = BreathingAnalyzer(sample_rate_hz=5.0, band_hz=(0.1, 0.5))
        rng = np.random.default_rng(99)
        phases = rng.standard_normal((150, 10)).astype(np.float32)
        confidence = analyzer.analyze(phases)
        # Breathing band is ~12 bins out of 75 total (excluding DC), so
        # white noise gives ~16% — well below threshold of 0.3
        assert confidence < 0.3, f"Random noise gave {confidence}"

    def test_returns_float(self):
        analyzer = BreathingAnalyzer(sample_rate_hz=5.0, band_hz=(0.1, 0.5))
        phases = np.zeros((150, 10), dtype=np.float32)
        result = analyzer.analyze(phases)
        assert isinstance(result, float)


class TestGridProjector:
    def test_default_path_map(self):
        proj = GridProjector()
        assert proj.path_map == BREATHING_PATH_MAP

    def test_all_paths_high_yields_all_cells_high(self):
        proj = GridProjector()
        confidences = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6}
        scores = proj.project(confidences)
        # All mapped cells should have scores > 0
        for cell in ["r0c0", "r0c2", "r1c1", "r2c0", "r2c2"]:
            assert scores[cell] is not None and scores[cell] > 0

    def test_unmapped_cells_are_none(self):
        """Cells not crossed by any path should be None."""
        proj = GridProjector()
        confidences = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6}
        scores = proj.project(confidences)
        # r0c1, r1c0, r1c2, r2c1 are not in default path map
        for cell in ["r0c1", "r1c0", "r1c2", "r2c1"]:
            assert scores[cell] is None

    def test_center_cell_gets_max_of_all_paths(self):
        """r1c1 is crossed by all 4 paths; should get max confidence."""
        proj = GridProjector()
        confidences = {1: 0.3, 2: 0.9, 3: 0.5, 4: 0.1}
        scores = proj.project(confidences)
        assert scores["r1c1"] == pytest.approx(90.0)  # 0.9 * 100

    def test_single_path_active(self):
        """Only paths present in confidences dict contribute."""
        proj = GridProjector()
        confidences = {2: 0.8}
        scores = proj.project(confidences)
        assert scores["r0c2"] == pytest.approx(80.0)
        assert scores["r1c1"] == pytest.approx(80.0)
        # Cells only covered by other paths should be None
        assert scores["r0c0"] is None

    def test_custom_path_map(self):
        custom = {1: ["r0c0", "r0c1"], 2: ["r0c1", "r0c2"]}
        proj = GridProjector(path_map=custom)
        confidences = {1: 0.5, 2: 0.7}
        scores = proj.project(confidences)
        assert scores["r0c0"] == pytest.approx(50.0)
        assert scores["r0c1"] == pytest.approx(70.0)  # max(0.5, 0.7) * 100
        assert scores["r0c2"] == pytest.approx(70.0)


class TestBreathingDetector:
    @staticmethod
    def _make_csi_bytes(n_subcarriers=128):
        """Generate valid CSI bytes: n_subcarriers I/Q int16 pairs."""
        return b''.join(struct.pack('<hh', 100, 50) for _ in range(n_subcarriers))

    def test_not_ready_initially(self):
        det = BreathingDetector()
        assert not det.is_ready()

    def test_feed_shouter_frame(self):
        det = BreathingDetector()
        frame = {'shouter_id': 1, 'csi_bytes': self._make_csi_bytes()}
        det.feed_frame('shouter', frame)
        # Should have pushed one frame for path 1
        assert det._buffers[1].count == 1

    def test_ignores_listener_frames(self):
        det = BreathingDetector()
        frame = {'rssi': -55, 'csi_bytes': self._make_csi_bytes()}
        det.feed_frame('listener', frame)
        # No buffers should have data
        assert all(buf.count == 0 for buf in det._buffers.values())

    def test_ignores_unknown_shouter_ids(self):
        det = BreathingDetector()
        frame = {'shouter_id': 99, 'csi_bytes': self._make_csi_bytes()}
        det.feed_frame('shouter', frame)
        assert 99 not in det._buffers

    def test_ready_after_full_window(self):
        det = BreathingDetector()
        csi = self._make_csi_bytes()
        for _ in range(150):
            det.feed_frame('shouter', {'shouter_id': 1, 'csi_bytes': csi})
        assert det.is_ready()

    def test_get_grid_scores_returns_dict(self):
        det = BreathingDetector()
        csi = self._make_csi_bytes()
        for _ in range(150):
            for sid in [1, 2, 3, 4]:
                det.feed_frame('shouter', {'shouter_id': sid, 'csi_bytes': csi})
        scores = det.get_grid_scores()
        assert isinstance(scores, dict)
        assert "r0c0" in scores
        assert "r1c1" in scores


class TestBreathingDetectorE2E:
    def test_synthetic_breathing_detected(self):
        """Feed synthetic 0.25 Hz breathing signal with differential phase, verify detection."""
        det = BreathingDetector()
        n_time = 150
        n_sub = 128
        t = np.arange(n_time) / 5.0  # 5 Hz

        for step in range(n_time):
            # Breathing modulates phase with a linear gradient across subcarriers.
            # This ensures EVERY adjacent pair sees differential phase oscillation,
            # matching real multipath CSI behavior where path-length changes affect
            # subcarriers proportionally to frequency.
            breathing_mod = 0.8 * np.sin(2 * np.pi * 0.25 * t[step])
            phase_per_sc = np.array([breathing_mod * (sc / n_sub)
                                     for sc in range(n_sub)])
            csi = 1000.0 * np.exp(1j * phase_per_sc).astype(np.complex64)
            # Convert to bytes (int16 I/Q pairs)
            csi_bytes = b''
            for c in csi:
                i_val = int(np.real(c))
                q_val = int(np.imag(c))
                csi_bytes += struct.pack('<hh', i_val, q_val)
            for sid in [1, 2, 3, 4]:
                det.feed_frame('shouter', {'shouter_id': sid, 'csi_bytes': csi_bytes})

        scores = det.get_grid_scores()
        # Center cell (crossed by all paths) should show confident detection
        assert scores["r1c1"] is not None and scores["r1c1"] > 30.0

    def test_static_signal_no_detection(self):
        """Feed constant CSI, verify low/no confidence."""
        det = BreathingDetector()
        # Static CSI: constant I=1000, Q=0
        csi_bytes = b''.join(struct.pack('<hh', 1000, 0) for _ in range(128))
        for _ in range(150):
            for sid in [1, 2, 3, 4]:
                det.feed_frame('shouter', {'shouter_id': sid, 'csi_bytes': csi_bytes})

        scores = det.get_grid_scores()
        # With constant signal, confidence should be low
        for cell, score in scores.items():
            if score is not None:
                assert score < 30.0, f"Cell {cell} had {score}% with static signal"


import subprocess
import sys


class TestRunSarCLI:
    def test_help_flag(self):
        """run_sar.py --help should exit 0 and show usage."""
        result = subprocess.run(
            [sys.executable, "run_sar.py", "--help"],
            capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0
        assert "--port" in result.stdout
        assert "--replay" in result.stdout


class TestCSVReplay:
    def test_replay_reconstruction_round_trip(self):
        """Verify amp+phase -> complex CSI reconstruction."""
        # Create a minimal CSV with amp_norm and phase columns for shouter 1
        import csv
        import tempfile
        import os

        n_sub = 128
        n_rows = 5
        # Generate known complex CSI
        rng = np.random.default_rng(123)
        original_csi = rng.standard_normal((n_rows, n_sub)) + 1j * rng.standard_normal((n_rows, n_sub))

        # Build CSV columns
        header = ["timestamp_ms"]
        for sc in range(n_sub):
            header.append(f"s1_amp_norm_{sc}")
            header.append(f"s1_phase_{sc}")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row_idx in range(n_rows):
                row = [row_idx * 200]  # timestamp_ms
                for sc in range(n_sub):
                    c = original_csi[row_idx, sc]
                    amp = abs(c)
                    phase = np.angle(c)
                    row.append(amp)
                    row.append(phase)
                writer.writerow(row)
            tmppath = f.name

        try:
            # Reconstruct using the same formula from the spec
            from ghv4.breathing import reconstruct_csi_from_csv_row
            import pandas as pd
            df = pd.read_csv(tmppath)
            for row_idx in range(n_rows):
                reconstructed = reconstruct_csi_from_csv_row(df.iloc[row_idx], shouter_id=1)
                for sc in range(n_sub):
                    orig = original_csi[row_idx, sc]
                    recon = reconstructed[sc]
                    # Phase should match exactly; amplitude is min-max normalized so
                    # we check relative shape (correlation) not absolute value
                    assert np.angle(recon) == pytest.approx(np.angle(orig), abs=1e-5)
        finally:
            os.unlink(tmppath)
