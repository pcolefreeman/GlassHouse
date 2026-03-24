"""Tests for ghv4.breathing — CSI breathing detection pipeline."""
import queue
import struct
import subprocess
import sys

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

    def test_default_sample_rate_is_snap_hz(self):
        """Default sample rate should be BREATHING_SNAP_HZ (20), not BUCKET_MS-derived (5)."""
        from ghv4.config import BREATHING_SNAP_HZ
        analyzer = BreathingAnalyzer()
        assert analyzer._fs == BREATHING_SNAP_HZ

    def test_synthetic_breathing_default_rate(self):
        """0.25 Hz breathing at default 20 Hz sample rate, 600-frame window."""
        analyzer = BreathingAnalyzer()  # default: 20 Hz
        n_time = 600
        n_pairs = 10
        t = np.arange(n_time) / 20.0
        phases = np.column_stack([np.sin(2 * np.pi * 0.25 * t)] * n_pairs).astype(np.float32)
        confidence = analyzer.analyze(phases)
        assert confidence > 0.5, f"Expected high confidence, got {confidence}"


class TestGridProjector:
    def test_default_path_map(self):
        proj = GridProjector()
        assert proj.path_map == BREATHING_PATH_MAP

    def test_all_paths_high_yields_all_9_cells(self):
        """All 6 paths active should cover all 9 cells."""
        proj = GridProjector()
        confidences = {
            (1, 2): 0.9, (1, 3): 0.8, (1, 4): 0.7,
            (2, 3): 0.6, (2, 4): 0.5, (3, 4): 0.4,
        }
        scores = proj.project(confidences)
        for cell in CELL_LABELS:
            assert scores[cell] is not None and scores[cell] > 0, \
                f"Cell {cell} should be covered but got {scores[cell]}"

    def test_single_path_covers_three_cells(self):
        """Path (1,2) = left edge covers r2c0, r1c0, r0c0 only."""
        proj = GridProjector()
        confidences = {(1, 2): 0.8}
        scores = proj.project(confidences)
        assert scores["r2c0"] == pytest.approx(80.0)
        assert scores["r1c0"] == pytest.approx(80.0)
        assert scores["r0c0"] == pytest.approx(80.0)
        # Other cells should be None
        assert scores["r0c1"] is None
        assert scores["r1c1"] is None

    def test_center_cell_max_of_crossing_paths(self):
        """r1c1 is crossed by (1,3) and (2,4); should get max."""
        proj = GridProjector()
        confidences = {(1, 3): 0.3, (2, 4): 0.9}
        scores = proj.project(confidences)
        assert scores["r1c1"] == pytest.approx(90.0)

    def test_custom_path_map(self):
        custom = {(1, 2): ["r0c0", "r0c1"], (2, 3): ["r0c1", "r0c2"]}
        proj = GridProjector(path_map=custom)
        confidences = {(1, 2): 0.5, (2, 3): 0.7}
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

    def test_feed_csi_snap_frame(self):
        """csi_snap frames should route to the canonical (min,max) buffer."""
        det = BreathingDetector()
        frame = {'reporter_id': 1, 'peer_id': 2, 'csi': self._make_csi_bytes()}
        det.feed_frame('csi_snap', frame)
        assert det._buffers[(1, 2)].count == 1

    def test_canonical_key_normalization(self):
        """Feeding (reporter=2, peer=1) should route to (1,2) buffer."""
        det = BreathingDetector()
        frame = {'reporter_id': 2, 'peer_id': 1, 'csi': self._make_csi_bytes()}
        det.feed_frame('csi_snap', frame)
        assert det._buffers[(1, 2)].count == 1

    def test_ignores_shouter_frames(self):
        """Old shouter frame type should be ignored."""
        det = BreathingDetector()
        frame = {'shouter_id': 1, 'csi_bytes': self._make_csi_bytes()}
        det.feed_frame('shouter', frame)
        assert all(buf.count == 0 for buf in det._buffers.values())

    def test_ignores_listener_frames(self):
        det = BreathingDetector()
        frame = {'rssi': -55, 'csi': self._make_csi_bytes()}
        det.feed_frame('listener', frame)
        assert all(buf.count == 0 for buf in det._buffers.values())

    def test_ignores_unknown_pair(self):
        """Pair (1,5) not in path map should be silently ignored."""
        det = BreathingDetector()
        frame = {'reporter_id': 1, 'peer_id': 5, 'csi': self._make_csi_bytes()}
        det.feed_frame('csi_snap', frame)
        assert (1, 5) not in det._buffers

    def test_ignores_missing_csi_key(self):
        """Frame without 'csi' key should be silently ignored."""
        det = BreathingDetector()
        frame = {'reporter_id': 1, 'peer_id': 2}
        det.feed_frame('csi_snap', frame)
        assert det._buffers[(1, 2)].count == 0

    def test_ready_after_full_window(self):
        from ghv4.config import BREATHING_WINDOW_N
        det = BreathingDetector()
        csi = self._make_csi_bytes()
        for _ in range(BREATHING_WINDOW_N):
            det.feed_frame('csi_snap', {'reporter_id': 1, 'peer_id': 2, 'csi': csi})
        assert det.is_ready()

    def test_get_grid_scores_returns_dict(self):
        from ghv4.config import BREATHING_WINDOW_N
        det = BreathingDetector()
        csi = self._make_csi_bytes()
        for _ in range(BREATHING_WINDOW_N):
            for key in BREATHING_PATH_MAP:
                det.feed_frame('csi_snap', {
                    'reporter_id': key[0], 'peer_id': key[1], 'csi': csi
                })
        scores = det.get_grid_scores()
        assert isinstance(scores, dict)
        # All 9 cells should be present
        for cell in CELL_LABELS:
            assert cell in scores


class TestBreathingDetectorE2E:
    def test_synthetic_breathing_one_hot_path(self):
        """Feed breathing on one path, static on others — hot path cells detected."""
        from ghv4.config import BREATHING_WINDOW_N, BREATHING_SNAP_HZ
        det = BreathingDetector()
        n_time = BREATHING_WINDOW_N
        n_sub = 128
        t = np.arange(n_time) / BREATHING_SNAP_HZ

        hot_path = (1, 3)  # BL→TR diagonal: r2c0, r1c1, r0c2
        static_csi = b''.join(struct.pack('<hh', 1000, 0) for _ in range(n_sub))

        for step in range(n_time):
            amp = int(1000 + 800 * np.sin(2 * np.pi * 0.25 * t[step]))
            breath_csi = b''.join(struct.pack('<hh', amp, 0) for _ in range(n_sub))
            for key in BREATHING_PATH_MAP:
                csi = breath_csi if key == hot_path else static_csi
                det.feed_frame('csi_snap', {
                    'reporter_id': key[0], 'peer_id': key[1], 'csi': csi
                })

        scores = det.get_grid_scores()
        # Hot path covers r2c0, r1c1, r0c2
        assert scores["r1c1"] is not None and scores["r1c1"] > 20.0, \
            f"Center cell should detect breathing, got {scores['r1c1']}"

    def test_all_paths_uniform_no_detection(self):
        """All paths with identical breathing → contrast ≈ 1 → no localized detection."""
        from ghv4.config import BREATHING_WINDOW_N, BREATHING_SNAP_HZ
        det = BreathingDetector()
        n_time = BREATHING_WINDOW_N
        n_sub = 128
        t = np.arange(n_time) / BREATHING_SNAP_HZ

        for step in range(n_time):
            amp = int(1000 + 800 * np.sin(2 * np.pi * 0.25 * t[step]))
            csi_bytes = b''.join(struct.pack('<hh', amp, 0) for _ in range(n_sub))
            for key in BREATHING_PATH_MAP:
                det.feed_frame('csi_snap', {
                    'reporter_id': key[0], 'peer_id': key[1], 'csi': csi_bytes
                })

        scores = det.get_grid_scores()
        # Uniform breathing on all paths: contrast ≈ 1 for all → low/no confidence
        # (correct: indistinguishable from environmental noise affecting all paths)
        for cell in CELL_LABELS:
            if scores[cell] is not None:
                assert scores[cell] < 50.0, \
                    f"Uniform paths should not trigger high confidence: {cell}={scores[cell]}"

    def test_static_signal_no_detection(self):
        """Feed constant CSI via csi_snap, verify low/no confidence."""
        from ghv4.config import BREATHING_WINDOW_N
        det = BreathingDetector()
        csi_bytes = b''.join(struct.pack('<hh', 1000, 0) for _ in range(128))
        for _ in range(BREATHING_WINDOW_N):
            for key in BREATHING_PATH_MAP:
                det.feed_frame('csi_snap', {
                    'reporter_id': key[0], 'peer_id': key[1], 'csi': csi_bytes
                })

        scores = det.get_grid_scores()
        for cell, score in scores.items():
            if score is not None:
                assert score < 30.0, f"Cell {cell} had {score}% with static signal"

    def test_raw_amplitude_energy_coherent_beats_incoherent(self):
        """PCA snr_eig: coherent multi-subcarrier breathing has higher snr_eig than noise."""
        from ghv4.config import BREATHING_WINDOW_N, BREATHING_SNAP_HZ, SUBCARRIERS
        from ghv4.breathing import BreathingDetector

        n_time = BREATHING_WINDOW_N
        n_sub = SUBCARRIERS
        t = np.arange(n_time) / BREATHING_SNAP_HZ
        rng = np.random.default_rng(42)

        base_amp = 1000.0
        breath_amp = 800.0
        breathing_signal = breath_amp * np.sin(2 * np.pi * 0.25 * t)
        coherent_real = np.full((n_time, n_sub), base_amp, dtype=np.float32)
        coherent_real[:, :10] += breathing_signal[:, None].astype(np.float32)
        coherent_window = coherent_real.astype(np.complex64)

        noise_real = (rng.standard_normal((n_time, n_sub)) * base_amp).astype(np.float32)
        noise_window = noise_real.astype(np.complex64)

        coherent_snr = BreathingDetector._raw_amplitude_energy(coherent_window)
        noise_snr = BreathingDetector._raw_amplitude_energy(noise_window)

        assert coherent_snr > noise_snr, (
            f"Coherent snr_eig {coherent_snr:.3f} should exceed noise {noise_snr:.3f}"
        )
        assert coherent_snr > 1.0, "Coherent 0.25 Hz signal should have snr_eig > 1"

    def test_phase_score_detects_breathing(self):
        """Phase-based CSI ratio (Approach C) detects 0.25 Hz phase modulation."""
        from ghv4.config import BREATHING_WINDOW_N, BREATHING_SNAP_HZ, SUBCARRIERS

        det = BreathingDetector()
        n_time = BREATHING_WINDOW_N
        n_sub = SUBCARRIERS
        t = np.arange(n_time) / BREATHING_SNAP_HZ

        # Phase-modulated CSI: different subcarriers get differential phase shift
        # simulating path-length change from breathing
        window = np.ones((n_time, n_sub), dtype=np.complex64)
        for sc in range(n_sub):
            # Differential phase modulation: amplitude scales with subcarrier index
            phase_mod = 0.3 * np.sin(2 * np.pi * 0.25 * t) * (sc / n_sub)
            window[:, sc] = np.exp(1j * phase_mod)

        score = det._phase_score(window)
        assert score > 0.3, f"Phase score {score:.3f} should detect 0.25 Hz phase modulation"

    def test_phase_score_low_for_static(self):
        """Static CSI (no phase modulation) yields low phase score."""
        from ghv4.config import BREATHING_WINDOW_N, SUBCARRIERS

        det = BreathingDetector()
        window = np.ones((BREATHING_WINDOW_N, SUBCARRIERS), dtype=np.complex64)
        score = det._phase_score(window)
        assert score < 0.1, f"Static signal phase score {score:.3f} should be near zero"

    def test_contrast_normalization_isolates_hot_path(self):
        """Inter-path contrast (Approach A): one elevated path detected, others suppressed."""
        from ghv4.config import BREATHING_WINDOW_N, BREATHING_SNAP_HZ, SUBCARRIERS

        det = BreathingDetector()
        n_time = BREATHING_WINDOW_N
        n_sub = SUBCARRIERS
        t = np.arange(n_time) / BREATHING_SNAP_HZ

        # Fill all 6 paths: 5 with static signal, 1 with strong breathing
        static_csi = b''.join(struct.pack('<hh', 1000, 0) for _ in range(n_sub))
        breath_csi_frames = []
        for step in range(n_time):
            amp = int(1000 + 800 * np.sin(2 * np.pi * 0.25 * t[step]))
            breath_csi_frames.append(
                b''.join(struct.pack('<hh', amp, 0) for _ in range(n_sub))
            )

        path_keys = list(BREATHING_PATH_MAP.keys())
        hot_path = path_keys[0]  # (1, 2)

        for step in range(n_time):
            for key in path_keys:
                if key == hot_path:
                    csi = breath_csi_frames[step]
                else:
                    csi = static_csi
                det.feed_frame('csi_snap', {
                    'reporter_id': key[0], 'peer_id': key[1], 'csi': csi
                })

        scores = det.get_grid_scores()
        # Hot path (1,2) covers r2c0, r1c0, r0c0 — these should be elevated
        hot_cells = set(BREATHING_PATH_MAP[hot_path])
        cold_cells = set(CELL_LABELS) - hot_cells

        for cell in hot_cells:
            assert scores[cell] is not None and scores[cell] > 20.0, \
                f"Hot cell {cell} should be elevated, got {scores[cell]}"

        for cell in cold_cells:
            # Cold cells may still have some score from overlapping paths
            # but should be much lower than hot cells
            if scores[cell] is not None:
                hot_min = min(scores[c] for c in hot_cells if scores[c] is not None)
                assert scores[cell] < hot_min, \
                    f"Cold cell {cell}={scores[cell]} should be below hot min {hot_min}"



class TestGetBufferFill:
    """Tests for BreathingDetector.get_buffer_fill()."""

    def test_empty_buffers_all_zero(self):
        """All paths should report 0.0 fill when no frames have been fed."""
        det = BreathingDetector()
        fill = det.get_buffer_fill()
        assert isinstance(fill, dict)
        for key, frac in fill.items():
            assert frac == 0.0, f"Path {key} should be 0.0, got {frac}"

    def test_partial_fill_fraction(self):
        """After feeding some frames, fill fraction should be between 0 and 1."""
        from ghv4.config import BREATHING_WINDOW_N
        det = BreathingDetector()
        csi = struct.pack('<hh', 100, 50) * 128
        n_feed = BREATHING_WINDOW_N // 4
        for _ in range(n_feed):
            det.feed_frame('csi_snap', {'reporter_id': 1, 'peer_id': 2, 'csi': csi})
        fill = det.get_buffer_fill()
        expected = n_feed / BREATHING_WINDOW_N
        assert fill[(1, 2)] == pytest.approx(expected, abs=0.01)

    def test_full_buffer_reports_one(self):
        """A full buffer should report fill fraction 1.0."""
        from ghv4.config import BREATHING_WINDOW_N
        det = BreathingDetector()
        csi = struct.pack('<hh', 100, 50) * 128
        for _ in range(BREATHING_WINDOW_N):
            det.feed_frame('csi_snap', {'reporter_id': 1, 'peer_id': 2, 'csi': csi})
        fill = det.get_buffer_fill()
        assert fill[(1, 2)] == pytest.approx(1.0)


class TestFewPathsFallback:
    """Tests for A+C scoring when fewer than BREATHING_MIN_PATHS_FOR_CONTRAST paths are ready."""

    def test_two_paths_uses_phase_only(self):
        """With only 2 ready paths (< MIN_PATHS_FOR_CONTRAST=3), contrast_score should be 0."""
        from ghv4.config import BREATHING_WINDOW_N, BREATHING_SNAP_HZ, BREATHING_PATH_MAP
        det = BreathingDetector()
        n_time = BREATHING_WINDOW_N
        n_sub = 128
        t = np.arange(n_time) / BREATHING_SNAP_HZ

        # Feed breathing signal on exactly 2 paths only
        path_keys = list(BREATHING_PATH_MAP.keys())[:2]
        for step in range(n_time):
            amp = int(1000 + 800 * np.sin(2 * np.pi * 0.25 * t[step]))
            csi = b''.join(struct.pack('<hh', amp, 0) for _ in range(n_sub))
            for key in path_keys:
                det.feed_frame('csi_snap', {
                    'reporter_id': key[0], 'peer_id': key[1], 'csi': csi
                })

        scores = det.get_grid_scores()
        # Should still produce scores (via phase path), not crash
        assert isinstance(scores, dict)
        assert len(scores) == 9

    def test_single_path_still_produces_scores(self):
        """Even with 1 ready path, get_grid_scores should return valid dict."""
        from ghv4.config import BREATHING_WINDOW_N, BREATHING_SNAP_HZ
        det = BreathingDetector()
        n_time = BREATHING_WINDOW_N
        t = np.arange(n_time) / BREATHING_SNAP_HZ
        for step in range(n_time):
            amp = int(1000 + 800 * np.sin(2 * np.pi * 0.25 * t[step]))
            csi = b''.join(struct.pack('<hh', amp, 0) for _ in range(128))
            det.feed_frame('csi_snap', {
                'reporter_id': 1, 'peer_id': 2, 'csi': csi
            })
        scores = det.get_grid_scores()
        assert isinstance(scores, dict)
        # Path (1,2) covers r0c0, r1c0, r2c0
        for cell in BREATHING_PATH_MAP[(1, 2)]:
            assert scores[cell] is not None


class TestRawAmplitudeEnergyEdgeCases:
    """Edge case tests for BreathingDetector._raw_amplitude_energy()."""

    def test_all_zero_window_returns_zero(self):
        """All-zero CSI window should return 0.0 (denominator guard)."""
        from ghv4.config import BREATHING_WINDOW_N, SUBCARRIERS
        window = np.zeros((BREATHING_WINDOW_N, SUBCARRIERS), dtype=np.complex64)
        result = BreathingDetector._raw_amplitude_energy(window)
        assert result == 0.0

    def test_dc_only_signal_low_snr(self):
        """Constant amplitude (DC only) should yield low snr_eig after detrend."""
        from ghv4.config import BREATHING_WINDOW_N, SUBCARRIERS
        window = np.full((BREATHING_WINDOW_N, SUBCARRIERS), 500 + 0j, dtype=np.complex64)
        result = BreathingDetector._raw_amplitude_energy(window)
        # After detrend, a constant signal becomes ~zero; snr_eig should be near 0
        assert result < 1.0, f"DC-only signal should have low snr_eig, got {result}"


class TestCombinedACScoring:
    """Tests for the max(contrast_score, phase_score) combination logic."""

    def test_phase_triggers_when_contrast_low(self):
        """Phase-modulated signal on a path with all paths at similar amplitude
        should still trigger detection via phase score even if contrast is ~1."""
        from ghv4.config import BREATHING_WINDOW_N, BREATHING_SNAP_HZ
        det = BreathingDetector()
        n_time = BREATHING_WINDOW_N
        n_sub = 128
        t = np.arange(n_time) / BREATHING_SNAP_HZ

        # All paths get same amplitude but one path gets phase modulation
        hot_path = (1, 3)
        for step in range(n_time):
            for key in BREATHING_PATH_MAP:
                if key == hot_path:
                    # Phase-modulated: differential phase shift between subcarriers
                    pairs = []
                    for sc in range(n_sub):
                        phase = 0.5 * np.sin(2 * np.pi * 0.25 * t[step]) * (sc / n_sub)
                        c = 1000 * np.exp(1j * phase)
                        pairs.append(struct.pack('<hh', int(c.real), int(c.imag)))
                    csi = b''.join(pairs)
                else:
                    # Static: uniform amplitude, no phase modulation
                    csi = b''.join(struct.pack('<hh', 1000, 0) for _ in range(n_sub))
                det.feed_frame('csi_snap', {
                    'reporter_id': key[0], 'peer_id': key[1], 'csi': csi
                })

        scores = det.get_grid_scores()
        # Hot path cells should have some detection via phase score
        hot_cells = set(BREATHING_PATH_MAP[hot_path])
        for cell in hot_cells:
            assert scores[cell] is not None, f"Hot cell {cell} should have a score"

    def test_no_ready_paths_returns_all_none(self):
        """If no buffers are full, all cells should be None."""
        det = BreathingDetector()
        # Feed just a few frames — not enough to fill any buffer
        csi = struct.pack('<hh', 100, 50) * 128
        det.feed_frame('csi_snap', {'reporter_id': 1, 'peer_id': 2, 'csi': csi})
        scores = det.get_grid_scores()
        for cell in CELL_LABELS:
            assert scores[cell] is None


class TestReconstructCSIEdgeCases:
    """Edge case tests for reconstruct_csi_from_csv_row."""

    def test_nan_values_produce_zero(self):
        """NaN amp/phase values should produce zero complex value for that subcarrier."""
        row = {}
        for sc in range(128):
            row[f"s1_amp_norm_{sc}"] = float('nan')
            row[f"s1_phase_{sc}"] = float('nan')
        result = reconstruct_csi_from_csv_row(row, shouter_id=1)
        assert np.all(result == 0 + 0j)

    def test_single_nonzero_subcarrier(self):
        """Only one subcarrier with data; rest should be zero."""
        row = {}
        for sc in range(128):
            row[f"s1_amp_norm_{sc}"] = float('nan')
            row[f"s1_phase_{sc}"] = float('nan')
        # Set subcarrier 10 to amp=2.0, phase=0
        row["s1_amp_norm_10"] = 2.0
        row["s1_phase_10"] = 0.0
        result = reconstruct_csi_from_csv_row(row, shouter_id=1)
        assert abs(result[10]) == pytest.approx(2.0, abs=0.01)
        assert result[0] == 0 + 0j


class TestRunSarCLI:
    def test_help_flag(self):
        """run_sar.py --help should exit 0 and show usage."""
        result = subprocess.run(
            [sys.executable, "run_sar.py", "--help"],
            capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0
        assert "--port" in result.stdout
        assert "--demo" in result.stdout

    def test_help_shows_demo_and_fullscreen(self):
        """run_sar.py --help should show --demo and --fullscreen flags."""
        result = subprocess.run(
            [sys.executable, "run_sar.py", "--help"],
            capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0
        assert "--demo" in result.stdout
        assert "--fullscreen" in result.stdout


class TestCSVReplay:
    """Tests for CSV reconstruction utility (legacy — replay removed from run_sar.py)."""

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


class TestBreathingDisplay:
    def test_import_guarded(self):
        """BreathingDisplay should be importable when pygame is available."""
        pygame = pytest.importorskip("pygame")
        from ghv4.breathing import BreathingDisplay
        assert BreathingDisplay is not None


class TestDemoThread:
    def test_produces_valid_grid_scores(self):
        """DemoThread should put dicts with 'type'='scores' and valid grid data."""
        import threading
        from ghv4.breathing import SARDemoThread
        result_queue = queue.Queue()
        stop_event = threading.Event()
        thread = SARDemoThread(result_queue, stop_event)
        thread.start()
        # Wait for a 'scores' item (first item may be a 'status' message)
        item = None
        try:
            import time
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                try:
                    candidate = result_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                if candidate["type"] == "scores":
                    item = candidate
                    break
        finally:
            stop_event.set()
            thread.join(timeout=2.0)
        assert item is not None, "No 'scores' item received within timeout"
        assert item["type"] == "scores"
        assert "grid" in item
        assert "r0c0" in item["grid"]
        assert "path_conf" in item
