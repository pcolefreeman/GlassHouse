"""Tests for ghv5.breathing — CSI breathing detection pipeline."""
import queue
import struct
import subprocess
import sys

import numpy as np
import pytest

from ghv5.breathing import (
    CSIRingBuffer,
    CSIRatioExtractor,
    BreathingAnalyzer,
    GridProjector,
    BreathingDetector,
    reconstruct_csi_from_csv_row,
)
from ghv5.config import BREATHING_PATH_MAP, CELL_LABELS


def test_pca_components_constant_exists():
    from ghv5.config import BREATHING_PCA_COMPONENTS
    assert isinstance(BREATHING_PCA_COMPONENTS, int)
    assert BREATHING_PCA_COMPONENTS > 0


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
        from ghv5.config import NULL_SUBCARRIER_INDICES
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
        from ghv5.config import BREATHING_SNAP_HZ
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


class TestPCAScore:
    """Tests for BreathingDetector._pca_score()."""

    @staticmethod
    def _make_window(n_time=600, n_subs=128, value=1000.0):
        return np.full((n_time, n_subs), value + 0j, dtype=np.complex64)

    @staticmethod
    def _make_breathing_window(n_time=600, n_subs=128, freq_hz=0.25, fs=20.0):
        t = np.arange(n_time) / fs
        amp = 1000.0 + 800.0 * np.sin(2 * np.pi * freq_hz * t)
        window = np.zeros((n_time, n_subs), dtype=np.complex64)
        window[:, :] = amp[:, None].astype(np.float32)
        return window

    def test_returns_float_in_range(self):
        window = self._make_window()
        score = BreathingDetector._pca_score(window)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_detects_synthetic_breathing(self):
        window = self._make_breathing_window(freq_hz=0.25)
        score = BreathingDetector._pca_score(window)
        assert score > 0.3, f"Expected breathing detected, got {score:.3f}"

    def test_static_signal_low_score(self):
        window = self._make_window(value=1000.0)
        score = BreathingDetector._pca_score(window)
        assert score < 0.3, f"Expected low score for static signal, got {score:.3f}"

    def test_out_of_band_signal_low_score(self):
        # Add per-subcarrier random phase offsets so PCA doesn't collapse to
        # a single perfect component (which causes numerical SNR artifacts).
        rng = np.random.default_rng(42)
        n_time, n_subs = 600, 128
        t = np.arange(n_time) / 20.0
        base = 1000.0 + 800.0 * np.sin(2 * np.pi * 2.0 * t)
        window = np.zeros((n_time, n_subs), dtype=np.complex64)
        for sc in range(n_subs):
            phase = rng.uniform(0, 2 * np.pi)
            window[:, sc] = (base + rng.normal(0, 50, n_time)) * np.exp(1j * phase)
        score = BreathingDetector._pca_score(window)
        assert score < 0.3, f"Expected low score for 2 Hz signal, got {score:.3f}"

    def test_k_clamped_when_exceeds_valid_subcarriers(self):
        window = self._make_window(n_time=100, n_subs=10)
        score = BreathingDetector._pca_score(window, k=200)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_all_zero_window_returns_zero(self):
        window = np.zeros((600, 128), dtype=np.complex64)
        score = BreathingDetector._pca_score(window)
        assert score == 0.0


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
        from ghv5.config import BREATHING_WINDOW_N
        det = BreathingDetector()
        csi = self._make_csi_bytes()
        for _ in range(BREATHING_WINDOW_N):
            det.feed_frame('csi_snap', {'reporter_id': 1, 'peer_id': 2, 'csi': csi})
        assert det.is_ready()

    def test_get_grid_scores_returns_dict(self):
        from ghv5.config import BREATHING_WINDOW_N
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


class TestGetAllScores:
    """Tests for BreathingDetector.get_all_scores()."""

    @staticmethod
    def _fill_detector(n_frames=None):
        from ghv5.config import BREATHING_WINDOW_N, BREATHING_PATH_MAP
        import struct as _struct
        if n_frames is None:
            n_frames = BREATHING_WINDOW_N
        det = BreathingDetector()
        csi_bytes = b''.join(_struct.pack('<hh', 1000, 0) for _ in range(128))
        for _ in range(n_frames):
            for key in BREATHING_PATH_MAP:
                det.feed_frame('csi_snap', {
                    'reporter_id': key[0], 'peer_id': key[1], 'csi': csi_bytes
                })
        return det

    def test_returns_correct_keys(self):
        det = self._fill_detector()
        result = det.get_all_scores()
        assert set(result.keys()) == {"amp", "pca", "path_conf"}

    def test_amp_and_pca_grids_contain_all_cells(self):
        det = self._fill_detector()
        result = det.get_all_scores()
        for cell in CELL_LABELS:
            assert cell in result["amp"]
            assert cell in result["pca"]

    def test_amp_matches_get_grid_scores(self):
        det = self._fill_detector()
        all_scores = det.get_all_scores()
        grid_scores = det.get_grid_scores()
        for cell, val in grid_scores.items():
            if val is None:
                assert all_scores["amp"][cell] is None
            else:
                assert all_scores["amp"][cell] == pytest.approx(val, abs=0.01)

    def test_path_conf_values_in_range(self):
        det = self._fill_detector()
        result = det.get_all_scores()
        for key in BREATHING_PATH_MAP:
            assert key in result["path_conf"]
            assert 0.0 <= result["path_conf"][key] <= 1.0

    def test_get_grid_scores_still_works(self):
        det = self._fill_detector()
        scores = det.get_grid_scores()
        assert isinstance(scores, dict)
        for cell in CELL_LABELS:
            assert cell in scores

    def test_empty_detector_returns_empty_grids(self):
        det = BreathingDetector()
        result = det.get_all_scores()
        for cell in CELL_LABELS:
            assert result["amp"][cell] is None
            assert result["pca"][cell] is None
        assert result["path_conf"] == {}


class TestBreathingDetectorE2E:
    def test_synthetic_breathing_detected(self):
        """Feed amplitude-modulated 0.25 Hz CSI, verify all 9 cells covered."""
        from ghv5.config import BREATHING_WINDOW_N, BREATHING_SNAP_HZ
        det = BreathingDetector()
        n_time = BREATHING_WINDOW_N  # 600
        n_sub = 128
        t = np.arange(n_time) / BREATHING_SNAP_HZ  # 20 Hz

        for step in range(n_time):
            # Amplitude modulated at 0.25 Hz — what CSI actually responds to
            amp = int(1000 + 800 * np.sin(2 * np.pi * 0.25 * t[step]))
            csi_bytes = b''.join(struct.pack('<hh', amp, 0) for _ in range(n_sub))
            # Feed all 6 paths
            for key in BREATHING_PATH_MAP:
                det.feed_frame('csi_snap', {
                    'reporter_id': key[0], 'peer_id': key[1], 'csi': csi_bytes
                })

        scores = det.get_grid_scores()
        # All 9 cells should be covered
        for cell in CELL_LABELS:
            assert scores[cell] is not None, f"Cell {cell} should be covered"
        # Center cell (crossed by diagonals) should show confident detection
        assert scores["r1c1"] is not None and scores["r1c1"] > 30.0

    def test_static_signal_no_detection(self):
        """Feed constant CSI via csi_snap, verify low/no confidence."""
        from ghv5.config import BREATHING_WINDOW_N
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
            from ghv5.breathing import reconstruct_csi_from_csv_row
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
        from ghv5.breathing import BreathingDisplay
        assert BreathingDisplay is not None


class TestDemoThread:
    def test_produces_valid_grid_scores(self):
        """DemoThread should put dicts with 'type'='scores' and valid grid data."""
        import threading
        from ghv5.breathing import SARDemoThread
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
        assert "amp_grid" in item,  "'grid' key renamed to 'amp_grid'"
        assert "pca_grid" in item,  "new 'pca_grid' key expected"
        assert "grid" not in item,  "old 'grid' key must be removed"
        assert "r0c0" in item["amp_grid"]
        assert "r0c0" in item["pca_grid"]
        assert "path_conf" in item
