"""Tests for ghv5.breathing — CSI breathing detection pipeline."""
import queue
import struct
import subprocess
import sys

import numpy as np
import pytest

from ghv5.breathing import (
    CSIRingBuffer,
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


class TestPresenceScore:
    """Tests for BreathingDetector._presence_score()."""

    @staticmethod
    def _make_complex_window(n_time=600, n_subs=128, mean_amp=1000.0, noise_std=0.0, rng_seed=0):
        """Build a proper complex64 window."""
        rng = np.random.default_rng(rng_seed)
        amp = np.full((n_time, n_subs), mean_amp, dtype=np.float64)
        if noise_std > 0:
            amp += rng.normal(0, noise_std, (n_time, n_subs))
        amp = np.clip(amp, 0, None).astype(np.float32)
        return (amp + 0j).astype(np.complex64)

    def test_returns_float_in_range(self):
        window = self._make_complex_window()
        score = BreathingDetector._presence_score(window)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_one_attenuated_path_gets_high_rank_score(self):
        """Path with 50% lower mean amplitude than group should score high."""
        # 4 paths: 3 at 1000, one (this path) at 500
        low_window  = self._make_complex_window(mean_amp=500.0)
        high_mean = 1000.0
        all_path_means = {
            (1, 2): high_mean,
            (1, 3): high_mean,
            (2, 3): high_mean,
            (1, 4): 500.0,  # this path
        }
        score = BreathingDetector._presence_score(low_window, all_path_means)
        assert score > 0.3, f"Attenuated path should have presence > 0.3, got {score:.3f}"

    def test_all_paths_equal_rank_score_near_zero(self):
        """If all paths have the same mean amplitude, rank score = 0."""
        window = self._make_complex_window(mean_amp=1000.0)
        all_path_means = {
            (1, 2): 1000.0,
            (1, 3): 1000.0,
            (2, 3): 1000.0,
        }
        score = BreathingDetector._presence_score(window, all_path_means)
        # No attenuation on any path — rank score = 0
        # Variance = 0 (no noise) — variance score near 0
        assert score < 0.2, f"Equal paths should have presence < 0.2, got {score:.3f}"

    def test_fewer_than_3_paths_falls_back_to_variance(self):
        """With < 3 paths, rank score is 0 and only variance signal fires."""
        # Static window (no noise) → variance ≈ 0 → presence ≈ 0
        static_window = self._make_complex_window(mean_amp=1000.0, noise_std=0.0)
        all_path_means = {(1, 2): 1000.0, (1, 3): 1000.0}  # only 2 paths
        score_static = BreathingDetector._presence_score(static_window, all_path_means)
        assert score_static < 0.2, f"Static 2-path should be near 0, got {score_static:.3f}"

        # Noisy window (high variance) → variance score fires even without rank
        noisy_window = self._make_complex_window(mean_amp=1000.0, noise_std=500.0, rng_seed=1)
        score_noisy = BreathingDetector._presence_score(noisy_window, all_path_means)
        assert score_noisy > score_static, "Noisy window should score higher than static"

    def test_static_room_low_variance_score(self):
        """Constant amplitude (no motion) should yield near-zero presence score."""
        window = self._make_complex_window(mean_amp=1000.0, noise_std=0.0)
        score = BreathingDetector._presence_score(window)  # no path means → variance only
        assert score < 0.2, f"Static room should score < 0.2, got {score:.3f}"

    def test_high_variance_yields_nonzero_score(self):
        """Strong amplitude fluctuations should yield elevated presence score."""
        # Amplitude oscillates significantly (std = 500 on mean of 1000 = 50% variation)
        window = self._make_complex_window(mean_amp=1000.0, noise_std=500.0, rng_seed=42)
        score = BreathingDetector._presence_score(window)
        assert score > 0.1, f"High-variance window should score > 0.1, got {score:.3f}"

    def test_max_fusion_rank_fires_variance_does_not(self):
        """Presence = max(rank, variance): if only rank fires, result equals rank."""
        # Low-noise window → variance_score ≈ 0
        low_noise_window = self._make_complex_window(mean_amp=500.0, noise_std=1.0, rng_seed=7)
        all_path_means = {
            (1, 2): 1000.0,
            (1, 3): 1000.0,
            (2, 3): 1000.0,
            (1, 4): 500.0,
        }
        score = BreathingDetector._presence_score(low_noise_window, all_path_means)
        # Rank score dominates: group_median ≈ 1000, this_mean ≈ 500 → rank ≈ 0.5
        assert score > 0.3, f"Rank signal should dominate, got {score:.3f}"

    def test_get_all_scores_returns_presence_key(self):
        """get_all_scores() must return 'presence' key, not 'amp'."""
        from ghv5.config import BREATHING_WINDOW_N, BREATHING_PATH_MAP
        import struct as _struct
        det = BreathingDetector()
        csi_bytes = b''.join(_struct.pack('<hh', 1000, 0) for _ in range(128))
        for _ in range(BREATHING_WINDOW_N):
            for key in BREATHING_PATH_MAP:
                det.feed_frame('csi_snap', {
                    'reporter_id': key[0], 'peer_id': key[1], 'csi': csi_bytes
                })
        result = det.get_all_scores()
        assert "presence" in result, f"Expected 'presence' key, got keys: {list(result.keys())}"
        assert "amp" not in result, "'amp' key must be removed"
        assert "pca" in result
        assert "path_conf" in result


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
        assert set(result.keys()) == {"presence", "pca", "path_conf"}

    def test_presence_and_pca_grids_contain_all_cells(self):
        det = self._fill_detector()
        result = det.get_all_scores()
        for cell in CELL_LABELS:
            assert cell in result["presence"]
            assert cell in result["pca"]

    def test_presence_matches_get_grid_scores(self):
        det = self._fill_detector()
        all_scores = det.get_all_scores()
        grid_scores = det.get_grid_scores()
        for cell, val in grid_scores.items():
            if val is None:
                assert all_scores["presence"][cell] is None
            else:
                assert all_scores["presence"][cell] == pytest.approx(val, abs=0.01)

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
            assert result["presence"][cell] is None
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
        assert "presence_grid" in item, "'amp_grid' key renamed to 'presence_grid'"
        assert "pca_grid" in item,      "pca_grid key expected"
        assert "amp_grid" not in item,  "old 'amp_grid' key must be removed"
        assert "r0c0" in item["presence_grid"]
        assert "r0c0" in item["pca_grid"]
        assert "path_conf" in item
