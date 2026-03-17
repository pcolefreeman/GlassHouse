# GHV2.1/tests/test_spacing_estimator.py
"""Unit tests for SpacingEstimator.

All tests use feed() directly (bypassing the daemon thread) by calling
_process() for synchronous control. The thread integration is tested
separately via feed() + time.sleep().
"""
import json
import math
import os
import struct
import tempfile
import time

import pytest


def _make_payload(reporter_id: int, peer_rssi: list, peer_count: list) -> bytes:
    """Build a 12-byte frame C payload for testing."""
    return struct.pack('<BB5b5B', 1, reporter_id, *peer_rssi, *peer_count)


@pytest.fixture
def tmp_spacing(tmp_path):
    return str(tmp_path / "spacing.json")


@pytest.fixture
def estimator(tmp_spacing, tmp_path):
    from ghv3_1.spacing_estimator import SpacingEstimator
    # Use a temp config with generic defaults so distance tests are deterministic
    cfg_path = str(tmp_path / "test_ranging_config.json")
    with open(cfg_path, "w") as f:
        json.dump({"n": 2.5, "rssi_ref_dbm": -40.0, "d0_m": 1.0}, f)
    return SpacingEstimator(spacing_path=tmp_spacing, config_path=cfg_path)


# ── _distance() ──────────────────────────────────────────────────────────────

def test_distance_at_ref_rssi(estimator):
    """At rssi_ref (-40 dBm) and d0=1.0m, distance should be 1.0m."""
    d = estimator._distance(-40.0)
    assert abs(d - 1.0) < 1e-6


def test_distance_increases_as_rssi_decreases(estimator):
    """Weaker RSSI → greater distance."""
    d_near = estimator._distance(-40.0)
    d_far  = estimator._distance(-60.0)
    assert d_far > d_near


def test_distance_formula(estimator):
    """Manual verification: d0=1, rssi_ref=-40, n=2.5, rssi=-50."""
    # d = 1.0 * 10^((-40 - (-50)) / (10 * 2.5)) = 10^(10/25) = 10^0.4
    expected = 10 ** 0.4
    d = estimator._distance(-50.0)
    assert abs(d - expected) < 1e-6


# ── _process() (single observation) ─────────────────────────────────────────

def test_process_updates_rssi(estimator):
    """First observation initialises _rssi without EMA averaging."""
    payload = _make_payload(
        reporter_id=1,
        peer_rssi=[0, 0, -55, -60, -65],
        peer_count=[0, 0,   5,   3,   7],
    )
    estimator._process({'payload': payload})
    assert estimator._rssi[1][2] == pytest.approx(-55.0)
    assert estimator._rssi[1][3] == pytest.approx(-60.0)


def test_process_applies_ema_on_second_observation(estimator):
    """Second observation blends with alpha=0.1."""
    payload1 = _make_payload(1, [0, 0, -50, 0, 0], [0, 0, 5, 0, 0])
    payload2 = _make_payload(1, [0, 0, -60, 0, 0], [0, 0, 5, 0, 0])
    estimator._process({'payload': payload1})
    estimator._process({'payload': payload2})
    # After init: rssi = -50. After EMA: 0.9*(-50) + 0.1*(-60) = -51.0
    assert estimator._rssi[1][2] == pytest.approx(-51.0, abs=0.01)


def test_process_ignores_zero_count_peers(estimator):
    """Peers with count=0 in the payload must not update _rssi."""
    payload = _make_payload(2, [0, -50, 0, -60, -55], [0, 0, 0, 5, 5])
    estimator._process({'payload': payload})
    # peer 1 has count=0 — should not update
    assert estimator._rssi[2][1] == 0.0
    # peer 3 has count=5 — should update
    assert estimator._rssi[2][3] == pytest.approx(-60.0)


def test_process_short_payload_ignored(estimator):
    """Payloads shorter than 12 bytes must be silently dropped."""
    estimator._process({'payload': bytes(6)})
    assert estimator._count.sum() == 0


# ── get_distances() ───────────────────────────────────────────────────────────

def test_get_distances_empty_when_no_data(estimator):
    assert estimator.get_distances() == {}


def test_get_distances_omits_pair_below_min_samples(estimator):
    """Pairs with 0 samples in either direction must be omitted."""
    # S1 sees S2 but S2 has not yet reported back (count[2][1] == 0)
    payload = _make_payload(1, [0, 0, -55, 0, 0], [0, 0, 1, 0, 0])
    estimator._process({'payload': payload})
    # S2 has never sent a frame — count[2][1] stays 0
    assert "1-2" not in estimator.get_distances()


def test_get_distances_includes_pair_when_both_above_threshold(estimator):
    """Pair appears once both directions have >= 1 sample."""
    # Feed 1 observation each direction
    for _ in range(1):
        p1 = _make_payload(1, [0, 0, -55, 0, 0], [0, 0, 1, 0, 0])
        p2 = _make_payload(2, [0, -55, 0, 0, 0], [0, 1, 0, 0, 0])
        estimator._process({'payload': p1})
        estimator._process({'payload': p2})
    distances = estimator.get_distances()
    assert "1-2" in distances
    assert distances["1-2"] > 0


def test_get_distances_returns_copy(estimator):
    """Mutating the returned dict must not affect internal state."""
    d = estimator.get_distances()
    d["fake"] = 99.9
    assert "fake" not in estimator.get_distances()


# ── _maybe_write() / spacing.json ────────────────────────────────────────────

def test_maybe_write_creates_json_after_threshold(estimator, tmp_spacing):
    """spacing.json is written once pair has >= 5 bidirectional samples."""
    for _ in range(5):
        p1 = _make_payload(1, [0, 0, -50, 0, 0], [0, 0, 1, 0, 0])
        p2 = _make_payload(2, [0, -50, 0, 0, 0], [0, 1, 0, 0, 0])
        estimator._process({'payload': p1})
        estimator._process({'payload': p2})
    estimator._last_write = 0  # reset rate limiter so write happens
    estimator._maybe_write()
    assert os.path.exists(tmp_spacing)
    with open(tmp_spacing) as f:
        data = json.load(f)
    assert "1-2" in data["pairs"]
    assert data["pairs"]["1-2"]["distance_m"] > 0
    assert data["pairs"]["1-2"]["samples"] == 5
    assert data["pairs"]["1-2"]["source"] == "rssi"


def test_maybe_write_rate_limited(estimator, tmp_spacing, monkeypatch):
    """Second call within 1 second must not rewrite the file."""
    for _ in range(5):
        p1 = _make_payload(1, [0, 0, -50, 0, 0], [0, 0, 1, 0, 0])
        p2 = _make_payload(2, [0, -50, 0, 0, 0], [0, 1, 0, 0, 0])
        estimator._process({'payload': p1})
        estimator._process({'payload': p2})
    estimator._last_write = 0
    estimator._maybe_write()
    mtime1 = os.path.getmtime(tmp_spacing)
    estimator._maybe_write()  # should be skipped (< 1s elapsed)
    mtime2 = os.path.getmtime(tmp_spacing)
    assert mtime1 == mtime2


def test_maybe_write_atomic(estimator, tmp_spacing):
    """spacing.json must never exist in a partially-written state."""
    # Verify that _maybe_write uses os.replace (not direct open)
    # We check that no .tmp file is left over after a write
    for _ in range(5):
        p1 = _make_payload(1, [0, 0, -50, 0, 0], [0, 0, 1, 0, 0])
        p2 = _make_payload(2, [0, -50, 0, 0, 0], [0, 1, 0, 0, 0])
        estimator._process({'payload': p1})
        estimator._process({'payload': p2})
    estimator._last_write = 0
    estimator._maybe_write()
    tmp_file = tmp_spacing + ".tmp"
    assert not os.path.exists(tmp_file)


# ── Config loading ────────────────────────────────────────────────────────────

def test_default_config_used_when_file_absent(tmp_spacing):
    from ghv3_1.spacing_estimator import SpacingEstimator, _DEFAULT_CONFIG
    est = SpacingEstimator(spacing_path=tmp_spacing, config_path="nonexistent.json")
    assert est._config == _DEFAULT_CONFIG


def test_custom_config_loaded(tmp_path, tmp_spacing):
    from ghv3_1.spacing_estimator import SpacingEstimator
    cfg_path = str(tmp_path / "ranging_config.json")
    with open(cfg_path, "w") as f:
        json.dump({"n": 3.0, "rssi_ref_dbm": -45.0, "d0_m": 2.0}, f)
    est = SpacingEstimator(spacing_path=tmp_spacing, config_path=cfg_path)
    assert est._config["n"] == 3.0
    assert est._config["rssi_ref_dbm"] == -45.0
    assert est._config["d0_m"] == 2.0


# ── SpacingEstimator + CSIMUSICEstimator integration ─────────────────────────

def test_get_distances_uses_music_when_available(tmp_spacing):
    """MUSIC distances must override RSSI for the same pair key."""
    from ghv3_1.spacing_estimator import SpacingEstimator, CSIMUSICEstimator
    music_est = CSIMUSICEstimator()
    # Inject a MUSIC distance directly into internal state
    with music_est._lock:
        music_est._distances['1-2'] = 7.1
    est = SpacingEstimator(spacing_path=tmp_spacing, music_estimator=music_est)
    # Also feed RSSI so SpacingEstimator has a value for 1-2
    p1 = _make_payload(1, [0, 0, -55, 0, 0], [0, 0, 1, 0, 0])
    p2 = _make_payload(2, [0, -55, 0, 0, 0], [0, 1, 0, 0, 0])
    est._process({'payload': p1})
    est._process({'payload': p2})
    distances = est.get_distances()
    # MUSIC value (7.1) must override RSSI value
    assert distances.get('1-2') == pytest.approx(7.1)


def test_get_distances_rssi_fallback_when_no_music(tmp_spacing):
    """When music_estimator is None, distances are RSSI-based."""
    from ghv3_1.spacing_estimator import SpacingEstimator
    est = SpacingEstimator(spacing_path=tmp_spacing, music_estimator=None)
    p1 = _make_payload(1, [0, 0, -55, 0, 0], [0, 0, 1, 0, 0])
    p2 = _make_payload(2, [0, -55, 0, 0, 0], [0, 1, 0, 0, 0])
    est._process({'payload': p1})
    est._process({'payload': p2})
    distances = est.get_distances()
    assert '1-2' in distances
    assert distances['1-2'] > 0


# ── MUSIC guard check tests ───────────────────────────────────────────────────

def test_music_collect_rejects_below_noise_floor():
    """CSI vectors below CSI_NOISE_FLOOR should be rejected."""
    from ghv3_1.spacing_estimator import CSIMUSICEstimator
    est = CSIMUSICEstimator()
    # Create a CSI payload that's all zeros (256 bytes for 128 subcarriers)
    zero_csi = bytes(256)
    est.collect(1, 2, zero_csi)
    with est._lock:
        assert (1, 2) not in est._H or len(est._H[(1, 2)]) == 0


def test_music_delay_handles_degenerate_input():
    """_music_delay must not crash on degenerate (rank-1) input."""
    import numpy as np
    from ghv3_1.spacing_estimator import CSIMUSICEstimator
    est = CSIMUSICEstimator()
    # All-identical rows — rank-1 covariance, eigenvalues mostly zero
    H = np.ones((121, 5), dtype=complex)
    result = est._music_delay(H)
    # Should not raise — returns either a valid tau or None
    assert result is None or isinstance(result, float)


def test_music_delay_returns_none_on_linalg_error(monkeypatch):
    """_music_delay must return None if eigh raises LinAlgError."""
    import numpy as np
    from ghv3_1.spacing_estimator import CSIMUSICEstimator
    est = CSIMUSICEstimator()

    def _raise_linalg_error(*args, **kwargs):
        raise np.linalg.LinAlgError("test")

    monkeypatch.setattr(np.linalg, "eigh", _raise_linalg_error)
    H = np.random.randn(121, 5) + 1j * np.random.randn(121, 5)
    result = est._music_delay(H)
    assert result is None


def test_maybe_write_includes_music_source(tmp_path):
    """spacing.json must show source='music' when MUSIC distance is available."""
    from ghv3_1.spacing_estimator import SpacingEstimator, CSIMUSICEstimator
    sp = str(tmp_path / "spacing.json")
    music_est = CSIMUSICEstimator()
    with music_est._lock:
        music_est._distances['1-2'] = 7.5
    est = SpacingEstimator(spacing_path=sp, music_estimator=music_est)
    # Feed RSSI so pair has data
    p1 = _make_payload(1, [0, 0, -55, 0, 0], [0, 0, 1, 0, 0])
    p2 = _make_payload(2, [0, -55, 0, 0, 0], [0, 1, 0, 0, 0])
    est._process({'payload': p1})
    est._process({'payload': p2})
    est._last_write = 0
    est._maybe_write()
    with open(sp) as f:
        data = json.load(f)
    assert data["pairs"]["1-2"]["source"] == "music"
    assert data["pairs"]["1-2"]["distance_m"] == pytest.approx(7.5)
