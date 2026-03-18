# GHV4/tests/test_spacing_estimator.py
"""Unit tests for SpacingEstimator (MUSIC-only, GHV4).

RSSI-based distance tests removed — GHV4 uses MUSIC exclusively.
"""
import json
import os
import time

import pytest


@pytest.fixture
def tmp_spacing(tmp_path):
    return str(tmp_path / "spacing.json")


# ── SpacingEstimator basic API ───────────────────────────────────────────────

def test_get_distances_empty_when_no_music(tmp_spacing):
    from ghv4.spacing_estimator import SpacingEstimator
    est = SpacingEstimator(spacing_path=tmp_spacing, music_estimator=None)
    assert est.get_distances() == {}


def test_get_distances_delegates_to_music(tmp_spacing):
    from ghv4.spacing_estimator import SpacingEstimator, CSIMUSICEstimator
    music_est = CSIMUSICEstimator()
    with music_est._lock:
        music_est._distances['1-2'] = 7.1
    est = SpacingEstimator(spacing_path=tmp_spacing, music_estimator=music_est)
    distances = est.get_distances()
    assert distances.get('1-2') == pytest.approx(7.1)


def test_get_distances_returns_copy(tmp_spacing):
    from ghv4.spacing_estimator import SpacingEstimator, CSIMUSICEstimator
    music_est = CSIMUSICEstimator()
    with music_est._lock:
        music_est._distances['1-2'] = 5.0
    est = SpacingEstimator(spacing_path=tmp_spacing, music_estimator=music_est)
    d = est.get_distances()
    d["fake"] = 99.9
    assert "fake" not in est.get_distances()


def test_feed_is_noop(tmp_spacing):
    from ghv4.spacing_estimator import SpacingEstimator
    est = SpacingEstimator(spacing_path=tmp_spacing)
    est.feed({})  # should not raise


def test_get_rssi_values_returns_empty(tmp_spacing):
    from ghv4.spacing_estimator import SpacingEstimator
    est = SpacingEstimator(spacing_path=tmp_spacing)
    assert est.get_rssi_values() == {}


# ── _maybe_write() / spacing.json ────────────────────────────────────────────

def test_maybe_write_creates_json(tmp_spacing):
    from ghv4.spacing_estimator import SpacingEstimator, CSIMUSICEstimator
    music_est = CSIMUSICEstimator()
    with music_est._lock:
        music_est._distances['1-2'] = 7.5
    est = SpacingEstimator(spacing_path=tmp_spacing, music_estimator=music_est)
    est._last_write = 0
    est._maybe_write()
    assert os.path.exists(tmp_spacing)
    with open(tmp_spacing) as f:
        data = json.load(f)
    assert data["version"] == 2
    assert data["pairs"]["1-2"]["distance_m"] == pytest.approx(7.5)
    assert data["pairs"]["1-2"]["source"] == "music"


def test_maybe_write_rate_limited(tmp_spacing):
    from ghv4.spacing_estimator import SpacingEstimator, CSIMUSICEstimator
    music_est = CSIMUSICEstimator()
    with music_est._lock:
        music_est._distances['1-2'] = 5.0
    est = SpacingEstimator(spacing_path=tmp_spacing, music_estimator=music_est)
    est._last_write = 0
    est._maybe_write()
    mtime1 = os.path.getmtime(tmp_spacing)
    est._maybe_write()  # should be skipped (< 1s elapsed)
    mtime2 = os.path.getmtime(tmp_spacing)
    assert mtime1 == mtime2


def test_maybe_write_atomic(tmp_spacing):
    from ghv4.spacing_estimator import SpacingEstimator, CSIMUSICEstimator
    music_est = CSIMUSICEstimator()
    with music_est._lock:
        music_est._distances['1-2'] = 5.0
    est = SpacingEstimator(spacing_path=tmp_spacing, music_estimator=music_est)
    est._last_write = 0
    est._maybe_write()
    assert not os.path.exists(tmp_spacing + ".tmp")


def test_maybe_write_skips_when_no_distances(tmp_spacing):
    from ghv4.spacing_estimator import SpacingEstimator, CSIMUSICEstimator
    music_est = CSIMUSICEstimator()
    est = SpacingEstimator(spacing_path=tmp_spacing, music_estimator=music_est)
    est._last_write = 0
    est._maybe_write()
    assert not os.path.exists(tmp_spacing)


# ── CSIMUSICEstimator guard checks ───────────────────────────────────────────

def test_music_collect_rejects_below_noise_floor():
    from ghv4.spacing_estimator import CSIMUSICEstimator
    est = CSIMUSICEstimator()
    zero_csi = bytes(256)
    est.collect(1, 2, zero_csi)
    with est._lock:
        assert (1, 2) not in est._H or len(est._H[(1, 2)]) == 0


def test_music_delay_handles_degenerate_input():
    import numpy as np
    from ghv4.spacing_estimator import CSIMUSICEstimator
    est = CSIMUSICEstimator()
    H = np.ones((121, 5), dtype=complex)
    result = est._music_delay(H)
    assert result is None or isinstance(result, float)


def test_music_delay_returns_none_on_linalg_error(monkeypatch):
    import numpy as np
    from ghv4.spacing_estimator import CSIMUSICEstimator
    est = CSIMUSICEstimator()

    def _raise_linalg_error(*args, **kwargs):
        raise np.linalg.LinAlgError("test")

    monkeypatch.setattr(np.linalg, "eigh", _raise_linalg_error)
    H = np.random.randn(121, 5) + 1j * np.random.randn(121, 5)
    result = est._music_delay(H)
    assert result is None


def test_music_reset_all():
    from ghv4.spacing_estimator import CSIMUSICEstimator
    est = CSIMUSICEstimator()
    with est._lock:
        est._distances['1-2'] = 5.0
        est._H[(1, 2)] = ['dummy']
    est.reset_all()
    with est._lock:
        assert len(est._distances) == 0
        assert len(est._H) == 0


def test_music_reset_pair():
    from ghv4.spacing_estimator import CSIMUSICEstimator
    est = CSIMUSICEstimator()
    with est._lock:
        est._distances['1-2'] = 5.0
        est._distances['1-3'] = 6.0
        est._H[(1, 2)] = ['dummy']
        est._H[(2, 1)] = ['dummy']
    est.reset_pair(1, 2)
    with est._lock:
        assert '1-2' not in est._distances
        assert '1-3' in est._distances
        assert (1, 2) not in est._H
        assert (2, 1) not in est._H
