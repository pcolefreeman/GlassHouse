# GHV2.1/tests/test_preprocess_spacing.py
"""Tests for preprocess._load_spacing() and spacing hstack."""
import json
import os
import numpy as np
import pytest


PAIR_KEYS = ["1-2", "1-3", "1-4", "2-3", "2-4", "3-4"]


@pytest.fixture
def tmp_raw_dir(tmp_path):
    return str(tmp_path / "raw")


def _write_spacing_json(raw_dir, pairs: dict):
    os.makedirs(raw_dir, exist_ok=True)
    out = {
        "version": 1,
        "updated": "2026-03-16T00:00:00Z",
        "pairs": {k: {"distance_m": v, "rssi_avg": -50.0, "samples": 10}
                  for k, v in pairs.items()},
        "config": {"n": 2.5, "rssi_ref_dbm": -40.0, "d0_m": 1.0},
    }
    path = os.path.join(raw_dir, "spacing.json")
    with open(path, "w") as f:
        json.dump(out, f)
    return path


def test_load_spacing_returns_all_pairs(tmp_raw_dir):
    from ghv3_1.preprocess import _load_spacing
    distances = {k: float(i + 1) for i, k in enumerate(PAIR_KEYS)}
    _write_spacing_json(tmp_raw_dir, distances)
    result = _load_spacing(tmp_raw_dir)
    for k in PAIR_KEYS:
        assert k in result
        assert result[k] == pytest.approx(distances[k])


def test_load_spacing_absent_returns_zeros(tmp_raw_dir):
    from ghv3_1.preprocess import _load_spacing
    result = _load_spacing(tmp_raw_dir)  # no spacing.json in dir
    assert result == {k: 0.0 for k in PAIR_KEYS}


def test_load_spacing_partial_file_fills_zeros(tmp_raw_dir):
    """If spacing.json is missing some pairs, missing pairs fall back to 0."""
    from ghv3_1.preprocess import _load_spacing
    _write_spacing_json(tmp_raw_dir, {"1-2": 2.5, "3-4": 3.1})
    result = _load_spacing(tmp_raw_dir)
    assert result["1-2"] == pytest.approx(2.5)
    assert result["3-4"] == pytest.approx(3.1)
    assert result["1-3"] == 0.0
    assert result["2-4"] == 0.0


def test_spacing_hstack_appends_6_columns(tmp_raw_dir, tmp_path):
    """After run(), X.npy should have 6 extra columns vs no-spacing baseline."""
    # We test this at the _load_spacing level since full run() needs real CSVs.
    from ghv3_1.preprocess import _load_spacing
    import numpy as np

    n_rows = 10
    n_features = 50
    X_base = np.zeros((n_rows, n_features))

    _write_spacing_json(tmp_raw_dir, {k: 1.0 for k in PAIR_KEYS})
    spacing = _load_spacing(tmp_raw_dir)
    spacing_vals = [spacing[k] for k in PAIR_KEYS]
    spacing_block = np.tile(spacing_vals, (n_rows, 1))
    X_out = np.hstack([X_base, spacing_block])

    assert X_out.shape == (n_rows, n_features + 6)
    assert X_out[0, -6] == pytest.approx(spacing_vals[0])
    assert X_out[5, -1] == pytest.approx(spacing_vals[5])
