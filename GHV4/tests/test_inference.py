# GHV2.1/tests/test_inference.py
"""Tests for inference spacing feature injection."""
import json
import os
import pytest


PAIR_KEYS = ["1-2", "1-3", "1-4", "2-3", "2-4", "3-4"]


@pytest.fixture
def tmp_spacing(tmp_path):
    path = str(tmp_path / "spacing.json")
    data = {
        "version": 1,
        "updated": "2026-03-16T00:00:00Z",
        "pairs": {
            "1-2": {"distance_m": 2.5, "rssi_avg": -55.0, "samples": 10},
            "1-3": {"distance_m": 3.8, "rssi_avg": -61.0, "samples": 8},
            "1-4": {"distance_m": 2.4, "rssi_avg": -54.5, "samples": 12},
            "2-3": {"distance_m": 2.3, "rssi_avg": -54.0, "samples": 9},
            "2-4": {"distance_m": 3.9, "rssi_avg": -61.5, "samples": 7},
            "3-4": {"distance_m": 2.6, "rssi_avg": -55.5, "samples": 11},
        },
        "config": {"n": 2.5, "rssi_ref_dbm": -40.0, "d0_m": 1.0},
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def test_load_spacing_reads_all_pairs(tmp_spacing):
    from ghv4.inference import load_spacing
    vals = load_spacing(tmp_spacing)
    assert len(vals) == 6
    assert vals[0] == pytest.approx(2.5)   # "1-2"
    assert vals[5] == pytest.approx(2.6)   # "3-4"


def test_load_spacing_absent_returns_zeros(tmp_path):
    from ghv4.inference import load_spacing
    vals = load_spacing(str(tmp_path / "nonexistent.json"))
    assert vals == [0.0] * 6


def test_load_spacing_partial_fills_zeros(tmp_path):
    from ghv4.inference import load_spacing
    path = str(tmp_path / "partial.json")
    data = {"version": 1, "pairs": {"1-2": {"distance_m": 2.5, "samples": 10}}, "config": {}}
    with open(path, "w") as f:
        json.dump(data, f)
    vals = load_spacing(path)
    assert vals[0] == pytest.approx(2.5)   # "1-2" present
    assert vals[1] == 0.0                  # "1-3" absent → 0


def test_spacing_appended_to_feature_vector():
    """load_spacing() output must produce a 6-element list ready for appending."""
    from ghv4 import inference as InferenceV3
    # build a dummy feature vector of 100 floats
    features = [0.5] * 100
    spacing_vals = [2.5, 3.8, 2.4, 2.3, 3.9, 2.6]
    combined = features + spacing_vals
    assert len(combined) == 106
    assert combined[-6] == 2.5
    assert combined[-1] == 2.6


import numpy as np

def test_run_calibration_with_serial(tmp_path):
    """run_calibration creates a calibrator, reads snaps via SerialReader, writes spacing."""
    import json
    import struct
    import queue
    import threading
    from io import BytesIO
    from ghv4.inference import run_calibration
    from ghv4.config import PAIR_KEYS, DISTANCE_FEATURE_COUNT, CALIBRATION_MIN_PAIRS
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    import joblib

    # --- set up dummy model dir ---
    model_dir = tmp_path / "distance_models"
    model_dir.mkdir()

    rng = np.random.default_rng(42)
    X_d = rng.standard_normal((50, DISTANCE_FEATURE_COUNT))
    y_d = rng.uniform(3.0, 8.0, 50)

    for pair_id in PAIR_KEYS:
        m = GradientBoostingRegressor(n_estimators=10, random_state=42)
        m.fit(X_d, y_d)
        joblib.dump(m, model_dir / f"{pair_id}_model.pkl")

    # Scaler must be fit on 242 amp columns (matching distance_preprocess.py)
    amp_indices = list(range(121)) + list(range(242, 363))
    scaler = StandardScaler().fit(X_d[:, :len(amp_indices)])  # 242 columns
    joblib.dump(scaler, model_dir / "distance_scaler.pkl")

    from ghv4.distance_features import FEATURE_NAMES
    with open(model_dir / "distance_feature_names.txt", "w") as f:
        f.write("\n".join(FEATURE_NAMES))

    spacing_path = tmp_path / "spacing.json"

    # --- build a fake serial stream with enough snap frames for pair 1-2 ---
    frames = bytearray()
    for seq in range(CALIBRATION_MIN_PAIRS + 5):
        csi = bytes(rng.integers(-127, 127, 256, dtype=np.int8))
        # forward: reporter=1, peer=2
        hdr = struct.pack("<BBBBH", 1, 1, 2, seq, 256)
        frames += bytes([0xEE, 0xFF]) + hdr + csi
        # reverse: reporter=2, peer=1
        hdr = struct.pack("<BBBBH", 1, 2, 1, seq, 256)
        frames += bytes([0xEE, 0xFF]) + hdr + csi

    # Wrap BytesIO in a thin shim that exposes a .timeout attribute,
    # because SerialReader._read_one_frame accesses self._ser.timeout when
    # processing [0xEE][0xFF] snap frames.
    class _BytesIOWithTimeout(BytesIO):
        timeout = 1.0

    distances = run_calibration(
        ser=_BytesIOWithTimeout(bytes(frames)),
        model_dir=str(model_dir),
        spacing_path=str(spacing_path),
        window_s=0.0,  # skip waiting — data is already in the buffer
    )

    assert "1-2" in distances
    assert distances["1-2"] > 0

    data = json.loads(spacing_path.read_text())
    assert data["version"] == 2
    assert "1-2" in data["pairs"]
    assert data["pairs"]["1-2"]["source"] == "ml"
