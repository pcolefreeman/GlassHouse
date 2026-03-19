"""Tests for distance_inference — calibration-phase distance prediction."""
import json
import os
import tempfile
import time
import numpy as np
import pytest
import joblib
from unittest.mock import MagicMock, patch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from ghv4.distance_inference import (
    DistanceCalibrator,
    load_distance_models,
)
from ghv4.config import (
    CALIBRATION_MIN_PAIRS,
    PAIR_KEYS,
    DISTANCE_FEATURE_COUNT,
)


def _make_model_dir(tmp_path):
    """Create a model directory with dummy trained models."""
    model_dir = tmp_path / "distance_models"
    model_dir.mkdir()

    rng = np.random.default_rng(42)
    X_dummy = rng.standard_normal((50, DISTANCE_FEATURE_COUNT))
    y_dummy = rng.uniform(3.0, 8.0, 50)

    for pair_id in PAIR_KEYS:
        model = GradientBoostingRegressor(n_estimators=10, random_state=42)
        model.fit(X_dummy, y_dummy)
        joblib.dump(model, model_dir / f"{pair_id}_model.pkl")

    amp_indices = list(range(121)) + list(range(242, 363))
    scaler = StandardScaler().fit(X_dummy[:, amp_indices])
    joblib.dump(scaler, model_dir / "distance_scaler.pkl")

    from ghv4.distance_features import FEATURE_NAMES
    with open(model_dir / "distance_feature_names.txt", "w") as f:
        f.write("\n".join(FEATURE_NAMES))

    return str(model_dir)


class TestLoadModels:
    def test_loads_all_six(self, tmp_path):
        model_dir = _make_model_dir(tmp_path)
        models, scaler = load_distance_models(model_dir)
        assert len(models) == 6
        assert all(pair in models for pair in PAIR_KEYS)
        assert scaler is not None

    def test_missing_pair_still_loads_rest(self, tmp_path):
        model_dir = _make_model_dir(tmp_path)
        os.remove(os.path.join(model_dir, "3-4_model.pkl"))
        models, scaler = load_distance_models(model_dir)
        assert "3-4" not in models
        assert len(models) == 5


class TestDistanceCalibrator:
    def test_feed_and_predict(self, tmp_path):
        model_dir = _make_model_dir(tmp_path)
        cal = DistanceCalibrator(model_dir)

        # Feed synthetic snap frames for pair 1-2
        rng = np.random.default_rng(42)
        for seq in range(CALIBRATION_MIN_PAIRS + 5):
            csi_fwd = bytes(rng.integers(-127, 127, 256, dtype=np.int8))
            csi_rev = bytes(rng.integers(-127, 127, 256, dtype=np.int8))
            cal.feed_snap(reporter_id=1, peer_id=2, snap_seq=seq, csi_bytes=csi_fwd)
            cal.feed_snap(reporter_id=2, peer_id=1, snap_seq=seq, csi_bytes=csi_rev)

        distances = cal.predict_distances()
        assert "1-2" in distances
        assert isinstance(distances["1-2"], float)
        assert distances["1-2"] > 0

    def test_insufficient_data_returns_none(self, tmp_path):
        model_dir = _make_model_dir(tmp_path)
        cal = DistanceCalibrator(model_dir)

        # Feed only 2 matched pairs (below CALIBRATION_MIN_PAIRS)
        rng = np.random.default_rng(42)
        for seq in range(2):
            csi = bytes(rng.integers(-127, 127, 256, dtype=np.int8))
            cal.feed_snap(reporter_id=1, peer_id=2, snap_seq=seq, csi_bytes=csi)
            cal.feed_snap(reporter_id=2, peer_id=1, snap_seq=seq, csi_bytes=csi)

        distances = cal.predict_distances()
        assert "1-2" not in distances  # below threshold

    def test_write_spacing_json(self, tmp_path):
        model_dir = _make_model_dir(tmp_path)
        spacing_path = tmp_path / "spacing.json"
        cal = DistanceCalibrator(model_dir)

        rng = np.random.default_rng(42)
        for seq in range(CALIBRATION_MIN_PAIRS + 5):
            csi = bytes(rng.integers(-127, 127, 256, dtype=np.int8))
            cal.feed_snap(reporter_id=1, peer_id=2, snap_seq=seq, csi_bytes=csi)
            cal.feed_snap(reporter_id=2, peer_id=1, snap_seq=seq, csi_bytes=csi)

        distances = cal.predict_distances()
        cal.write_spacing(str(spacing_path), distances)

        data = json.loads(spacing_path.read_text())
        assert data["version"] == 2
        assert "1-2" in data["pairs"]
        assert data["pairs"]["1-2"]["source"] == "ml"
