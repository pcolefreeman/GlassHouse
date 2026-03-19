"""Tests for distance_train — per-pair regressor training."""
import os
import tempfile
import numpy as np
import pytest
import joblib
from unittest.mock import patch

from ghv4.distance_train import (
    train_pair_model,
    geometric_consistency_check,
    run,
)
from ghv4.config import DISTANCE_MAX_TREES, DISTANCE_FEATURE_COUNT


class TestTrainPairModel:
    def test_returns_fitted_model(self):
        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, DISTANCE_FEATURE_COUNT))
        y = rng.uniform(2.0, 10.0, n)
        groups = np.array(["s1"] * 50 + ["s2"] * 50)

        model, metrics = train_pair_model(X, y, groups)
        assert hasattr(model, "predict")
        assert "mae" in metrics
        assert "rmse" in metrics
        assert metrics["mae"] >= 0

    def test_max_trees_respected(self):
        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, DISTANCE_FEATURE_COUNT))
        y = rng.uniform(2.0, 10.0, n)
        groups = np.array(["s1"] * 50 + ["s2"] * 50)

        model, _ = train_pair_model(X, y, groups)
        if hasattr(model, "n_estimators"):
            assert model.n_estimators <= DISTANCE_MAX_TREES


class TestGeometricConsistency:
    def test_consistent_rectangle(self):
        distances = {
            "1-4": 3.0, "2-3": 3.0,  # width
            "1-2": 4.0, "3-4": 4.0,  # depth
            "1-3": 5.0, "2-4": 5.0,  # diagonal
        }
        errors = geometric_consistency_check(distances)
        assert errors["diag_1-3"] == pytest.approx(0.0, abs=0.01)
        assert errors["diag_2-4"] == pytest.approx(0.0, abs=0.01)

    def test_inconsistent_diagonal(self):
        distances = {
            "1-4": 3.0, "2-3": 3.0,
            "1-2": 4.0, "3-4": 4.0,
            "1-3": 8.0, "2-4": 8.0,  # wrong diagonal
        }
        errors = geometric_consistency_check(distances)
        assert abs(errors["diag_1-3"]) > 1.0


class TestRunPipeline:
    def test_creates_model_files(self, tmp_path):
        processed_dir = tmp_path / "processed"
        model_dir = tmp_path / "models"
        processed_dir.mkdir()

        rng = np.random.default_rng(42)
        n = 60
        X = rng.standard_normal((n, DISTANCE_FEATURE_COUNT))
        y = rng.uniform(3.0, 8.0, n)

        # Save for pair 1-2 only
        np.save(processed_dir / "1-2_X.npy", X)
        np.save(processed_dir / "1-2_y.npy", y)

        # Need scaler for deployment copy
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X[:, :121])
        joblib.dump(scaler, processed_dir / "distance_scaler.pkl")
        with open(processed_dir / "distance_feature_names.txt", "w") as f:
            f.write("\n".join([f"feat_{i}" for i in range(DISTANCE_FEATURE_COUNT)]))

        # Need groups for GroupKFold
        groups = np.array(["s1"] * 30 + ["s2"] * 30)
        np.save(processed_dir / "1-2_groups.npy", groups)

        run(str(processed_dir), str(model_dir))

        assert (model_dir / "1-2_model.pkl").exists()
        assert (model_dir / "distance_scaler.pkl").exists()
        assert (model_dir / "distance_feature_names.txt").exists()
