"""End-to-end test: synthetic data → preprocess → train → calibrate → spacing.json."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
import joblib
from sklearn.preprocessing import StandardScaler

from ghv4.config import (
    PAIR_KEYS,
    CALIBRATION_MIN_PAIRS,
    DISTANCE_FEATURE_COUNT,
)
from ghv4.distance_preprocess import derive_distances, run as preprocess_run
from ghv4.distance_train import run as train_run
from ghv4.distance_inference import DistanceCalibrator
from ghv4.distance_features import FEATURE_NAMES


def _generate_synthetic_csvs(raw_dir, width_m=3.0, depth_m=4.0, n_per_pair=40):
    """Generate synthetic distance CSV data for all 6 pairs."""
    import pandas as pd

    rng = np.random.default_rng(42)
    dists = derive_distances(width_m, depth_m)
    feat_cols = [f"feat_{i}" for i in range(DISTANCE_FEATURE_COUNT)]

    for pair_id in PAIR_KEYS:
        data = {col: rng.standard_normal(n_per_pair) for col in feat_cols}
        data["pair_id"] = [pair_id] * n_per_pair
        data["distance_m"] = [dists[pair_id]] * n_per_pair
        data["session_id"] = (
            ["sess_01"] * (n_per_pair // 2) + ["sess_02"] * (n_per_pair // 2)
        )
        data["timestamp"] = list(range(n_per_pair))
        data["width_m"] = [width_m] * n_per_pair
        data["depth_m"] = [depth_m] * n_per_pair
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(raw_dir, f"pair_{pair_id.replace('-','_')}.csv"), index=False)


def test_full_pipeline(tmp_path):
    raw_dir = str(tmp_path / "raw")
    processed_dir = str(tmp_path / "processed")
    model_dir = str(tmp_path / "models")

    os.makedirs(raw_dir)

    # Step 1: Generate synthetic data
    _generate_synthetic_csvs(raw_dir, width_m=3.0, depth_m=4.0)

    # Step 2: Preprocess
    preprocess_run(raw_dir, processed_dir)

    for pair_id in PAIR_KEYS:
        assert os.path.exists(os.path.join(processed_dir, f"{pair_id}_X.npy"))

    # Step 3: Train (groups.npy saved by preprocessing)
    train_run(processed_dir, model_dir)

    for pair_id in PAIR_KEYS:
        assert os.path.exists(os.path.join(model_dir, f"{pair_id}_model.pkl"))

    # Step 4: Calibration inference
    cal = DistanceCalibrator(model_dir)

    rng = np.random.default_rng(99)
    for pair_id in PAIR_KEYS:
        i, j = (int(x) for x in pair_id.split("-"))
        for seq in range(CALIBRATION_MIN_PAIRS + 5):
            csi = bytes(rng.integers(-127, 127, 256, dtype=np.int8))
            cal.feed_snap(i, j, seq, csi)
            cal.feed_snap(j, i, seq, csi)

    distances = cal.predict_distances()
    assert len(distances) == 6

    # Step 5: Write spacing.json
    spacing_path = str(tmp_path / "spacing.json")
    cal.write_spacing(spacing_path, distances)

    data = json.loads(Path(spacing_path).read_text())
    assert data["version"] == 2
    assert len(data["pairs"]) == 6
    for pair_id in PAIR_KEYS:
        assert data["pairs"][pair_id]["source"] == "ml"
        assert data["pairs"][pair_id]["distance_m"] > 0
