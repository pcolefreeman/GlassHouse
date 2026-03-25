"""Training pipeline for per-pair ML distance regressors.

Trains 6 independent GBT/RF models (one per shouter pair) using
GroupKFold cross-validation. PC only.
"""
from __future__ import annotations

import logging
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, cross_val_predict

from ghv4.config import DISTANCE_MAX_TREES, PAIR_KEYS, DISTANCE_FEATURE_COUNT, DISTANCE_CV_FOLDS

_log = logging.getLogger(__name__)


def train_pair_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = DISTANCE_CV_FOLDS,
) -> Tuple[Any, Dict[str, float]]:
    """Train a regressor for one shouter pair.

    Tries GBT and RF, selects by lowest MAE on GroupKFold CV.

    Returns:
        (best_model, metrics_dict) where metrics_dict has
        'mae', 'rmse', 'within_1m' keys.
    """
    candidates = {
        "gbt": GradientBoostingRegressor(
            n_estimators=min(100, DISTANCE_MAX_TREES),
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        ),
        "rf": RandomForestRegressor(
            n_estimators=min(100, DISTANCE_MAX_TREES),
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
    }

    unique_groups = np.unique(groups)
    actual_splits = min(n_splits, len(unique_groups))
    if actual_splits < 2:
        # Not enough groups for CV — train on all data
        _log.warning("Only %d group(s), skipping CV", len(unique_groups))
        best = candidates["gbt"]
        best.fit(X, y)
        preds = best.predict(X)
        return best, _compute_metrics(y, preds)

    gkf = GroupKFold(n_splits=actual_splits)
    best_name, best_mae = None, float("inf")
    best_preds = None

    for name, model in candidates.items():
        preds = cross_val_predict(model, X, y, groups=groups, cv=gkf)
        mae = mean_absolute_error(y, preds)
        _log.info("  %s CV MAE: %.3f m", name, mae)
        if mae < best_mae:
            best_name, best_mae, best_preds = name, mae, preds

    # Refit best on full data
    best_model = candidates[best_name]
    best_model.fit(X, y)
    metrics = _compute_metrics(y, best_preds)
    _log.info("  Best: %s (MAE=%.3f, RMSE=%.3f, within_1m=%.1f%%)",
              best_name, metrics["mae"], metrics["rmse"],
              metrics["within_1m"] * 100)
    return best_model, metrics


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    within_1m = float(np.mean(np.abs(y_true - y_pred) <= 1.0))
    return {"mae": mae, "rmse": rmse, "within_1m": within_1m}


def geometric_consistency_check(
    distances: Dict[str, float],
) -> Dict[str, float]:
    """Check diagonal consistency: diag vs sqrt(side1^2 + side2^2).

    Returns dict of errors (predicted_diag - expected_diag) for each diagonal.
    """
    errors = {}
    # Diagonal 1-3: sides 1-2 (depth) and 2-3 (width)
    if all(k in distances for k in ("1-2", "2-3", "1-3")):
        expected = math.sqrt(distances["1-2"]**2 + distances["2-3"]**2)
        errors["diag_1-3"] = distances["1-3"] - expected

    # Diagonal 2-4: sides 1-2 (depth) and 1-4 (width)
    if all(k in distances for k in ("1-4", "3-4", "2-4")):
        expected = math.sqrt(distances["1-4"]**2 + distances["3-4"]**2)
        errors["diag_2-4"] = distances["2-4"] - expected

    return errors


def run(processed_dir: str, model_dir: str) -> None:
    """Train all available pair models from preprocessed data.

    Args:
        processed_dir: Contains {pair}_X.npy, {pair}_y.npy, {pair}_groups.npy
        model_dir: Output directory for .pkl models + scaler copy
    """
    os.makedirs(model_dir, exist_ok=True)
    processed = Path(processed_dir)

    trained = {}
    for pair_id in PAIR_KEYS:
        x_path = processed / f"{pair_id}_X.npy"
        y_path = processed / f"{pair_id}_y.npy"
        g_path = processed / f"{pair_id}_groups.npy"

        if not x_path.exists() or not y_path.exists():
            _log.info("Pair %s: no data, skipping", pair_id)
            continue

        X = np.load(x_path)
        y = np.load(y_path)
        groups = np.load(g_path, allow_pickle=True) if g_path.exists() else np.zeros(len(y))

        _log.info("Training pair %s (%d samples)...", pair_id, len(y))
        model, metrics = train_pair_model(X, y, groups)
        trained[pair_id] = metrics

        model_path = os.path.join(model_dir, f"{pair_id}_model.pkl")
        joblib.dump(model, model_path)
        _log.info("Pair %s: saved → %s", pair_id, model_path)

    # Copy scaler + feature names to model dir for deployment
    for fname in ("distance_scaler.pkl", "distance_feature_names.txt"):
        src = processed / fname
        if src.exists():
            shutil.copy2(src, os.path.join(model_dir, fname))

    # Geometric consistency if we have enough pairs
    if len(trained) == 6:
        _log.info("All 6 pairs trained. Run inference for geometric check.")

    _log.info("Training complete. %d pair model(s) saved to %s",
              len(trained), model_dir)
