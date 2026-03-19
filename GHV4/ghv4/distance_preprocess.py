"""Preprocessing pipeline for ML distance estimation.

Loads raw distance CSVs, matches forward/reverse snapshot pairs,
extracts features, scales, and outputs per-pair numpy arrays.

PC only — not deployed to Pi.
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ghv4.config import (
    DISTANCE_FEATURE_COUNT,
    PAIR_KEYS,
    NULL_SUBCARRIER_INDICES,
    SUBCARRIERS,
)
from ghv4.distance_features import FEATURE_NAMES

_log = logging.getLogger(__name__)


def derive_distances(width_m: float, depth_m: float) -> Dict[str, float]:
    """Compute all 6 pairwise distances from rectangle dimensions.

    Layout (operator facing rubble):
        S2----S3        S1-S4 = width (bottom)
        |    |          S2-S3 = width (top)
        S1----S4        S1-S2 = depth (left)
                        S3-S4 = depth (right)
    """
    diag = math.sqrt(width_m**2 + depth_m**2)
    return {
        "1-2": depth_m,
        "1-3": diag,
        "1-4": width_m,
        "2-3": width_m,
        "2-4": diag,
        "3-4": depth_m,
    }


def match_paired_samples(
    df: pd.DataFrame,
) -> List[Tuple[pd.Series, pd.Series]]:
    """Match forward/reverse snapshot rows by (reporter_id, peer_id, snap_seq).

    For a pair (i, j) where i < j:
      forward  = reporter_id=i, peer_id=j
      reverse  = reporter_id=j, peer_id=i

    Returns list of (forward_row, reverse_row) tuples.
    """
    pairs = []
    grouped = df.groupby("snap_seq")
    for seq, group in grouped:
        if len(group) < 2:
            continue
        # Find forward and reverse within this seq
        for _, row_a in group.iterrows():
            rid_a, pid_a = int(row_a["reporter_id"]), int(row_a["peer_id"])
            lo, hi = min(rid_a, pid_a), max(rid_a, pid_a)
            # Find the reverse
            mask = (group["reporter_id"] == pid_a) & (group["peer_id"] == rid_a)
            rev_rows = group[mask]
            if rev_rows.empty:
                continue
            rev_row = rev_rows.iloc[0]
            if rid_a < pid_a:
                pairs.append((row_a, rev_row))
            # Only add once per direction pair per seq
            break
    return pairs


def build_dataset(
    raw_dir: str, pair_id: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all CSVs for a given pair, return X, y, groups.

    Args:
        raw_dir: Directory containing raw CSVs.
        pair_id: e.g. "1-2"

    Returns:
        X: (N, 484) feature matrix
        y: (N,) ground-truth distances in meters
        groups: (N,) session IDs for GroupKFold
    """
    csvs = sorted(Path(raw_dir).glob("*.csv"))
    all_X, all_y, all_groups = [], [], []

    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        if "pair_id" not in df.columns:
            continue
        sub = df[df["pair_id"] == pair_id]
        if sub.empty:
            continue

        feat_cols = [c for c in sub.columns if c.startswith("feat_")]
        if len(feat_cols) != DISTANCE_FEATURE_COUNT:
            _log.warning(
                "%s: expected %d feat cols, got %d — skipping",
                csv_path.name, DISTANCE_FEATURE_COUNT, len(feat_cols),
            )
            continue

        X_block = sub[feat_cols].values.astype(np.float64)
        y_block = sub["distance_m"].values.astype(np.float64)
        groups_block = sub["session_id"].values

        all_X.append(X_block)
        all_y.append(y_block)
        all_groups.append(groups_block)

    if not all_X:
        return np.empty((0, DISTANCE_FEATURE_COUNT)), np.empty(0), np.empty(0)

    return (
        np.vstack(all_X),
        np.concatenate(all_y),
        np.concatenate(all_groups),
    )


def run(raw_dir: str, out_dir: str) -> None:
    """Full preprocessing pipeline: raw CSVs → per-pair X.npy, y.npy, scaler.

    Args:
        raw_dir: Path to directory with raw distance CSV files.
        out_dir: Path to output directory for processed artifacts.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Fit shared scaler on ALL pairs' data
    all_X_blocks = []
    pair_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for pair_id in PAIR_KEYS:
        X, y, groups = build_dataset(raw_dir, pair_id)
        if X.shape[0] == 0:
            _log.info("Pair %s: no data found, skipping", pair_id)
            continue
        pair_data[pair_id] = (X, y, groups)
        all_X_blocks.append(X)
        _log.info("Pair %s: %d samples loaded", pair_id, X.shape[0])

    if not all_X_blocks:
        _log.warning("No data found in %s", raw_dir)
        return

    # Fit scaler on amp_norm columns only (first 121 of each 242 direction block)
    all_X = np.vstack(all_X_blocks)
    scaler = StandardScaler()

    # amp_norm indices: [0:121] (fwd) and [242:363] (rev)
    amp_indices = list(range(121)) + list(range(242, 363))
    scaler.fit(all_X[:, amp_indices])

    # Apply scaler and save per-pair
    for pair_id, (X, y, groups) in pair_data.items():
        X_scaled = X.copy()
        X_scaled[:, amp_indices] = scaler.transform(X[:, amp_indices])
        # Phase columns already in [-1, 1] from extract_snap_features
        np.nan_to_num(X_scaled, copy=False)

        np.save(os.path.join(out_dir, f"{pair_id}_X.npy"), X_scaled)
        np.save(os.path.join(out_dir, f"{pair_id}_y.npy"), y)
        np.save(os.path.join(out_dir, f"{pair_id}_groups.npy"), groups)
        _log.info("Pair %s: saved %d samples", pair_id, X.shape[0])

    # Save shared artifacts
    joblib.dump(scaler, os.path.join(out_dir, "distance_scaler.pkl"))
    with open(os.path.join(out_dir, "distance_feature_names.txt"), "w") as f:
        f.write("\n".join(FEATURE_NAMES))

    _log.info("Preprocessing complete → %s", out_dir)
