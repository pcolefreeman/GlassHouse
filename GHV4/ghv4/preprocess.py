"""preprocess.py — GHV2 raw CSV → training-ready arrays.

Applies the pipeline from docs/preprocessing.md:
  1. Load all CSVs from RAW_DIR (skips empty files)
  2. Drop: null subcarriers, snr (collinear), raw amp, noise_floor
  3. Scale: amp_norm → StandardScaler, phase/pdiff → /π, rssi → StandardScaler
  4. NaN fill with 0 (post-scale; Random Forest handles it natively anyway)
  5. Extract y as (N, 9) binary matrix via eda_utils.parse_label()
  6. Save X.npy, y.npy, feature_names.txt, scaler.pkl to OUT_DIR

Usage:
    python preprocess.py                        # uses defaults below
    python preprocess.py --raw-dir path/to/raw --out-dir path/to/out
"""
import argparse
import glob
import json
import math
import os

import joblib
import numpy as np
import pandas as pd

from ghv4 import eda_utils
from ghv4.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    NULL_SUBCARRIER_INDICES,
    NULL_PDIFF_INDICES,
    META_COLS,
    PAIR_KEYS,
)

RAW_DIR = str(DATA_RAW_DIR)
OUT_DIR = str(DATA_PROCESSED_DIR)
NULL_SUBS = NULL_SUBCARRIER_INDICES
NULL_PDIFF = NULL_PDIFF_INDICES


def _load_spacing(raw_dir: str) -> dict:
    """Load spacing.json from raw_dir. Returns zeros for missing/absent pairs."""
    path = os.path.join(raw_dir, "spacing.json")
    zeros = {k: 0.0 for k in PAIR_KEYS}
    try:
        with open(path) as f:
            data = json.load(f)
        pairs = data.get("pairs", {})
        return {k: float(pairs.get(k, {}).get("distance_m", 0.0))
                for k in PAIR_KEYS}
    except FileNotFoundError:
        print(f"  [SPACING] spacing.json not found in {raw_dir} — using zeros")
        return zeros


def _build_drop_set(df: pd.DataFrame) -> set:
    """Return set of column names to drop per preprocessing.md."""
    drop = set()
    for col in df.columns:
        if col in META_COLS:
            continue
        # noise_floor — always constant
        if col.endswith("_noise_floor"):
            drop.add(col)
            continue
        # parse suffix: e.g. s1_amp_3  /  s1_tx_amp_norm_32
        # feature type is the segment before the final integer
        parts = col.rsplit("_", 1)
        if len(parts) != 2:
            continue
        prefix, idx_str = parts
        try:
            idx = int(idx_str)
        except ValueError:
            continue

        # raw amp (not amp_norm)
        if prefix.endswith("_amp"):
            drop.add(col)
            continue
        # snr — collinear with amp_norm
        if prefix.endswith("_snr"):
            drop.add(col)
            continue
        # null subcarrier positions for amp_norm / phase
        if prefix.endswith("_amp_norm") and idx in NULL_SUBS:
            drop.add(col)
            continue
        if prefix.endswith("_phase") and idx in NULL_SUBS:
            drop.add(col)
            continue
        # null pdiff positions
        if prefix.endswith("_pdiff") and idx in NULL_PDIFF:
            drop.add(col)
            continue

    return drop


def _split_feature_cols(feat_cols):
    """Return three lists: amp_norm_cols, phase_pdiff_cols, rssi_cols."""
    amp_norm, phase_pdiff, rssi = [], [], []
    for c in feat_cols:
        if "_amp_norm_" in c:
            amp_norm.append(c)
        elif "_phase_" in c or "_pdiff_" in c:
            phase_pdiff.append(c)
        elif c.endswith("_rssi"):
            rssi.append(c)
    return amp_norm, phase_pdiff, rssi


def run(raw_dir: str, out_dir: str):
    # ── 1. Load ───────────────────────────────────────────────────────────────
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if len(df) == 0:
            print(f"  [SKIP] {os.path.basename(f)} — empty")
            continue
        frames.append(df)
        print(f"  [LOAD] {os.path.basename(f)} — {len(df)} rows  label={df['label'].unique()}")

    if not frames:
        print("No data loaded. Exiting.")
        return

    data = pd.concat(frames, ignore_index=True)
    print(f"\nCombined: {data.shape[0]} rows x {data.shape[1]} cols")

    # ── 2. Drop ───────────────────────────────────────────────────────────────
    drop_cols = _build_drop_set(data)
    data = data.drop(columns=list(drop_cols))
    feat_cols = [c for c in data.columns if c not in META_COLS]
    print(f"After drop: {len(feat_cols)} feature columns  ({len(drop_cols)} removed)")

    # ── 3. Scale ──────────────────────────────────────────────────────────────
    X = data[feat_cols].values.astype(np.float32)
    amp_norm_cols, phase_pdiff_cols, rssi_cols = _split_feature_cols(feat_cols)

    col_index = {c: i for i, c in enumerate(feat_cols)}

    # amp_norm → StandardScaler
    amp_idx  = [col_index[c] for c in amp_norm_cols]
    rssi_idx = [col_index[c] for c in rssi_cols]
    pp_idx   = [col_index[c] for c in phase_pdiff_cols]

    # Fit column-wise using nanmean/nanstd (each row is NaN for 3/4 shouters)
    def _nanscale(X_full, idx):
        """Standardise columns at idx in-place; returns (mean, std) arrays."""
        sub = X_full[:, idx]
        mean = np.nanmean(sub, axis=0)
        std  = np.nanstd(sub,  axis=0)
        std[std == 0] = 1.0          # avoid divide-by-zero on constant cols
        X_full[:, idx] = (sub - mean) / std
        return mean, std

    amp_mean, amp_std   = (np.array([]), np.array([]))
    rssi_mean, rssi_std = (np.array([]), np.array([]))

    if amp_idx:
        amp_mean, amp_std = _nanscale(X, amp_idx)
    if rssi_idx:
        rssi_mean, rssi_std = _nanscale(X, rssi_idx)

    if pp_idx:
        X[:, pp_idx] = X[:, pp_idx] / math.pi   # /π → (−1, 1]

    # ── 4. NaN fill ───────────────────────────────────────────────────────────
    nan_count = np.isnan(X).sum()
    X = np.nan_to_num(X, nan=0.0)
    print(f"NaN fill: {nan_count} values set to 0")

    # ── 5. Labels ─────────────────────────────────────────────────────────────
    y = np.stack(data["label"].apply(eda_utils.parse_label).values).astype(np.int8)
    label_counts = data["label"].value_counts().sort_index()
    print(f"\nLabel distribution:\n{label_counts.to_string()}")
    print(f"\nX shape: {X.shape}   y shape: {y.shape}")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)

    with open(os.path.join(out_dir, "feature_names.txt"), "w") as fh:
        fh.write("\n".join(feat_cols))

    joblib.dump({
        "amp_cols":  amp_norm_cols, "amp_mean":  amp_mean,  "amp_std":  amp_std,
        "rssi_cols": rssi_cols,     "rssi_mean": rssi_mean, "rssi_std": rssi_std,
    }, os.path.join(out_dir, "scaler.pkl"))

    print(f"\nSaved to {out_dir}:")
    print(f"  X.npy          {X.shape}")
    print(f"  y.npy          {y.shape}")
    print(f"  feature_names.txt  ({len(feat_cols)} names)")
    print(f"  scaler.pkl")

    # ── 6b. Spacing features (unscaled) ──────────────────────────────────────
    spacing = _load_spacing(raw_dir)
    spacing_vals = [spacing[k] for k in PAIR_KEYS]
    spacing_names = ["dist_s1_s2", "dist_s1_s3", "dist_s1_s4",
                     "dist_s2_s3", "dist_s2_s4", "dist_s3_s4"]
    spacing_block = np.tile(np.array(spacing_vals, dtype=np.float32), (X.shape[0], 1))
    X = np.hstack([X, spacing_block])
    feat_cols = feat_cols + spacing_names
    print(f"Spacing features appended: {spacing_names}")
    print(f"  X shape after spacing: {X.shape}")

    # Re-save with spacing
    np.save(os.path.join(out_dir, "X.npy"), X)
    with open(os.path.join(out_dir, "feature_names.txt"), "w") as fh:
        fh.write("\n".join(feat_cols))
    print(f"  X.npy          {X.shape}  (updated with spacing)")
    print(f"  feature_names.txt  ({len(feat_cols)} names)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()
    run(args.raw_dir, args.out_dir)
