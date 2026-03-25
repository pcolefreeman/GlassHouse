"""eda_utils.py — EDA helper functions for GHV4 data analysis.

All heavy logic used by eda.ipynb. Designed to handle empty DataFrames
gracefully (data may not exist yet when building the notebook).
"""
from typing import Optional
import os
import re
import math

import numpy as np
import pandas as pd

from ghv5.config import (
    META_COLS,
    EXPECTED_COLS,
    BUCKET_MS,
    ACTIVE_SHOUTER_IDS,
)

SHOUTER_IDS = ACTIVE_SHOUTER_IDS  # backwards-compat alias used throughout file


# ── Dimension parsing ──────────────────────────────────────────────────────────

def parse_dimensions(filename):
    # type: (str) -> tuple  # (float, float) or (None, None)
    r"""Extract (width_m, depth_m) from filename via regex r'(\d+\.?\d*)x(\d+\.?\d*)m'.

    Uses basename only so full paths work correctly.
    Returns (None, None) if pattern not found.
    """
    basename = os.path.basename(filename)
    m = re.search(r'(\d+\.?\d*)x(\d+\.?\d*)m', basename)
    if m:
        return float(m.group(1)), float(m.group(2))
    return (None, None)


# ── Column grouping ────────────────────────────────────────────────────────────

def group_columns(df):
    # type: (pd.DataFrame) -> dict
    """Group DataFrame columns by shouter and direction.

    Returns dict with keys: "meta", "s1", "s1_tx", "s2", "s2_tx",
    "s3", "s3_tx", "s4", "s4_tx".

    Uses negative lookahead r'^s{id}_(?!tx_)' to separate listener-rx columns
    (prefix s{id}_) from shouter-tx columns (prefix s{id}_tx_).
    """
    groups = {"meta": [c for c in META_COLS if c in df.columns]}
    for sid in SHOUTER_IDS:
        # listener-rx: starts with s{id}_ but NOT s{id}_tx_
        rx_pat = re.compile(r'^s' + str(sid) + r'_(?!tx_)')
        groups[f"s{sid}"] = [c for c in df.columns if rx_pat.match(c)]
        # shouter-tx: starts with s{id}_tx_
        groups[f"s{sid}_tx"] = [c for c in df.columns if c.startswith(f"s{sid}_tx_")]
    return groups


# ── Label parser ───────────────────────────────────────────────────────────────

def parse_label(label, n_cells=9, row_context=""):
    # type: (str, int, str) -> np.ndarray
    """Decode a GHV2 label string into a (n_cells,) binary target vector.

    Cell index = grid_row * 3 + grid_col.

    Valid formats:
        "empty"             → all zeros
        "r0c1"              → cell 1 = 1
        "r0c0+r2c2"         → cells 0 and 8 = 1
        "r0c0+r1c1+r2c2"    → three cells = 1

    Labels are CASE-SENSITIVE. Unrecognised formats print a WARNING and
    return all-zeros. row_context is included in the warning for debugging.
    """
    target = np.zeros(n_cells, dtype=int)
    if label == "empty":
        return target

    parts = label.split("+")
    has_error = False
    for part in parts:
        m = re.fullmatch(r'r([0-2])c([0-2])', part)
        if not m:
            has_error = True
            continue
        row = int(m.group(1))
        col = int(m.group(2))
        idx = row * 3 + col
        if idx < n_cells:
            target[idx] = 1
    if has_error:
        print(
            f"WARNING: Unrecognised part in label '{label}' at {row_context}"
            f" — invalid parts ignored, valid parts retained"
        )
    return target


# ── CSV loading & validation ───────────────────────────────────────────────────

def load_csv(path, manual_dims=None):
    # type: (str, Optional[tuple]) -> tuple  # (pd.DataFrame, tuple)
    """Load and validate a GHV2 CSV file.

    Validation:
    - Raises FileNotFoundError if path does not exist.
    - Raises ValueError if any of the 5 required meta columns are missing.
    - Prints a WARNING (does not raise) if the DataFrame has zero data rows.
    - Prints a WARNING (does not raise) if column count != EXPECTED_COLS.

    Dimension resolution order:
    1. manual_dims if not None
    2. parse_dimensions(basename(path))
    3. (None, None)

    Returns (df, area_dims).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    missing_meta = [c for c in META_COLS if c not in df.columns]
    if missing_meta:
        raise ValueError(
            f"CSV missing required meta columns: {missing_meta}"
        )

    if len(df) == 0:
        print(f"WARNING: CSV has no data rows (header only): {path}")

    if len(df.columns) != EXPECTED_COLS:
        diff = len(df.columns) - EXPECTED_COLS
        print(
            f"WARNING: Expected {EXPECTED_COLS} columns, "
            f"got {len(df.columns)} (diff={diff:+d})"
        )

    # Resolve dimensions
    if manual_dims is not None:
        area_dims = manual_dims
    else:
        area_dims = parse_dimensions(path)

    return df, area_dims


# ── Statistical helpers ────────────────────────────────────────────────────────

def describe_dataset(df, groups):
    # type: (pd.DataFrame, dict) -> dict
    """Return shape and per-group missing-value percentages."""
    result = {"shape": df.shape}
    for name, cols in groups.items():
        if not cols:
            continue
        sub = df[cols]
        total_vals = max(sub.size, 1)
        missing_pct = float(sub.isna().sum().sum()) / total_vals * 100.0
        result[name] = {
            "n_cols":      len(cols),
            "missing_pct": round(missing_pct, 2),
        }
    return result


def outlier_summary(df, groups):
    # type: (pd.DataFrame, dict) -> dict
    """IQR-based outlier count per shouter group (meta group skipped)."""
    result = {}
    for name, cols in groups.items():
        if name == "meta" or not cols:
            continue
        numeric_cols = df[cols].select_dtypes(include=[float, int]).columns.tolist()
        if not numeric_cols:
            continue
        sub = df[numeric_cols]
        q   = sub.quantile([0.25, 0.75])
        q1, q3 = q.loc[0.25], q.loc[0.75]
        iqr = q3 - q1
        is_outlier = (sub < (q1 - 1.5 * iqr)) | (sub > (q3 + 1.5 * iqr))
        total_outliers = int(is_outlier.sum().sum())
        total_values   = int(is_outlier.count().sum())
        result[name] = {
            "n_outliers":   total_outliers,
            "outlier_pct":  round(100.0 * total_outliers / max(total_values, 1), 2),
        }
    return result


# ── Temporal analysis ──────────────────────────────────────────────────────────

def temporal_stats(df):
    # type: (pd.DataFrame) -> dict
    """Sampling rate and gap detection.

    Sorts by timestamp_ms first (CSVWriter uses set iteration — order not guaranteed).
    A gap is any interval > 2 × BUCKET_MS (> 400 ms = at least 1 missed bucket).
    Returns dict with sampling_rate_hz, mean/std interval, n_gaps, gap_list.
    """
    if len(df) < 2:
        return {
            "sampling_rate_hz": None,
            "mean_interval_ms": None,
            "std_interval_ms":  None,
            "n_gaps":           0,
            "gap_list":         [],
        }

    ts     = df["timestamp_ms"].sort_values().values.astype(float)
    diffs  = np.diff(ts)

    mean_diff = float(np.mean(diffs))
    std_diff  = float(np.std(diffs))
    rate      = 1000.0 / mean_diff if mean_diff > 0 else 0.0

    gap_thresh = 2 * BUCKET_MS  # > 400 ms
    gap_list   = [
        (int(ts[i + 1]), int(diffs[i]))
        for i, d in enumerate(diffs) if d > gap_thresh
    ]

    return {
        "sampling_rate_hz": round(rate, 2),
        "mean_interval_ms": round(mean_diff, 1),
        "std_interval_ms":  round(std_diff, 1),
        "n_gaps":           len(gap_list),
        "gap_list":         gap_list,   # [(timestamp_ms, gap_duration_ms), ...]
    }


# ── Spatial analysis ───────────────────────────────────────────────────────────

def per_cell_stats(df):
    # type: (pd.DataFrame) -> pd.DataFrame
    """Count and mean RSSI per (grid_row, grid_col).

    RSSI = mean of s1_rssi, s2_rssi, s3_rssi, s4_rssi (listener-rx only),
    averaged across all present shouters per row. NaN if no RSSI columns present.
    """
    rssi_cols     = [f"s{sid}_rssi" for sid in SHOUTER_IDS]
    present_rssi  = [c for c in rssi_cols if c in df.columns]

    work = df.copy()
    if present_rssi:
        work["_mean_rssi"] = work[present_rssi].mean(axis=1)
    else:
        work["_mean_rssi"] = float("nan")

    stats = (
        work.groupby(["grid_row", "grid_col"])
        .agg(count=("timestamp_ms", "count"), mean_rssi=("_mean_rssi", "mean"))
        .reset_index()
    )
    return stats


# ── Feature analysis ───────────────────────────────────────────────────────────

def correlation_matrix(df, group_cols):
    # type: (pd.DataFrame, list) -> pd.DataFrame
    """Correlation matrix for scalar columns (rssi, noise_floor) in a group."""
    scalar_cols = [
        c for c in group_cols
        if c.endswith("_rssi") or c.endswith("_noise_floor")
    ]
    if not scalar_cols:
        return pd.DataFrame()
    return df[scalar_cols].corr()


def phase_polar_data(df, group_cols):
    # type: (pd.DataFrame, list) -> np.ndarray
    """Flat array of valid phase values for polar histogram plotting."""
    phase_cols = [c for c in group_cols if "_phase_" in c]
    if not phase_cols:
        return np.array([])
    vals = df[phase_cols].values.flatten().astype(float)
    return vals[~np.isnan(vals)]


# ── Recommendations ────────────────────────────────────────────────────────────

def model_recommendation(df):
    # type: (pd.DataFrame) -> str
    """Data-driven model selection guidance.

    Returns a fixed fallback string when df is empty (no data yet).
    """
    if len(df) == 0:
        return (
            "No data available yet — recommendations will be generated "
            "once a CSV capture is loaded."
        )

    lines = []
    n_rows = len(df)
    n_cols = len(df.columns)

    if "label" in df.columns:
        label_counts = df["label"].value_counts()
        n_labels     = len(label_counts)
        lines.append(
            f"Dataset: {n_rows} rows, {n_cols} columns, {n_labels} unique labels."
        )
        balance = label_counts.min() / max(label_counts.max(), 1)
        if balance < 0.5:
            lines.append(
                "WARNING: Class imbalance detected. "
                "Consider oversampling minority classes (e.g. SMOTE)."
            )

    rssi_cols = [c for c in df.columns if c.endswith("_rssi")]
    if rssi_cols:
        missing_pct = df[rssi_cols].isna().mean().mean() * 100
        if missing_pct > 20:
            lines.append(
                f"WARNING: {missing_pct:.1f}% missing RSSI values — "
                "check for shouter MISS frames."
            )

    lines += [
        "",
        "Recommended model pipeline:",
        "1. PCA per shouter group (s1, s1_tx, s2, ...) — reduces 5,128 features "
           "to manageable dimensionality; per-group preserves spatial meaning.",
        "2. Per-cell binary classifier: Random Forest (sklearn) — "
           "robust baseline, handles high-dim CSI, built-in feature importance.",
        "3. Comparison: SVM (RBF kernel) — strong precedent in WiFi CSI "
           "fingerprinting literature.",
        "4. Deploy: sklearn pipeline → joblib .pkl — "
           "already supported by InferenceV2.load_model().",
        "",
        "Label encoding: use eda_utils.parse_label() to produce "
           "(9,) binary target vectors for 9 independent binary classifiers.",
    ]
    return "\n".join(lines)


def labeling_recommendation():
    # type: () -> str
    """Multi-person labeling strategy for GHV2 training data collection."""
    return (
        "LABELING STRATEGY FOR MULTI-PERSON GHV2 DATA COLLECTION\n"
        "=========================================================\n\n"
        "1. EMPTY PASS\n"
        "   Command: python GlassHouseV2.py --label empty --row 0 --col 0 "
        "[--width W --depth D]\n"
        "   Purpose: Negative baseline for all 9 cell classifiers.\n\n"
        "2. SINGLE-PERSON PASSES  (9 sessions, one per cell)\n"
        "   Command: python GlassHouseV2.py --label r{row}c{col} --row {row} --col {col}\n"
        "   Example: --label r0c0 --row 0 --col 0\n"
        "   Purpose: Positive training data for cell (row, col) classifier.\n\n"
        "3. MULTI-PERSON PASSES\n"
        '   Command: python GlassHouseV2.py --label "r{row_a}c{col_a}+r{row_b}c{col_b}"\n'
        '   Example: --label "r0c0+r2c2"\n'
        "   Purpose: Simultaneous occupancy in two cells.\n"
        "   Extend with +r{row}c{col} for 3+ people.\n\n"
        "4. LABEL DECODING\n"
        "   parse_label('r0c0+r2c2') → [1,0,0, 0,0,0, 0,0,1]  # (9,) binary vector\n"
        "   Each position = grid_row * 3 + grid_col\n\n"
        "NOTES:\n"
        "- Labels are CASE-SENSITIVE (r0c1 ✓, r0C1 ✗)\n"
        "- No validation in GlassHouseV2.py — double-check before collecting\n"
        "- Minimum recommended: 1 empty + 9 single-person = 10 sessions"
    )
