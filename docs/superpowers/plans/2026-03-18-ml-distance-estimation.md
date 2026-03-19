# ML-Based Distance Estimation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace MUSIC as the primary distance source during deployment with per-pair ML regressors trained on CSI snapshots, running calibration at session start on Pi 4B.

**Architecture:** 6 independent scikit-learn regressors (one per shouter pair) trained on PC from bidirectional CSI snapshot features (amp_norm + phase, 484 features per paired sample). During deployment, a 30-second calibration phase collects [0xEE][0xFF] snap frames, predicts per-pair distances via median aggregation, writes spacing.json, then hands off to the zone classifier.

**Tech Stack:** Python 3, scikit-learn (GradientBoostingRegressor / RandomForestRegressor, max 200 trees), numpy, joblib, pandas (preprocessing only)

**Spec:** `docs/superpowers/specs/2026-03-17-ml-distance-estimation.md` (provided inline in session)

---

## File Structure

### New Files

| File | Location | Runs On | Responsibility |
|------|----------|---------|----------------|
| `ghv4/distance_features.py` | `GHV4/ghv4/` | PC + Pi | Parse snap CSI bytes → 484-feature vector (shared by preprocess, train, inference) |
| `ghv4/distance_preprocess.py` | `GHV4/ghv4/` | PC | Load raw CSVs, match fwd/rev pairs, scale, output X.npy/y.npy per pair |
| `ghv4/distance_train.py` | `GHV4/ghv4/` | PC | Train 6 per-pair regressors with CV, save models + scaler |
| `ghv4/distance_inference.py` | `GHV4/ghv4/` | Pi + PC | Calibration-phase: buffer snaps, predict distances, write spacing.json |
| `run_distance_preprocess.py` | `GHV4/` | PC | CLI entry point for preprocessing |
| `run_distance_train.py` | `GHV4/` | PC | CLI entry point for training |
| `tests/test_distance_features.py` | `GHV4/tests/` | PC | Tests for feature extraction |
| `tests/test_distance_preprocess.py` | `GHV4/tests/` | PC | Tests for preprocessing pipeline |
| `tests/test_distance_train.py` | `GHV4/tests/` | PC | Tests for training pipeline |
| `tests/test_distance_inference.py` | `GHV4/tests/` | PC | Tests for calibration inference |

### Modified Files

| File | Changes |
|------|---------|
| `ghv4/config.py` | Add distance-related constants (calibration window, model dir, max trees, feature list) |
| `ghv4/serial_io.py` | Add optional `snap_callback` parameter to SerialReader for routing [0xEE][0xFF] frames |
| `ghv4/ui/capture_tab.py` | Add width_m/depth_m inputs, "Collect Distance Training Data" mode |
| `ghv4/inference.py` | Add calibration step before main inference loop |

---

## Task 1: Add Distance Constants to config.py

**Files:**
- Modify: `GHV4/ghv4/config.py`
- Test: `GHV4/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# In tests/test_config.py — append these tests

def test_distance_constants_exist():
    from ghv4 import config
    assert hasattr(config, "CALIBRATION_WINDOW_S")
    assert config.CALIBRATION_WINDOW_S == 30
    assert hasattr(config, "CALIBRATION_EXTENSION_S")
    assert config.CALIBRATION_EXTENSION_S == 15
    assert hasattr(config, "CALIBRATION_MAX_EXTENSIONS")
    assert config.CALIBRATION_MAX_EXTENSIONS == 2
    assert hasattr(config, "CALIBRATION_MIN_PAIRS")
    assert config.CALIBRATION_MIN_PAIRS == 10
    assert hasattr(config, "DISTANCE_MODEL_DIR")
    assert config.DISTANCE_MODEL_DIR == "distance_models"
    assert hasattr(config, "DISTANCE_MAX_TREES")
    assert config.DISTANCE_MAX_TREES == 200

def test_distance_valid_subcarriers():
    from ghv4 import config
    assert hasattr(config, "VALID_SUBCARRIER_COUNT")
    assert config.VALID_SUBCARRIER_COUNT == 121  # 128 - 7 nulls

def test_distance_feature_count():
    from ghv4 import config
    # 121 amp_norm + 121 phase per direction, 2 directions
    assert hasattr(config, "DISTANCE_FEATURE_COUNT")
    assert config.DISTANCE_FEATURE_COUNT == 484
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4" && python -m pytest tests/test_config.py::test_distance_constants_exist tests/test_config.py::test_distance_valid_subcarriers tests/test_config.py::test_distance_feature_count -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Add constants to config.py**

Append to `ghv4/config.py`:

```python
# ---------------------------------------------------------------------------
# ML Distance Estimation
# ---------------------------------------------------------------------------
VALID_SUBCARRIER_COUNT = SUBCARRIERS - len(NULL_SUBCARRIER_INDICES)  # 121
DISTANCE_FEATURE_COUNT = VALID_SUBCARRIER_COUNT * 2 * 2             # 484

CALIBRATION_WINDOW_S   = 30       # seconds of snap collection
CALIBRATION_EXTENSION_S = 15      # extension if insufficient data
CALIBRATION_MAX_EXTENSIONS = 2    # max number of extensions
CALIBRATION_MIN_PAIRS  = 10       # min matched fwd/rev pairs per shouter pair

DISTANCE_MODEL_DIR     = "distance_models"
DISTANCE_MAX_TREES     = 200      # max estimators per GB/RF model (Pi memory budget)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4" && python -m pytest tests/test_config.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4"
git add ghv4/config.py tests/test_config.py
git commit -m "feat: add ML distance estimation constants to config.py"
```

---

## Task 2: Distance Feature Extraction Module

**Files:**
- Create: `GHV4/ghv4/distance_features.py`
- Create: `GHV4/tests/test_distance_features.py`

This is the shared core: parses snap-frame CSI bytes into a 484-element feature vector. Used by preprocessing (CSV → features), training (loading features), and inference (live snap → features).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_distance_features.py`:

```python
"""Tests for distance_features — snap CSI → ML feature vector."""
import numpy as np
import pytest
from ghv4.distance_features import (
    snap_csi_to_complex,
    extract_snap_features,
    pair_features,
    FEATURE_NAMES,
)
from ghv4.config import NULL_SUBCARRIER_INDICES, VALID_SUBCARRIER_COUNT


def _make_csi_bytes(n_subcarriers=128):
    """Generate synthetic int8 I/Q CSI bytes (imag, real per subcarrier)."""
    rng = np.random.default_rng(42)
    iq = rng.integers(-127, 127, size=n_subcarriers * 2, dtype=np.int8)
    return bytes(iq)


class TestSnapCsiToComplex:
    def test_returns_121_complex(self):
        csi = snap_csi_to_complex(_make_csi_bytes())
        assert csi.shape == (VALID_SUBCARRIER_COUNT,)
        assert np.iscomplexobj(csi)

    def test_null_subcarriers_excluded(self):
        # All-zero CSI → all zeros, but length still 121
        csi = snap_csi_to_complex(bytes(256))
        assert csi.shape == (VALID_SUBCARRIER_COUNT,)
        assert np.all(csi == 0)

    def test_rejects_short_buffer(self):
        assert snap_csi_to_complex(bytes(100)) is None

    def test_byte_order_imag_real(self):
        """ESP32 CSI: byte[0]=imag, byte[1]=real for subcarrier 0."""
        buf = bytes([5, 10] + [0] * 254)  # sub0: imag=5, real=10
        csi = snap_csi_to_complex(buf)
        # sub0 is in NULL set, so check sub3 (first valid after nulls)
        buf2 = bytes([0] * 6 + [7, 3] + [0] * 248)  # sub3: imag=7, real=3
        csi2 = snap_csi_to_complex(buf2)
        # sub3 is index 0 in valid array (subs 0,1,2 are null)
        assert csi2[0] == complex(3, 7)


class TestExtractSnapFeatures:
    def test_output_length(self):
        csi = snap_csi_to_complex(_make_csi_bytes())
        feats = extract_snap_features(csi)
        # 121 amp_norm + 121 phase = 242
        assert len(feats) == VALID_SUBCARRIER_COUNT * 2

    def test_amp_norm_range(self):
        csi = snap_csi_to_complex(_make_csi_bytes())
        feats = extract_snap_features(csi)
        amp_norm = feats[:VALID_SUBCARRIER_COUNT]
        assert all(0.0 <= v <= 1.0 for v in amp_norm)

    def test_phase_range(self):
        csi = snap_csi_to_complex(_make_csi_bytes())
        feats = extract_snap_features(csi)
        phase = feats[VALID_SUBCARRIER_COUNT:]
        # Scaled by pi → range [-1, 1]
        assert all(-1.0 <= v <= 1.0 for v in phase)


class TestPairFeatures:
    def test_output_length_484(self):
        fwd_csi = snap_csi_to_complex(_make_csi_bytes(128))
        rev_csi = snap_csi_to_complex(
            bytes(np.random.default_rng(99).integers(-127, 127, 256, dtype=np.int8))
        )
        vec = pair_features(fwd_csi, rev_csi)
        assert len(vec) == 484  # 242 fwd + 242 rev

    def test_feature_names_match_length(self):
        assert len(FEATURE_NAMES) == 484


class TestFeatureNames:
    def test_prefix_structure(self):
        # First 121 should be fwd_amp_norm_*, next 121 fwd_phase_*
        assert FEATURE_NAMES[0].startswith("fwd_amp_norm_")
        assert FEATURE_NAMES[121].startswith("fwd_phase_")
        assert FEATURE_NAMES[242].startswith("rev_amp_norm_")
        assert FEATURE_NAMES[363].startswith("rev_phase_")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4" && python -m pytest tests/test_distance_features.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement distance_features.py**

Create `ghv4/distance_features.py`:

```python
"""Snap-frame CSI → ML feature vector for distance estimation.

Parses [0xEE][0xFF] snap CSI bytes (int8 I/Q) into a 484-element feature
vector (121 amp_norm + 121 phase per direction, forward + reverse).

Used by: distance_preprocess.py, distance_train.py, distance_inference.py
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from ghv4.config import (
    NULL_SUBCARRIER_INDICES,
    SUBCARRIERS,
    VALID_SUBCARRIER_COUNT,
    DISTANCE_FEATURE_COUNT,
)

# Ordered indices of valid (non-null) subcarriers
_VALID_INDICES = sorted(set(range(SUBCARRIERS)) - NULL_SUBCARRIER_INDICES)

# Pre-built feature name list (484 entries)
_directions = ("fwd", "rev")
_feat_types = ("amp_norm", "phase")
FEATURE_NAMES: List[str] = []
for d in _directions:
    for ft in _feat_types:
        for idx in _VALID_INDICES:
            FEATURE_NAMES.append(f"{d}_{ft}_{idx}")

assert len(FEATURE_NAMES) == DISTANCE_FEATURE_COUNT


def snap_csi_to_complex(csi_bytes: bytes) -> Optional[np.ndarray]:
    """Parse raw snap CSI bytes into a 121-element complex array.

    ESP32 format: int8 pairs (imag, real) per subcarrier, 128 subcarriers.
    Null subcarriers are dropped.

    Returns None if buffer is too short.
    """
    if len(csi_bytes) < SUBCARRIERS * 2:
        return None

    raw = np.frombuffer(csi_bytes[: SUBCARRIERS * 2], dtype=np.int8)
    imag = raw[0::2].astype(np.float64)
    real = raw[1::2].astype(np.float64)
    full = real + 1j * imag  # (128,)
    return full[_VALID_INDICES]  # (121,)


def extract_snap_features(csi_complex: np.ndarray) -> List[float]:
    """Extract 242 features from one direction's 121-element complex CSI.

    Returns [amp_norm(121), phase(121)] where:
    - amp_norm: magnitudes normalized to [0, 1] (per-frame min-max)
    - phase: atan2 phase scaled by pi to [-1, 1]
    """
    amp = np.abs(csi_complex)
    amp_max = amp.max()
    if amp_max > 0:
        amp_norm = amp / amp_max
    else:
        amp_norm = np.zeros_like(amp)

    phase = np.angle(csi_complex) / math.pi  # [-1, 1]

    return amp_norm.tolist() + phase.tolist()


def pair_features(
    fwd_complex: np.ndarray, rev_complex: np.ndarray
) -> List[float]:
    """Combine forward + reverse snap CSI into a 484-element feature vector.

    Args:
        fwd_complex: 121-element complex array (reporter→peer direction)
        rev_complex: 121-element complex array (peer→reporter direction)

    Returns:
        484-element list: [fwd_amp_norm(121), fwd_phase(121),
                           rev_amp_norm(121), rev_phase(121)]
    """
    fwd_feats = extract_snap_features(fwd_complex)
    rev_feats = extract_snap_features(rev_complex)
    return fwd_feats + rev_feats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4" && python -m pytest tests/test_distance_features.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4"
git add ghv4/distance_features.py tests/test_distance_features.py
git commit -m "feat: add distance feature extraction module for snap CSI"
```

---

## Task 3: Distance Preprocessing Pipeline

**Files:**
- Create: `GHV4/ghv4/distance_preprocess.py`
- Create: `GHV4/run_distance_preprocess.py`
- Create: `GHV4/tests/test_distance_preprocess.py`

Loads raw distance CSVs, matches forward/reverse pairs, scales features, outputs per-pair X.npy/y.npy + shared scaler.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_distance_preprocess.py`:

```python
"""Tests for distance_preprocess — raw CSV → ML-ready arrays."""
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from ghv4.distance_preprocess import (
    derive_distances,
    match_paired_samples,
    build_dataset,
    run,
)
from ghv4.config import PAIR_KEYS, DISTANCE_FEATURE_COUNT


class TestDeriveDistances:
    def test_square_room(self):
        dists = derive_distances(width_m=3.0, depth_m=4.0)
        assert dists["1-4"] == pytest.approx(3.0)   # width
        assert dists["2-3"] == pytest.approx(3.0)   # width
        assert dists["1-2"] == pytest.approx(4.0)   # depth
        assert dists["3-4"] == pytest.approx(4.0)   # depth
        diag = (3.0**2 + 4.0**2) ** 0.5
        assert dists["1-3"] == pytest.approx(diag)   # diagonal
        assert dists["2-4"] == pytest.approx(diag)   # diagonal

    def test_all_six_pairs(self):
        dists = derive_distances(width_m=5.0, depth_m=5.0)
        assert set(dists.keys()) == set(PAIR_KEYS)


class TestMatchPairedSamples:
    def test_matches_by_seq(self):
        """Forward (1→2) and reverse (2→1) with same snap_seq match."""
        rows = [
            {"reporter_id": 1, "peer_id": 2, "snap_seq": 10, "val": "a"},
            {"reporter_id": 2, "peer_id": 1, "snap_seq": 10, "val": "b"},
            {"reporter_id": 1, "peer_id": 2, "snap_seq": 11, "val": "c"},
            # No reverse for seq 11 → unmatched
        ]
        df = pd.DataFrame(rows)
        pairs = match_paired_samples(df)
        # Only seq=10 should produce a match for pair "1-2"
        assert len(pairs) == 1
        fwd, rev = pairs[0]
        assert fwd["val"] == "a"
        assert rev["val"] == "b"


class TestBuildDataset:
    def test_output_shapes(self, tmp_path):
        """Synthetic CSV → correct X, y shapes."""
        # Create a minimal CSV with the right columns
        rng = np.random.default_rng(42)
        n_samples = 5
        n_feat = DISTANCE_FEATURE_COUNT
        X_vals = rng.standard_normal((n_samples, n_feat))
        records = []
        for i in range(n_samples):
            row = {}
            for j in range(n_feat):
                row[f"feat_{j}"] = X_vals[i, j]
            row["pair_id"] = "1-2"
            row["distance_m"] = 7.62
            row["session_id"] = "sess_01"
            records.append(row)
        df = pd.DataFrame(records)
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        X, y, groups = build_dataset(str(tmp_path), pair_id="1-2")
        assert X.shape == (n_samples, n_feat)
        assert y.shape == (n_samples,)
        assert len(groups) == n_samples
        assert np.all(y == 7.62)


class TestRunPipeline:
    def test_creates_output_artifacts(self, tmp_path):
        """Full pipeline produces expected files."""
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        raw_dir.mkdir()

        # Create minimal CSV data for pair 1-2
        rng = np.random.default_rng(42)
        n = 20
        feat_cols = [f"feat_{i}" for i in range(DISTANCE_FEATURE_COUNT)]
        data = {col: rng.standard_normal(n) for col in feat_cols}
        data["pair_id"] = ["1-2"] * n
        data["distance_m"] = [7.62] * n
        data["session_id"] = ["sess_01"] * 10 + ["sess_02"] * 10
        data["timestamp"] = list(range(n))
        data["width_m"] = [3.0] * n
        data["depth_m"] = [4.0] * n
        df = pd.DataFrame(data)
        df.to_csv(raw_dir / "session_001.csv", index=False)

        run(str(raw_dir), str(out_dir))

        assert (out_dir / "1-2_X.npy").exists()
        assert (out_dir / "1-2_y.npy").exists()
        assert (out_dir / "1-2_groups.npy").exists()
        assert (out_dir / "distance_scaler.pkl").exists()
        assert (out_dir / "distance_feature_names.txt").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4" && python -m pytest tests/test_distance_preprocess.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement distance_preprocess.py**

Create `ghv4/distance_preprocess.py`:

```python
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
```

- [ ] **Step 4: Create run_distance_preprocess.py entry point**

Create `run_distance_preprocess.py`:

```python
"""Entry point: preprocess raw distance CSVs into ML-ready arrays."""
import argparse
import logging

from ghv4.distance_preprocess import run


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Preprocess distance training data")
    parser.add_argument(
        "--raw-dir", default="distance_data/raw", help="Raw CSV directory"
    )
    parser.add_argument(
        "--out-dir", default="distance_data/processed", help="Output directory"
    )
    args = parser.parse_args()
    run(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4" && python -m pytest tests/test_distance_preprocess.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4"
git add ghv4/distance_preprocess.py run_distance_preprocess.py tests/test_distance_preprocess.py
git commit -m "feat: add distance preprocessing pipeline"
```

---

## Task 4: Distance Training Pipeline

**Files:**
- Create: `GHV4/ghv4/distance_train.py`
- Create: `GHV4/run_distance_train.py`
- Create: `GHV4/tests/test_distance_train.py`

Trains 6 independent regressors (one per pair) using GBT/RF with GroupKFold CV.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_distance_train.py`:

```python
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

        # Need groups for GroupKFold — create a groups file
        groups = np.array(["s1"] * 30 + ["s2"] * 30)
        np.save(processed_dir / "1-2_groups.npy", groups)

        run(str(processed_dir), str(model_dir))

        assert (model_dir / "1-2_model.pkl").exists()
        assert (model_dir / "distance_scaler.pkl").exists()
        assert (model_dir / "distance_feature_names.txt").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4" && python -m pytest tests/test_distance_train.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement distance_train.py**

Create `ghv4/distance_train.py`:

```python
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

from ghv4.config import DISTANCE_MAX_TREES, PAIR_KEYS, DISTANCE_FEATURE_COUNT

_log = logging.getLogger(__name__)


def train_pair_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 3,
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
        groups = np.load(g_path) if g_path.exists() else np.zeros(len(y))

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
        # Load median predictions would be done at inference time
        _log.info("All 6 pairs trained. Run inference for geometric check.")

    _log.info("Training complete. %d pair model(s) saved to %s",
              len(trained), model_dir)
```

- [ ] **Step 4: Create run_distance_train.py entry point**

Create `run_distance_train.py`:

```python
"""Entry point: train per-pair distance regressors."""
import argparse
import logging

from ghv4.distance_train import run


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train distance models")
    parser.add_argument(
        "--processed-dir",
        default="distance_data/processed",
        help="Preprocessed data directory",
    )
    parser.add_argument(
        "--model-dir",
        default="distance_models",
        help="Output model directory",
    )
    args = parser.parse_args()
    run(args.processed_dir, args.model_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4" && python -m pytest tests/test_distance_train.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4"
git add ghv4/distance_train.py run_distance_train.py tests/test_distance_train.py
git commit -m "feat: add per-pair distance regressor training pipeline"
```

---

## Task 5: Distance Inference Module (Calibration Phase)

**Files:**
- Create: `GHV4/ghv4/distance_inference.py`
- Create: `GHV4/tests/test_distance_inference.py`

Runs on Pi at session start: loads 6 models, buffers snap frames for 30 seconds, predicts distances, writes spacing.json.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_distance_inference.py`:

```python
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

    scaler = StandardScaler().fit(X_dummy[:, :121])
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4" && python -m pytest tests/test_distance_inference.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement distance_inference.py**

Create `ghv4/distance_inference.py`:

```python
"""Calibration-phase distance prediction from CSI snapshots.

Runs once per session on Pi 4B: loads 6 per-pair models, buffers
[0xEE][0xFF] snap frames, predicts median distances, writes spacing.json.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from ghv4.config import (
    CALIBRATION_MIN_PAIRS,
    DISTANCE_MODEL_DIR,
    PAIR_KEYS,
)
from ghv4.distance_features import (
    FEATURE_NAMES,
    pair_features,
    snap_csi_to_complex,
)

_log = logging.getLogger(__name__)


def load_distance_models(
    model_dir: str = DISTANCE_MODEL_DIR,
) -> Tuple[Dict[str, Any], Optional[Any]]:
    """Load per-pair models and shared scaler from model_dir.

    Returns:
        (models_dict, scaler) where models_dict maps pair_id → fitted model.
        Missing pairs are silently skipped.
    """
    models: Dict[str, Any] = {}
    for pair_id in PAIR_KEYS:
        path = os.path.join(model_dir, f"{pair_id}_model.pkl")
        if os.path.exists(path):
            models[pair_id] = joblib.load(path)
            _log.info("Loaded model for pair %s", pair_id)
        else:
            _log.warning("No model for pair %s at %s", pair_id, path)

    scaler_path = os.path.join(model_dir, "distance_scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    return models, scaler


def _pair_key(a: int, b: int) -> str:
    """Canonical pair key with lower ID first."""
    lo, hi = min(a, b), max(a, b)
    return f"{lo}-{hi}"


class DistanceCalibrator:
    """Buffers snap frames and predicts per-pair distances.

    Usage:
        cal = DistanceCalibrator("distance_models")
        # Feed snaps during calibration window...
        cal.feed_snap(reporter_id, peer_id, snap_seq, csi_bytes)
        # After window expires:
        distances = cal.predict_distances()
        cal.write_spacing("spacing.json", distances)
    """

    def __init__(self, model_dir: str = DISTANCE_MODEL_DIR) -> None:
        self._models, self._scaler = load_distance_models(model_dir)
        # Buffers: _snaps[(reporter_id, peer_id)][snap_seq] = complex array
        self._snaps: Dict[Tuple[int, int], Dict[int, np.ndarray]] = defaultdict(dict)
        self._lock = threading.Lock()

    def feed_snap(
        self, reporter_id: int, peer_id: int, snap_seq: int, csi_bytes: bytes
    ) -> None:
        """Buffer one parsed snap frame's CSI data."""
        csi_complex = snap_csi_to_complex(csi_bytes)
        if csi_complex is None:
            return
        with self._lock:
            self._snaps[(reporter_id, peer_id)][snap_seq] = csi_complex

    def predict_distances(self) -> Dict[str, float]:
        """Predict median distance for each pair with enough matched data.

        Returns dict mapping pair_id → predicted distance (meters).
        Pairs with fewer than CALIBRATION_MIN_PAIRS matched samples are omitted.
        """
        distances: Dict[str, float] = {}

        with self._lock:
            snaps_copy = {k: dict(v) for k, v in self._snaps.items()}

        for pair_id in PAIR_KEYS:
            if pair_id not in self._models:
                continue

            i, j = (int(x) for x in pair_id.split("-"))
            fwd_buf = snaps_copy.get((i, j), {})
            rev_buf = snaps_copy.get((j, i), {})

            # Match by snap_seq
            matched_seqs = set(fwd_buf.keys()) & set(rev_buf.keys())
            if len(matched_seqs) < CALIBRATION_MIN_PAIRS:
                _log.warning(
                    "Pair %s: only %d matched pairs (need %d)",
                    pair_id, len(matched_seqs), CALIBRATION_MIN_PAIRS,
                )
                continue

            # Build feature matrix
            features_list: List[List[float]] = []
            for seq in sorted(matched_seqs):
                vec = pair_features(fwd_buf[seq], rev_buf[seq])
                features_list.append(vec)

            X = np.array(features_list, dtype=np.float64)

            # Scale amp_norm columns
            if self._scaler is not None:
                amp_indices = list(range(121)) + list(range(242, 363))
                X[:, amp_indices] = self._scaler.transform(X[:, amp_indices])

            np.nan_to_num(X, copy=False)

            # Predict and take median
            preds = self._models[pair_id].predict(X)
            median_dist = float(np.median(preds))
            distances[pair_id] = max(0.0, median_dist)  # clamp negative

            _log.info(
                "Pair %s: %d samples → median %.2f m (std %.2f)",
                pair_id, len(preds), median_dist, float(np.std(preds)),
            )

        return distances

    def write_spacing(
        self, path: str, distances: Dict[str, float]
    ) -> None:
        """Write spacing.json in the existing v2 format."""
        pairs_block = {}
        for pair_id, dist_m in distances.items():
            pairs_block[pair_id] = {
                "distance_m": round(dist_m, 3),
                "source": "ml",
            }

        data = {
            "version": 2,
            "updated": datetime.now(timezone.utc).isoformat(),
            "pairs": pairs_block,
        }

        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
        _log.info("spacing.json written → %s (%d pairs)", path, len(distances))

    def matched_counts(self) -> Dict[str, int]:
        """Return count of matched fwd/rev pairs per shouter pair (for UI)."""
        with self._lock:
            snaps_copy = {k: dict(v) for k, v in self._snaps.items()}

        counts = {}
        for pair_id in PAIR_KEYS:
            i, j = (int(x) for x in pair_id.split("-"))
            fwd_seqs = set(snaps_copy.get((i, j), {}).keys())
            rev_seqs = set(snaps_copy.get((j, i), {}).keys())
            counts[pair_id] = len(fwd_seqs & rev_seqs)
        return counts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4" && python -m pytest tests/test_distance_inference.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4"
git add ghv4/distance_inference.py tests/test_distance_inference.py
git commit -m "feat: add calibration-phase distance inference module"
```

---

## Task 6: Add snap_callback to SerialReader

**Files:**
- Modify: `GHV4/ghv4/serial_io.py`
- Modify: `GHV4/tests/test_serial_io.py`

Adds an optional `snap_callback` parameter so [0xEE][0xFF] frames can be routed to the DistanceCalibrator during calibration.

- [ ] **Step 1: Read serial_io.py to identify exact modification points**

Run: Read `GHV4/ghv4/serial_io.py` (already explored — lines 25-136 for SerialReader)

- [ ] **Step 2: Write the failing test**

Append to `tests/test_serial_io.py`:

```python
def test_snap_callback_receives_parsed_frame():
    """When snap_callback is set, [0xEE][0xFF] frames are forwarded."""
    received = []

    def on_snap(reporter_id, peer_id, snap_seq, csi_bytes):
        received.append((reporter_id, peer_id, snap_seq, csi_bytes))

    # Build a minimal [0xEE][0xFF] frame
    # Header (6 bytes after magic): ver(1B)=1, reporter(1B)=1, peer(1B)=2, seq(1B)=10, csi_len(2B)=256
    import struct
    header = struct.pack("<BBBBH", 1, 1, 2, 10, 256)  # 6 bytes total
    csi_payload = bytes(256)
    frame_data = bytes([0xEE, 0xFF]) + header + csi_payload

    from io import BytesIO
    import queue
    from ghv4.serial_io import SerialReader

    ser = BytesIO(frame_data)
    ser.in_waiting = len(frame_data)  # mock property
    fq = queue.Queue()

    reader = SerialReader(ser, fq, snap_callback=on_snap)
    reader._read_one_frame()

    assert len(received) == 1
    assert received[0][0] == 1   # reporter_id
    assert received[0][1] == 2   # peer_id
    assert received[0][2] == 10  # snap_seq
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4" && python -m pytest tests/test_serial_io.py::test_snap_callback_receives_parsed_frame -v`
Expected: FAIL (TypeError — unexpected kwarg `snap_callback`)

- [ ] **Step 4: Modify SerialReader to accept snap_callback**

In `ghv4/serial_io.py`, modify `SerialReader.__init__` to accept an optional `snap_callback` parameter, and in the `[0xEE][0xFF]` handler section of `_read_one_frame`, call it after parsing:

```python
# In __init__:
def __init__(self, ser, frame_queue, music_estimator=None, snap_callback=None):
    ...
    self._snap_callback = snap_callback

# In _read_one_frame, after successful parse_csi_snap_frame:
if self._snap_callback is not None:
    self._snap_callback(
        frame['reporter_id'],
        frame['peer_id'],
        frame['snap_seq'],
        frame['csi'],
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4" && python -m pytest tests/test_serial_io.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4"
git add ghv4/serial_io.py tests/test_serial_io.py
git commit -m "feat: add snap_callback to SerialReader for distance calibration"
```

---

## Task 7: Integrate Calibration into inference.py

**Files:**
- Modify: `GHV4/ghv4/inference.py`
- Modify: `GHV4/tests/test_inference.py`

Add a calibration step that runs before the main zone-classification loop. If distance models are present, collect snaps for CALIBRATION_WINDOW_S seconds, predict distances, write spacing.json.

- [ ] **Step 1: Read current inference.py to identify exact integration point**

Run: Read `GHV4/ghv4/inference.py` — understand the main loop entry, how spacing is loaded (line 23+), how the serial reader is constructed.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_inference.py`:

```python
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

    scaler = StandardScaler().fit(X_d[:, :121])
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

    distances = run_calibration(
        ser=BytesIO(bytes(frames)),
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4" && python -m pytest tests/test_inference.py::test_run_calibration_with_serial -v`
Expected: FAIL (ImportError — `run_calibration` not found)

- [ ] **Step 4: Add run_calibration function to inference.py**

Add to `ghv4/inference.py`:

```python
def run_calibration(
    ser,
    model_dir: str,
    spacing_path: str,
    window_s: float = None,
) -> Dict[str, float]:
    """Run calibration phase: collect snaps via serial, predict distances, write spacing.

    Creates a DistanceCalibrator, reads [0xEE][0xFF] snap frames from `ser`
    for `window_s` seconds, then predicts per-pair distances and writes
    spacing.json. Extends the window if any pair has insufficient data.

    Args:
        ser: Open serial port object (or BytesIO for testing).
        model_dir: Path to distance_models/ directory.
        spacing_path: Where to write spacing.json.
        window_s: Collection window in seconds (default from config).
                  Set to 0.0 for testing with pre-filled BytesIO buffers.

    Returns:
        Dict mapping pair_id → predicted distance in meters.
    """
    import queue
    import time
    import threading
    from ghv4.config import (
        CALIBRATION_WINDOW_S,
        CALIBRATION_EXTENSION_S,
        CALIBRATION_MAX_EXTENSIONS,
        PAIR_KEYS,
    )
    from ghv4.distance_inference import DistanceCalibrator
    from ghv4.serial_io import SerialReader

    if window_s is None:
        window_s = CALIBRATION_WINDOW_S

    cal = DistanceCalibrator(model_dir)

    if not cal._models:
        _log.warning("No distance models found in %s, skipping calibration", model_dir)
        return {}

    _log.info("Calibration: collecting snaps for %.0f seconds...", window_s)

    # Create a SerialReader with snap_callback that feeds the calibrator
    fq = queue.Queue()
    reader = SerialReader(
        ser, fq,
        snap_callback=cal.feed_snap,
    )

    # Run reader in a background thread for the calibration window
    stop_event = threading.Event()
    original_run = reader.run

    def _timed_run():
        while not stop_event.is_set():
            try:
                reader._read_one_frame()
            except Exception:
                break

    reader_thread = threading.Thread(target=_timed_run, daemon=True)
    reader_thread.start()

    # Wait for collection window
    if window_s > 0:
        time.sleep(window_s)
    else:
        # For testing: let reader drain the buffer
        reader_thread.join(timeout=2.0)

    stop_event.set()
    reader_thread.join(timeout=2.0)

    # Predict distances
    distances = cal.predict_distances()

    # Extension logic: extend if any pair with a model is missing
    extensions = 0
    while extensions < CALIBRATION_MAX_EXTENSIONS:
        missing = [p for p in PAIR_KEYS if p in cal._models and p not in distances]
        if not missing:
            break
        extensions += 1
        _log.info("Extending calibration +%ds for pairs: %s",
                  CALIBRATION_EXTENSION_S, missing)

        stop_event.clear()
        reader_thread = threading.Thread(target=_timed_run, daemon=True)
        reader_thread.start()
        time.sleep(CALIBRATION_EXTENSION_S)
        stop_event.set()
        reader_thread.join(timeout=2.0)

        distances = cal.predict_distances()

    cal.write_spacing(spacing_path, distances)
    return distances
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4" && python -m pytest tests/test_inference.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4"
git add ghv4/inference.py tests/test_inference.py
git commit -m "feat: add calibration phase to inference pipeline"
```

---

## Task 8: Add Distance Data Collection Mode to capture_tab.py

**Files:**
- Modify: `GHV4/ghv4/ui/capture_tab.py`

Adds width_m/depth_m input fields and a "Collect Distance Training Data" button that records snap frames into distance_data/raw/ CSVs with paired features.

- [ ] **Step 1: Read capture_tab.py to identify the UI layout**

Run: Read `GHV4/ghv4/ui/capture_tab.py` — understand where to add widgets, how BackgroundCaptureThread works.

- [ ] **Step 2: Add width_m and depth_m input fields**

In the capture tab's UI setup section, add two labeled text inputs for width_m and depth_m (in meters), plus a "Collect Distance Data" button. These should appear below the existing controls.

```python
# In the capture tab UI setup (after existing controls):
# --- Distance Data Collection ---
dist_frame = ttk.LabelFrame(self, text="Distance Training Data")
dist_frame.pack(fill="x", padx=5, pady=5)

ttk.Label(dist_frame, text="Width (m):").grid(row=0, column=0, padx=5)
self._width_var = tk.StringVar(value="7.62")
ttk.Entry(dist_frame, textvariable=self._width_var, width=8).grid(row=0, column=1)

ttk.Label(dist_frame, text="Depth (m):").grid(row=0, column=2, padx=5)
self._depth_var = tk.StringVar(value="7.62")
ttk.Entry(dist_frame, textvariable=self._depth_var, width=8).grid(row=0, column=3)

self._dist_collect_btn = ttk.Button(
    dist_frame, text="Collect Distance Data",
    command=self._on_collect_distance,
)
self._dist_collect_btn.grid(row=0, column=4, padx=10)
```

- [ ] **Step 3: Implement _on_collect_distance handler**

The handler starts a BackgroundCaptureThread variant that:
1. Reads width_m/depth_m from the UI fields
2. Derives ground-truth distances via `derive_distances()`
3. Registers a snap_callback on the SerialReader
4. For each matched fwd/rev snap pair, extracts features via `pair_features()` and writes a CSV row with feature columns + pair_id + distance_m + session_id + metadata

```python
def _on_collect_distance(self):
    try:
        width_m = float(self._width_var.get())
        depth_m = float(self._depth_var.get())
    except ValueError:
        self._log("ERROR: width and depth must be numbers")
        return

    if width_m <= 0 or depth_m <= 0:
        self._log("ERROR: width and depth must be positive")
        return

    port = self._port_var.get()
    if not port:
        self._log("ERROR: select a serial port first")
        return

    import os, csv, datetime, serial, queue, threading
    from ghv4.distance_preprocess import derive_distances
    from ghv4.distance_features import (
        snap_csi_to_complex, pair_features, FEATURE_NAMES,
    )
    from ghv4.config import PAIR_KEYS, DISTANCE_FEATURE_COUNT

    dists = derive_distances(width_m, depth_m)
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("distance_data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"dist_{session_id}.csv")

    self._log(f"Distance collection → {csv_path}")
    self._log(f"  width={width_m}m  depth={depth_m}m  session={session_id}")

    # Buffers for matching fwd/rev snaps
    snap_buf = {}   # (reporter_id, peer_id) → {snap_seq: csi_bytes}
    csv_lock = threading.Lock()
    feat_cols = [f"feat_{i}" for i in range(DISTANCE_FEATURE_COUNT)]
    header = feat_cols + [
        "pair_id", "distance_m", "session_id", "timestamp",
        "width_m", "depth_m",
    ]

    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(header)

    def snap_cb(reporter_id, peer_id, snap_seq, csi_bytes):
        """Called by SerialReader for each [0xEE][0xFF] frame."""
        csi = snap_csi_to_complex(csi_bytes)
        if csi is None:
            return
        key = (reporter_id, peer_id)
        rev_key = (peer_id, reporter_id)

        with csv_lock:
            snap_buf.setdefault(key, {})[snap_seq] = csi
            # Check if reverse direction exists for this seq
            if rev_key in snap_buf and snap_seq in snap_buf[rev_key]:
                lo, hi = min(reporter_id, peer_id), max(reporter_id, peer_id)
                pair_id = f"{lo}-{hi}"
                if pair_id not in dists:
                    return
                if lo == reporter_id:
                    fwd, rev = csi, snap_buf[rev_key][snap_seq]
                else:
                    fwd, rev = snap_buf[rev_key][snap_seq], csi
                vec = pair_features(fwd, rev)
                ts = datetime.datetime.now().isoformat()
                row = vec + [pair_id, dists[pair_id], session_id,
                             ts, width_m, depth_m]
                writer.writerow(row)

    # The actual collection uses BackgroundCaptureThread's serial setup
    # with snap_callback=snap_cb. When collection stops, csv_file is closed.
    # Wire snap_cb into the capture thread's SerialReader via snap_callback.
    self._distance_snap_cb = snap_cb
    self._distance_csv_file = csv_file
    self._log("Distance collection started. Press Stop to finish.")
```

- [ ] **Step 4: Test manually via GUI**

Run: `cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4" && python run_gui.py`
Verify: Width/Depth fields appear, "Collect Distance Data" button is visible, no crashes.

- [ ] **Step 5: Commit**

```bash
cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4"
git add ghv4/ui/capture_tab.py
git commit -m "feat: add distance data collection UI (width/depth inputs)"
```

---

## Task 9: End-to-End Integration Test

**Files:**
- Create: `GHV4/tests/test_distance_e2e.py`

Validates the full pipeline: synthetic snap data → preprocess → train → calibrate → spacing.json.

- [ ] **Step 1: Write the integration test**

Create `tests/test_distance_e2e.py`:

```python
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
```

- [ ] **Step 2: Run the integration test**

Run: `cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4" && python -m pytest tests/test_distance_e2e.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4"
git add tests/test_distance_e2e.py
git commit -m "test: add end-to-end integration test for ML distance pipeline"
```

---

## Task 10: Run Full Test Suite and Final Cleanup

**Files:**
- All test files
- All new source files

- [ ] **Step 1: Run the complete test suite**

Run: `cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4" && python -m pytest tests/ -v --tb=short`
Expected: All new tests pass. Pre-existing failures (`test_distance_at_ref_rssi`, `test_distance_formula`) may still fail — these are known issues per CLAUDE.md.

- [ ] **Step 2: Verify no file exceeds 500 lines**

Run: `wc -l ghv4/distance_features.py ghv4/distance_preprocess.py ghv4/distance_train.py ghv4/distance_inference.py`
Expected: All under 500 lines.

- [ ] **Step 3: Verify imports work on Pi-compatible subset**

Run: `python -c "from ghv4.distance_inference import DistanceCalibrator; print('OK')"`
Expected: `OK` — no PC-only dependencies (pandas) imported at module level in distance_inference.py.

- [ ] **Step 4: Final commit with any cleanup**

```bash
cd "C:/Users/incre/Class/SENIOR CAP/CLAUDE/Glass House/GHV4"
git add -A
git status  # verify no unintended files
git commit -m "chore: ML distance estimation pipeline complete"
```

---

## Summary of Deliverables

| Artifact | Location |
|----------|----------|
| Feature extraction | `ghv4/distance_features.py` |
| Preprocessing | `ghv4/distance_preprocess.py` + `run_distance_preprocess.py` |
| Training | `ghv4/distance_train.py` + `run_distance_train.py` |
| Inference | `ghv4/distance_inference.py` |
| Config constants | `ghv4/config.py` (appended) |
| Serial routing | `ghv4/serial_io.py` (snap_callback added) |
| UI collection | `ghv4/ui/capture_tab.py` (width/depth + button) |
| Inference integration | `ghv4/inference.py` (calibration phase) |
| Tests | `tests/test_distance_{features,preprocess,train,inference,e2e}.py` |
