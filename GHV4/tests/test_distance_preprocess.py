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
