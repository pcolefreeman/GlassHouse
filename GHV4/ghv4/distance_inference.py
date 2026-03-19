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
        (models_dict, scaler) where models_dict maps pair_id -> fitted model.
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

        Returns dict mapping pair_id -> predicted distance (meters).
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
                "Pair %s: %d samples -> median %.2f m (std %.2f)",
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
        _log.info("spacing.json written -> %s (%d pairs)", path, len(distances))

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
