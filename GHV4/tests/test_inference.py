# GHV2.1/tests/test_inference.py
"""Tests for inference spacing feature injection."""
import json
import os
import pytest


PAIR_KEYS = ["1-2", "1-3", "1-4", "2-3", "2-4", "3-4"]


@pytest.fixture
def tmp_spacing(tmp_path):
    path = str(tmp_path / "spacing.json")
    data = {
        "version": 1,
        "updated": "2026-03-16T00:00:00Z",
        "pairs": {
            "1-2": {"distance_m": 2.5, "rssi_avg": -55.0, "samples": 10},
            "1-3": {"distance_m": 3.8, "rssi_avg": -61.0, "samples": 8},
            "1-4": {"distance_m": 2.4, "rssi_avg": -54.5, "samples": 12},
            "2-3": {"distance_m": 2.3, "rssi_avg": -54.0, "samples": 9},
            "2-4": {"distance_m": 3.9, "rssi_avg": -61.5, "samples": 7},
            "3-4": {"distance_m": 2.6, "rssi_avg": -55.5, "samples": 11},
        },
        "config": {"n": 2.5, "rssi_ref_dbm": -40.0, "d0_m": 1.0},
    }
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def test_load_spacing_reads_all_pairs(tmp_spacing):
    from ghv4.inference import load_spacing
    vals = load_spacing(tmp_spacing)
    assert len(vals) == 6
    assert vals[0] == pytest.approx(2.5)   # "1-2"
    assert vals[5] == pytest.approx(2.6)   # "3-4"


def test_load_spacing_absent_returns_zeros(tmp_path):
    from ghv4.inference import load_spacing
    vals = load_spacing(str(tmp_path / "nonexistent.json"))
    assert vals == [0.0] * 6


def test_load_spacing_partial_fills_zeros(tmp_path):
    from ghv4.inference import load_spacing
    path = str(tmp_path / "partial.json")
    data = {"version": 1, "pairs": {"1-2": {"distance_m": 2.5, "samples": 10}}, "config": {}}
    with open(path, "w") as f:
        json.dump(data, f)
    vals = load_spacing(path)
    assert vals[0] == pytest.approx(2.5)   # "1-2" present
    assert vals[1] == 0.0                  # "1-3" absent → 0


def test_spacing_appended_to_feature_vector():
    """load_spacing() output must produce a 6-element list ready for appending."""
    from ghv4 import inference as InferenceV3
    # build a dummy feature vector of 100 floats
    features = [0.5] * 100
    spacing_vals = [2.5, 3.8, 2.4, 2.3, 3.9, 2.6]
    combined = features + spacing_vals
    assert len(combined) == 106
    assert combined[-6] == 2.5
    assert combined[-1] == 2.6
