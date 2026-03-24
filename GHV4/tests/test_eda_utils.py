import io, math, os, tempfile, pytest
import numpy as np
import pandas as pd

from ghv4 import eda_utils


# ── parse_dimensions ──────────────────────────────────────────────────────────────

def test_parse_dimensions_float_dims():
    w, d = eda_utils.parse_dimensions("capture_6.0x4.0m_2026-03-15_143022.csv")
    assert w == pytest.approx(6.0)
    assert d == pytest.approx(4.0)

def test_parse_dimensions_integer_dims():
    w, d = eda_utils.parse_dimensions("capture_5x3m_2026-01-01.csv")
    assert w == pytest.approx(5.0)
    assert d == pytest.approx(3.0)

def test_parse_dimensions_no_dims_returns_none_tuple():
    result = eda_utils.parse_dimensions("capture_2026-03-15_143022.csv")
    assert result == (None, None)

def test_parse_dimensions_uses_basename_only():
    """Full path should still extract from the filename portion."""
    w, d = eda_utils.parse_dimensions("data/processed/capture_10.5x8.0m_ts.csv")
    assert w == pytest.approx(10.5)
    assert d == pytest.approx(8.0)


# ── group_columns ──────────────────────────────────────────────────────────────

def _make_df_with_cols(col_names):
    return pd.DataFrame(columns=col_names)

def test_group_columns_separates_rx_and_tx():
    cols = (
        ["timestamp_ms", "label", "zone_id", "grid_row", "grid_col"]
        + ["s1_amp_0", "s1_rssi", "s1_noise_floor"]
        + ["s1_tx_amp_0", "s1_tx_rssi", "s1_tx_noise_floor"]
    )
    df = _make_df_with_cols(cols)
    groups = eda_utils.group_columns(df)
    assert "s1_amp_0" in groups["s1"]
    assert "s1_rssi"  in groups["s1"]
    assert "s1_tx_amp_0"    in groups["s1_tx"]
    assert "s1_tx_rssi"     in groups["s1_tx"]
    assert "s1_tx_amp_0" not in groups["s1"]

def test_group_columns_meta_group():
    cols = ["timestamp_ms", "label", "zone_id", "grid_row", "grid_col", "s1_rssi"]
    df = _make_df_with_cols(cols)
    groups = eda_utils.group_columns(df)
    assert set(groups["meta"]) == {"timestamp_ms", "label", "zone_id", "grid_row", "grid_col"}

def test_group_columns_missing_shouter_gives_empty_list():
    """If no s3_ columns exist, s3 and s3_tx groups should be empty lists."""
    cols = ["timestamp_ms", "label", "zone_id", "grid_row", "grid_col", "s1_rssi"]
    df = _make_df_with_cols(cols)
    groups = eda_utils.group_columns(df)
    assert groups["s3"] == []
    assert groups["s3_tx"] == []


# ── parse_label ────────────────────────────────────────────────────────────────

def test_parse_label_single_cell():
    vec = eda_utils.parse_label("r0c1")
    assert list(vec) == [0, 1, 0, 0, 0, 0, 0, 0, 0]

def test_parse_label_compound():
    vec = eda_utils.parse_label("r0c0+r2c2")
    assert list(vec) == [1, 0, 0, 0, 0, 0, 0, 0, 1]

def test_parse_label_empty():
    vec = eda_utils.parse_label("empty")
    assert list(vec) == [0, 0, 0, 0, 0, 0, 0, 0, 0]

def test_parse_label_unrecognised_returns_zeros(capsys):
    vec = eda_utils.parse_label("r0C1")  # uppercase C — invalid
    assert list(vec) == [0, 0, 0, 0, 0, 0, 0, 0, 0]
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "r0C1" in captured.out

def test_parse_label_row_context_in_warning(capsys):
    eda_utils.parse_label("bad_label", row_context="row 5, ts=12345ms")
    captured = capsys.readouterr()
    assert "row 5, ts=12345ms" in captured.out

def test_parse_label_three_people():
    vec = eda_utils.parse_label("r0c0+r1c1+r2c2")
    assert list(vec) == [1, 0, 0, 0, 1, 0, 0, 0, 1]


# ── load_csv ───────────────────────────────────────────────────────────────────

META_COLS = ["timestamp_ms", "label", "zone_id", "grid_row", "grid_col", "activity"]

def _write_temp_csv(rows, filename="capture_6.0x4.0m_test.csv"):
    d = tempfile.mkdtemp()
    path = os.path.join(d, filename)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path

def test_load_csv_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        eda_utils.load_csv("/nonexistent/path/capture.csv")

def test_load_csv_missing_meta_col_raises():
    path = _write_temp_csv([{"timestamp_ms": 1, "label": "empty"}])
    with pytest.raises(ValueError, match="missing required meta columns"):
        eda_utils.load_csv(path)

def test_load_csv_empty_returns_empty_df_no_raise(capsys, tmp_path):
    # Write a CSV with meta column headers but zero data rows.
    path = str(tmp_path / "capture_6.0x4.0m_empty.csv")
    pd.DataFrame(columns=META_COLS).to_csv(path, index=False)
    df, dims = eda_utils.load_csv(path)
    assert len(df) == 0
    out = capsys.readouterr().out
    assert "WARNING" in out

def test_load_csv_parses_dims_from_filename():
    rows = [{c: 0 for c in META_COLS}]
    path = _write_temp_csv(rows, filename="capture_8.0x5.0m_test.csv")
    df, (w, d) = eda_utils.load_csv(path)
    assert w == pytest.approx(8.0)
    assert d == pytest.approx(5.0)

def test_load_csv_manual_dims_override_filename():
    rows = [{c: 0 for c in META_COLS}]
    path = _write_temp_csv(rows, filename="capture_8.0x5.0m_test.csv")
    df, (w, d) = eda_utils.load_csv(path, manual_dims=(3.0, 2.0))
    assert w == pytest.approx(3.0)
    assert d == pytest.approx(2.0)

def test_load_csv_no_dims_returns_none_tuple():
    rows = [{c: 0 for c in META_COLS}]
    path = _write_temp_csv(rows, filename="capture_nodims.csv")
    df, dims = eda_utils.load_csv(path)
    assert dims == (None, None)


# ── temporal_stats ────────────────────────────────────────────────────────────

def _make_meta_df(timestamps, labels=None):
    """Helper: build a minimal DataFrame with meta columns."""
    if labels is None:
        labels = ["empty"] * len(timestamps)
    return pd.DataFrame({
        "timestamp_ms": timestamps,
        "label":        labels,
        "zone_id":      [0] * len(timestamps),
        "grid_row":     [0] * len(timestamps),
        "grid_col":     [0] * len(timestamps),
    })

def test_temporal_stats_sorts_before_gap_detection():
    """Rows out of order should still detect the gap correctly."""
    df = _make_meta_df([600, 200, 400, 0])  # unsorted
    stats = eda_utils.temporal_stats(df)
    # Sorted: 0, 200, 400, 600 — diffs all 200 ms — no gaps
    assert stats["n_gaps"] == 0

def test_temporal_stats_detects_gap():
    """A jump > 2×BUCKET_MS (>400 ms) should be flagged as a gap."""
    df = _make_meta_df([0, 200, 800, 1000])  # gap between 200 and 800 (600 ms)
    stats = eda_utils.temporal_stats(df)
    assert stats["n_gaps"] == 1
    assert stats["gap_list"][0][1] == 600  # gap_duration_ms

def test_temporal_stats_empty_df():
    df = _make_meta_df([])
    stats = eda_utils.temporal_stats(df)
    assert stats["n_gaps"] == 0
    assert stats["sampling_rate_hz"] is None


# ── per_cell_stats ────────────────────────────────────────────────────────────

def test_per_cell_stats_counts():
    rows = [
        {"timestamp_ms": i, "label": "r0c0", "zone_id": 0,
         "grid_row": 0, "grid_col": 0,
         "s1_rssi": -60.0, "s2_rssi": -65.0, "s3_rssi": -70.0, "s4_rssi": -75.0}
        for i in range(5)
    ] + [
        {"timestamp_ms": i + 100, "label": "r1c1", "zone_id": 0,
         "grid_row": 1, "grid_col": 1,
         "s1_rssi": -55.0, "s2_rssi": -60.0, "s3_rssi": -65.0, "s4_rssi": -70.0}
        for i in range(3)
    ]
    df = pd.DataFrame(rows)
    stats = eda_utils.per_cell_stats(df)
    r0c0 = stats[(stats["grid_row"] == 0) & (stats["grid_col"] == 0)]
    assert int(r0c0["count"].iloc[0]) == 5
    # mean RSSI = mean(-60, -65, -70, -75) = -67.5
    assert r0c0["mean_rssi"].iloc[0] == pytest.approx(-67.5)

def test_per_cell_stats_no_rssi_cols():
    """Should not crash when RSSI columns are absent."""
    df = _make_meta_df([0, 200, 400])
    df["grid_row"] = [0, 0, 1]
    df["grid_col"] = [0, 0, 1]
    stats = eda_utils.per_cell_stats(df)
    assert "count" in stats.columns


# ── describe_dataset ──────────────────────────────────────────────────────────

def test_describe_dataset_returns_shape():
    df = _make_meta_df([0, 200, 400])
    groups = eda_utils.group_columns(df)
    result = eda_utils.describe_dataset(df, groups)
    assert result["shape"] == (3, 5)


# ── outlier_summary ───────────────────────────────────────────────────────────

def test_outlier_summary_no_outliers():
    df = _make_meta_df([0, 200, 400, 600, 800])
    df["s1_rssi"] = [-60.0, -61.0, -60.5, -59.5, -60.2]
    groups = eda_utils.group_columns(df)
    result = eda_utils.outlier_summary(df, groups)
    assert result["s1"]["n_outliers"] == 0

def test_outlier_summary_detects_outlier():
    df = _make_meta_df([0, 200, 400, 600, 800, 1000])
    df["s1_rssi"] = [-60.0, -60.0, -60.0, -60.0, -60.0, -200.0]  # -200 is outlier
    groups = eda_utils.group_columns(df)
    result = eda_utils.outlier_summary(df, groups)
    assert result["s1"]["n_outliers"] >= 1


# ── model_recommendation ──────────────────────────────────────────────────────

def test_model_recommendation_empty_df_returns_fallback():
    df = pd.DataFrame()
    result = eda_utils.model_recommendation(df)
    assert "No data available yet" in result

def test_model_recommendation_nonempty_df_mentions_random_forest():
    df = _make_meta_df([0, 200, 400])
    result = eda_utils.model_recommendation(df)
    assert "Random Forest" in result


# ── labeling_recommendation ───────────────────────────────────────────────────

def test_labeling_recommendation_returns_string():
    result = eda_utils.labeling_recommendation()
    assert isinstance(result, str)
    assert "empty" in result
    assert "r{row}c{col}" in result
