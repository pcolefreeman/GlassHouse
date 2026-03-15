# GHV2 EDA Pipeline & Live Heatmap Display — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a parameterized EDA notebook (`eda.ipynb`) and two helper modules (`ghv2_viz.py`, `eda_utils.py`) that analyse GHV2 CSV data and provide a reusable 3×3 heatmap used by both EDA and the future Raspberry Pi live display.

**Architecture:** `ghv2_viz.py` owns heatmap rendering (dual-mode: confidence for live display, raw for EDA). `eda_utils.py` owns all data loading, parsing, statistical, and recommendation logic. `eda.ipynb` calls both and is the primary GitHub-renderable deliverable. `GlassHouseV2.py` gains `--width`/`--depth` args and embeds dimensions in the output filename.

**Tech Stack:** Python 3.8+, pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, joblib, pytest

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| **Modify** | `GlassHouseV2.py` | Add `--width`/`--depth` CLI args; embed dims in output filename |
| **Create** | `ghv2_viz.py` | `render_heatmap()` — dual-mode (confidence / raw), reusable by notebook + Pi |
| **Create** | `eda_utils.py` | All EDA helpers: loading, parsing, stats, recommendations |
| **Create** | `eda.ipynb` | 8-section EDA notebook; GitHub-renderable primary deliverable |
| **Create** | `tests/test_ghv2_viz.py` | Unit tests for `render_heatmap` |
| **Create** | `tests/test_eda_utils.py` | Unit tests for all `eda_utils` functions |

**Reference:** Spec at `docs/superpowers/specs/2026-03-15-ghv2-eda-heatmap-design.md`

---

## Chunk 1: GlassHouseV2.py + ghv2_viz.py

---

### Task 1: GlassHouseV2.py — width/depth args and filename convention

**Files:**
- Modify: `GlassHouseV2.py:136-176` (the `main()` function)
- Create: `tests/test_glass_house_filename.py`

- [ ] **Step 1: Extract `_build_output_filename` helper above `main()` in `GlassHouseV2.py`**

Add this function just above `def main():` in `GlassHouseV2.py`:

```python
def _build_output_filename(out_dir, width, depth, timestamp=None):
    """Build the CSV output filename, embedding area dimensions when provided.

    Args:
        out_dir:   Directory path for output file.
        width:     Area width in metres, or None.
        depth:     Area depth in metres, or None.
        timestamp: Optional datetime string (YYYY-MM-DD_HHMMSS); generated if None.
    Returns:
        Full absolute path string.
    """
    import datetime as _dt
    ts = timestamp or _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if width is not None and depth is not None:
        filename = f"capture_{width:.1f}x{depth:.1f}m_{ts}.csv"
    else:
        filename = f"capture_{ts}.csv"
    return os.path.join(out_dir, filename)
```

- [ ] **Step 2: Write failing tests that import the real helper**

Create `tests/test_glass_house_filename.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from GlassHouseV2 import _build_output_filename


def test_filename_with_dims():
    path = _build_output_filename("/tmp/data", 6.0, 4.0, timestamp="2026-01-01_120000")
    name = os.path.basename(path)
    assert "6.0x4.0m" in name
    assert name.endswith(".csv")
    assert name.startswith("capture_")

def test_filename_without_dims():
    path = _build_output_filename("/tmp/data", None, None, timestamp="2026-01-01_120000")
    name = os.path.basename(path)
    assert "x" not in name
    assert name.startswith("capture_")
    assert name.endswith(".csv")

def test_filename_int_dims():
    path = _build_output_filename("/tmp/data", 5.0, 3.0, timestamp="2026-01-01_120000")
    name = os.path.basename(path)
    assert "5.0x3.0m" in name

def test_filename_includes_output_dir():
    path = _build_output_filename("/my/dir", 6.0, 4.0, timestamp="2026-01-01_120000")
    assert path.startswith("/my/dir")
```

- [ ] **Step 3: Run tests to verify they fail**

```
pytest tests/test_glass_house_filename.py -v
```

Expected: `ImportError` — `cannot import name '_build_output_filename'` (function doesn't exist yet)

- [ ] **Step 4: Add `_build_output_filename` and update `main()` to use it**

Add `_build_output_filename` above `main()` (code in Step 1 above), then modify `main()`:

```python
def main():
    import argparse, datetime
    parser = argparse.ArgumentParser(description="GHV2 data collection")
    parser.add_argument('--port',   default=SERIAL_PORT)
    parser.add_argument('--output', default=OUTPUT_CSV)
    parser.add_argument('--label',  default='unknown')
    parser.add_argument('--zone',   type=int,   default=0)
    parser.add_argument('--row',    type=int,   default=0)
    parser.add_argument('--col',    type=int,   default=0)
    parser.add_argument('--width',  type=float, default=None,
                        help="Search area width in metres (embeds in filename)")
    parser.add_argument('--depth',  type=float, default=None,
                        help="Search area depth in metres (embeds in filename)")
    args = parser.parse_args()

    out_dir     = os.path.dirname(os.path.abspath(args.output))
    output_path = _build_output_filename(out_dir, args.width, args.depth)

    os.makedirs(out_dir, exist_ok=True)
    meta = {'label': args.label, 'zone_id': args.zone,
            'grid_row': args.row, 'grid_col': args.col}

    frame_queue = queue.Queue()
    ser = serial.Serial(args.port, BAUD_RATE, timeout=1)
    print(f"[GHV2] {args.port}  →  {output_path}")
    print(f"[GHV2] label={args.label}  zone={args.zone}  row={args.row}  col={args.col}")
    if args.width and args.depth:
        print(f"[GHV2] area={args.width:.1f}m × {args.depth:.1f}m")
    print("[GHV2] Press Ctrl+C to stop\n")

    with open(output_path, 'w', newline='') as f_out:
        reader = SerialReader(ser, frame_queue)
        writer = CSVWriter(frame_queue, f_out)
        reader.start()
        writer.start()
        try:
            while True:
                time.sleep(BUCKET_MS / 1000.0)
                frame_queue.put(('flush', dict(meta)))
        except KeyboardInterrupt:
            print("\n[GHV2] Stopping…")
        finally:
            reader.stop()
            frame_queue.put(None)
            writer.join(timeout=2)
            ser.close()
    print(f"[GHV2] Saved to {output_path}")
```

- [ ] **Step 5: Run tests to verify they pass**

```
pytest tests/test_glass_house_filename.py tests/test_glass_house_v2.py -v
```

Expected: all `PASS`

- [ ] **Step 6: Commit**

```
git add GlassHouseV2.py tests/test_glass_house_filename.py
git commit -m "feat(collect): add --width/--depth args and embed dims in output filename"
```

---

### Task 2: Create `ghv2_viz.py` — render_heatmap

**Files:**
- Create: `ghv2_viz.py`
- Create: `tests/test_ghv2_viz.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ghv2_viz.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless for CI
import matplotlib.pyplot as plt
import matplotlib.figure
import pytest

# ── import guard (file doesn't exist yet — tests will fail to import) ──────────
import ghv2_viz

GRID_CONF = np.array([
    [0.9, 0.3, 0.8],
    [0.2, 0.75, 0.1],
    [0.6, 0.4, 0.95],
], dtype=float)

GRID_RAW = np.array([
    [-65.0, -72.0, -58.0],
    [-80.0, -61.0, -75.0],
    [-55.0, -68.0, -71.0],
], dtype=float)

AREA = (6.0, 4.0)


# ── return type ────────────────────────────────────────────────────────────────

def test_render_heatmap_no_ax_returns_figure():
    fig = ghv2_viz.render_heatmap(GRID_CONF, AREA, "Test", mode="confidence")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')


def test_render_heatmap_with_ax_returns_none():
    fig_ext, ax = plt.subplots()
    result = ghv2_viz.render_heatmap(GRID_CONF, AREA, "Test", mode="confidence", ax=ax)
    assert result is None
    plt.close('all')


def test_raw_mode_returns_figure():
    fig = ghv2_viz.render_heatmap(GRID_RAW, AREA, "RSSI", mode="raw")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')


def test_raw_mode_with_ax_returns_none():
    fig_ext, ax = plt.subplots()
    result = ghv2_viz.render_heatmap(GRID_RAW, AREA, "RSSI", mode="raw", ax=ax)
    assert result is None
    plt.close('all')


def test_no_area_dims_does_not_raise():
    """(None, None) area_dims should produce cell-index labels without error."""
    fig = ghv2_viz.render_heatmap(GRID_CONF, (None, None), "No dims")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')


def test_all_below_threshold_does_not_raise():
    grid = np.zeros((3, 3), dtype=float)  # all confidence = 0
    fig = ghv2_viz.render_heatmap(grid, AREA, "All dark", threshold=0.70)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')


def test_all_above_threshold_does_not_raise():
    grid = np.ones((3, 3), dtype=float)  # all confidence = 1
    fig = ghv2_viz.render_heatmap(grid, AREA, "All lit", threshold=0.70)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')


def test_raw_mode_uniform_values_does_not_raise():
    """Uniform grid (max == min) should not divide by zero."""
    grid = np.full((3, 3), -70.0)
    fig = ghv2_viz.render_heatmap(grid, AREA, "Uniform", mode="raw")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_ghv2_viz.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'ghv2_viz'`

- [ ] **Step 3: Create `ghv2_viz.py`**

```python
"""ghv2_viz.py — GHV2 heatmap rendering, shared by EDA notebook and Pi live display.

Public API:
    render_heatmap(grid_values, area_dims, title, mode, threshold,
                   cmap_lit, cmap_raw, ax) -> Optional[Figure]
"""
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def render_heatmap(
    grid_values,            # np.ndarray shape (3, 3)
    area_dims,              # (width_m, depth_m) or (None, None)
    title,                  # str
    mode="confidence",      # "confidence" | "raw"
    threshold=0.70,         # float: confidence threshold (mode="confidence" only)
    cmap_lit="#FF6B35",     # rescue orange (mode="confidence" only)
    cmap_raw="YlOrRd",      # colormap for raw values
    ax=None,                # None → new Figure; provided → redraw in-place
):
    # type: (...) -> Optional[matplotlib.figure.Figure]
    """Render a 3×3 occupancy or analysis heatmap.

    Returns a new Figure when ax=None (EDA use).
    Returns None when ax is provided (live Pi animation — caller owns Figure).
    """
    owns_fig = (ax is None)
    if owns_fig:
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor('#0d0d0d' if mode == "confidence" else 'white')
    else:
        fig = None
        ax.cla()

    grid = np.array(grid_values, dtype=float)

    if mode == "confidence":
        _draw_confidence(ax, grid, threshold, cmap_lit)
    else:
        _draw_raw(ax, grid, cmap_raw)

    _set_axis_labels(ax, area_dims, mode)
    ax.set_title(title, color='white' if mode == "confidence" else 'black', pad=10)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')

    if owns_fig:
        plt.tight_layout()
        return fig
    return None


def _draw_confidence(ax, grid, threshold, cmap_lit):
    """Confidence mode: dark below threshold, rescue-orange fill + glow above."""
    ax.set_facecolor('#0d0d0d')
    for row in range(3):
        for col in range(3):
            conf = float(grid[row, col])
            # row 0 is displayed at top → y = 2 - row
            x, y = col, 2 - row

            if conf >= threshold:
                # Solid rescue-orange fill
                rect = patches.Rectangle(
                    (x, y), 1, 1,
                    facecolor=cmap_lit, edgecolor='white', linewidth=1.5,
                )
                ax.add_patch(rect)
                # Glow overlay (stacked on top; only when strictly above threshold).
                # Spec formula: glow_alpha = (conf - threshold) / (1 - threshold) → [0, 1].
                # Scaled by 0.3 as an implementation detail to keep it subtle and
                # prevent the white overlay from washing out the rescue-orange fill.
                if conf > threshold:
                    glow_alpha = (conf - threshold) / (1.0 - threshold) * 0.3
                    glow = patches.Rectangle(
                        (x, y), 1, 1,
                        facecolor='white', alpha=glow_alpha, edgecolor='none',
                    )
                    ax.add_patch(glow)
                label_color = 'white'
            else:
                rect = patches.Rectangle(
                    (x, y), 1, 1,
                    facecolor='#1a1a1a', edgecolor='#444444', linewidth=1.5,
                )
                ax.add_patch(rect)
                label_color = '#666666'

            pct = int(conf * 100)
            ax.text(
                x + 0.5, y + 0.5, f"{pct}%",
                ha='center', va='center',
                fontsize=12, color=label_color, fontweight='bold',
            )


def _draw_raw(ax, grid, cmap_raw):
    """Raw mode: normalised colormap applied to arbitrary float values."""
    ax.set_facecolor('white')
    vmin = float(np.nanmin(grid))
    vmax = float(np.nanmax(grid))
    eps  = 1e-9
    norm_grid = (grid - vmin) / (vmax - vmin + eps)

    # matplotlib 3.7+ removed plt.get_cmap(); use matplotlib.colormaps (3.5+)
    cmap = matplotlib.colormaps.get_cmap(cmap_raw)
    for row in range(3):
        for col in range(3):
            norm_val = float(norm_grid[row, col])
            color    = cmap(norm_val)
            x, y     = col, 2 - row

            rect = patches.Rectangle(
                (x, y), 1, 1,
                facecolor=color, edgecolor='white', linewidth=1.5,
            )
            ax.add_patch(rect)

            raw_val    = float(grid[row, col])
            text_color = 'black' if norm_val > 0.5 else 'white'
            ax.text(
                x + 0.5, y + 0.5, f"{raw_val:.1f}",
                ha='center', va='center',
                fontsize=11, color=text_color,
            )


def _set_axis_labels(ax, area_dims, mode):
    """Set cell-centre tick labels — physical metres or cell indices."""
    tick_color = 'white' if mode == "confidence" else 'black'

    # Ticks are always at cell-centre positions in plot coordinates [0, 3].
    # Cell centres are at 0.5, 1.5, 2.5 (col 0, 1, 2 respectively).
    # Labels are physical metres when area_dims is known, cell indices otherwise.
    if area_dims is not None and area_dims[0] is not None:
        width_m, depth_m = float(area_dims[0]), float(area_dims[1])
        cw = width_m / 3.0
        cd = depth_m / 3.0

        # Cell centres in plot coords = [0.5, 1.5, 2.5]; labels show physical range
        x_labels = [f"{i * cw:.1f}–{(i + 1) * cw:.1f} m" for i in range(3)]
        ax.set_xticks([0.5, 1.5, 2.5])
        ax.set_xticklabels(x_labels, fontsize=9, color=tick_color)
        ax.set_xlabel(f"Width ({width_m:.1f} m)", color=tick_color)

        # Row 0 is at top (y=2 in plot); row 2 is at bottom (y=0 in plot).
        # y-tick centres: row 2 → y=0.5, row 1 → y=1.5, row 0 → y=2.5
        y_labels = [
            f"{(2 - i) * cd:.1f}–{(3 - i) * cd:.1f} m"   # row at y-centre i+0.5
            for i in range(3)                               # i=0 → bottom, i=2 → top
        ]
        ax.set_yticks([0.5, 1.5, 2.5])
        ax.set_yticklabels(y_labels, fontsize=9, color=tick_color)
        ax.set_ylabel(f"Depth ({depth_m:.1f} m)", color=tick_color)
    else:
        ax.set_xticks([0.5, 1.5, 2.5])
        ax.set_xticklabels(['0', '1', '2'], color=tick_color)
        ax.set_xlabel("Cell index (col)", color=tick_color)

        ax.set_yticks([0.5, 1.5, 2.5])
        ax.set_yticklabels(['2', '1', '0'], color=tick_color)
        ax.set_ylabel("Cell index (row)", color=tick_color)

    ax.tick_params(colors=tick_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(tick_color)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_ghv2_viz.py -v
```

Expected: all `PASS`

- [ ] **Step 5: Commit**

```
git add ghv2_viz.py tests/test_ghv2_viz.py
git commit -m "feat(viz): add ghv2_viz.py with dual-mode render_heatmap (confidence + raw)"
```

---

## Chunk 2: eda_utils.py

---

### Task 3: `eda_utils.py` — dimension parsing, column grouping, label parsing, CSV loading

**Files:**
- Create: `eda_utils.py`
- Create: `tests/test_eda_utils.py`

- [ ] **Step 1: Write failing tests for parse_dimensions, group_columns, parse_label, load_csv**

Create `tests/test_eda_utils.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import io, math, tempfile, pytest
import numpy as np
import pandas as pd

import eda_utils


# ── parse_dimensions ──────────────────────────────────────────────────────────

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

META_COLS = ["timestamp_ms", "label", "zone_id", "grid_row", "grid_col"]

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
    # pd.DataFrame([]) has no columns, so we must write with explicit columns.
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
```

- [ ] **Step 2: Run to verify failures**

```
pytest tests/test_eda_utils.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'eda_utils'`

- [ ] **Step 3: Create `eda_utils.py` with parse_dimensions, group_columns, parse_label, load_csv**

```python
"""eda_utils.py — EDA helper functions for GHV2 data analysis.

All heavy logic used by eda.ipynb. Designed to handle empty DataFrames
gracefully (data may not exist yet when building the notebook).
"""
from typing import Optional
import os
import re
import math

import numpy as np
import pandas as pd

import csi_parser

# ── Constants ─────────────────────────────────────────────────────────────────
META_COLS          = ["timestamp_ms", "label", "zone_id", "grid_row", "grid_col"]
EXPECTED_COLS      = 5133
BUCKET_MS          = csi_parser.BUCKET_MS   # 200
SHOUTER_IDS        = csi_parser.ACTIVE_SHOUTER_IDS  # [1, 2, 3, 4]


# ── Dimension parsing ──────────────────────────────────────────────────────────

def parse_dimensions(filename):
    # type: (str) -> tuple  # (float, float) or (None, None)
    """Extract (width_m, depth_m) from filename via regex r'(\d+\.?\d*)x(\d+\.?\d*)m'.

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
    for part in parts:
        m = re.fullmatch(r'r([0-2])c([0-2])', part)
        if not m:
            print(
                f"WARNING: Unrecognised label '{label}' at {row_context}"
                f" — treated as empty"
            )
            return np.zeros(n_cells, dtype=int)
        row = int(m.group(1))
        col = int(m.group(2))
        idx = row * 3 + col
        if idx < n_cells:
            target[idx] = 1
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
        area_dims = parse_dimensions(os.path.basename(path))

    return df, area_dims
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_eda_utils.py::test_parse_dimensions_float_dims \
       tests/test_eda_utils.py::test_group_columns_separates_rx_and_tx \
       tests/test_eda_utils.py::test_parse_label_single_cell \
       tests/test_eda_utils.py::test_load_csv_missing_file_raises \
       -v
```

Expected: all `PASS`

- [ ] **Step 5: Run all eda_utils tests collected so far**

```
pytest tests/test_eda_utils.py -v
```

Expected: all `PASS`

- [ ] **Step 6: Commit**

```
git add eda_utils.py tests/test_eda_utils.py
git commit -m "feat(eda): add parse_dimensions, group_columns, parse_label, load_csv"
```

---

### Task 4: `eda_utils.py` — statistical, temporal, spatial, and recommendation functions

**Files:**
- Modify: `eda_utils.py` (append new functions)
- Modify: `tests/test_eda_utils.py` (append new tests)

- [ ] **Step 1: Append failing tests to `tests/test_eda_utils.py`**

Add to the bottom of `tests/test_eda_utils.py`:

```python
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
```

- [ ] **Step 2: Run to see new tests fail**

```
pytest tests/test_eda_utils.py -v -k "temporal or per_cell or describe or outlier or model_recommendation or labeling"
```

Expected: `FAIL` — functions not yet defined

- [ ] **Step 3: Append statistical functions to `eda_utils.py`**

Add after `load_csv` in `eda_utils.py`:

```python
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
        q1  = sub.quantile(0.25)
        q3  = sub.quantile(0.75)
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
```

- [ ] **Step 4: Run all eda_utils tests**

```
pytest tests/test_eda_utils.py -v
```

Expected: all `PASS`

- [ ] **Step 5: Commit**

```
git add eda_utils.py tests/test_eda_utils.py
git commit -m "feat(eda): add temporal, spatial, feature, and recommendation functions to eda_utils"
```

---

## Chunk 3: eda.ipynb

---

### Task 5: Create `eda.ipynb` — full EDA notebook

**Files:**
- Create: `eda.ipynb`

The notebook is not unit-tested with pytest. Instead, verify correctness by running all cells against an empty DataFrame (the `load_csv` warning path). Each section should complete without errors even when `df` has zero rows.

- [ ] **Step 1: Create `eda.ipynb` with all 9 sections**

Create `eda.ipynb` with the following cell structure. Each cell is shown as a fenced code block; the actual file is a Jupyter notebook JSON:

**Cell 1 — Markdown: Title**
```markdown
# GHV2 Exploratory Data Analysis

Search & Rescue WiFi CSI system — bidirectional CSI from 4 ESP32 shouters placed at area corners.

**Run all cells.** If the CSV has no data yet, each section will display a "no data" state gracefully.
```

**Cell 2 — Code: Section 0 — Config**
```python
import sys, os
sys.path.insert(0, os.path.abspath('..') if not os.path.exists('csi_parser.py') else '.')

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import eda_utils
import ghv2_viz

# ── Configuration ─────────────────────────────────────────────────────────────
CSV_PATH           = "data/processed/capture.csv"   # update to your actual file
MANUAL_DIMS        = (None, None)   # override with (width_m, depth_m) if needed
CONFIDENCE_THRESHOLD = 0.70
PLOT_DPI           = 150

%matplotlib inline
plt.rcParams['figure.dpi'] = PLOT_DPI
print(f"Config: CSV={CSV_PATH}  dims={MANUAL_DIMS}  threshold={CONFIDENCE_THRESHOLD}")
```

**Cell 3 — Markdown: Section 1 — Data Loading**
```markdown
## Section 1 — Data Loading & Schema
```

**Cell 4 — Code: Load CSV**
```python
try:
    df, area_dims = eda_utils.load_csv(CSV_PATH, manual_dims=MANUAL_DIMS or None)
    groups = eda_utils.group_columns(df)
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Area dimensions: {area_dims}")
    print(f"Column groups: { {k: len(v) for k, v in groups.items()} }")
    display(df[eda_utils.META_COLS].head())
except FileNotFoundError as e:
    print(f"[INFO] {e}\nCreate a CSV first using GlassHouseV2.py.")
    df, area_dims, groups = pd.DataFrame(), (None, None), {}
```

**Cell 5 — Markdown: Section 2 — Statistical Summary**
```markdown
## Section 2 — Statistical Summary
```

**Cell 6 — Code: Descriptives + missing values + outliers**
```python
if len(df) == 0:
    print("[INFO] No data — skipping statistical summary.")
else:
    desc = eda_utils.describe_dataset(df, groups)
    print(f"Shape: {desc['shape']}")
    summary_rows = [
        {"group": k, "n_cols": v["n_cols"], "missing_pct": v["missing_pct"]}
        for k, v in desc.items() if isinstance(v, dict)
    ]
    display(pd.DataFrame(summary_rows))

    outliers = eda_utils.outlier_summary(df, groups)
    print("\nOutlier summary (IQR method):")
    display(pd.DataFrame(outliers).T)
```

**Cell 7 — Code: Missing value heatmap**
```python
if len(df) > 0:
    # Show missing values for meta + scalar columns only (full 5133 is too wide)
    scalar_cols = groups.get("meta", [])
    for sid in [1, 2, 3, 4]:
        scalar_cols += [c for c in df.columns
                        if c in (f"s{sid}_rssi", f"s{sid}_noise_floor",
                                 f"s{sid}_tx_rssi", f"s{sid}_tx_noise_floor")]
    if scalar_cols:
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.heatmap(df[scalar_cols].isna().T, ax=ax, cbar=False,
                    xticklabels=False, yticklabels=True, cmap="Reds")
        ax.set_title("Missing Values — Meta + Scalar Columns (red = missing)")
        plt.tight_layout()
        plt.show()
```

**Cell 8 — Markdown: Section 3 — Temporal Analysis**
```markdown
## Section 3 — Temporal Analysis
```

**Cell 9 — Code: Temporal stats + RSSI time series**
```python
if len(df) == 0:
    print("[INFO] No data — skipping temporal analysis.")
else:
    stats = eda_utils.temporal_stats(df)
    print(f"Sampling rate: {stats['sampling_rate_hz']} Hz  "
          f"(mean interval: {stats['mean_interval_ms']} ms ± {stats['std_interval_ms']} ms)")
    print(f"Gaps detected (> 400 ms): {stats['n_gaps']}")
    if stats["gap_list"]:
        print("Gap locations (timestamp_ms, duration_ms):")
        for ts, dur in stats["gap_list"][:10]:
            print(f"  t={ts}ms  gap={dur}ms")

    rssi_cols = [f"s{sid}_rssi" for sid in [1, 2, 3, 4] if f"s{sid}_rssi" in df.columns]
    if rssi_cols:
        fig, ax = plt.subplots(figsize=(12, 4))
        ts_sorted = df.sort_values("timestamp_ms")
        for col in rssi_cols:
            ax.plot(ts_sorted["timestamp_ms"], ts_sorted[col],
                    alpha=0.7, linewidth=0.8, label=col)
        ax.set_xlabel("timestamp_ms")
        ax.set_ylabel("RSSI (dBm)")
        ax.set_title("RSSI Time Series — All Shouters (listener-rx)")
        ax.legend()
        plt.tight_layout()
        plt.show()
```

**Cell 10 — Markdown: Section 4 — Spatial Analysis**
```markdown
## Section 4 — Spatial Analysis
```

**Cell 11 — Code: Per-cell stats + coverage bar chart**
```python
if len(df) == 0:
    print("[INFO] No data — skipping spatial analysis.")
else:
    cell_stats = eda_utils.per_cell_stats(df)
    print("Per-cell statistics:")
    display(cell_stats)

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"r{int(r)}c{int(c)}" for r, c in
              zip(cell_stats["grid_row"], cell_stats["grid_col"])]
    ax.bar(labels, cell_stats["count"], color="#4C9BE8")
    ax.set_xlabel("Grid cell")
    ax.set_ylabel("Sample count")
    ax.set_title("Sample Count per Grid Cell")
    plt.tight_layout()
    plt.show()
```

**Cell 12 — Markdown: Section 5 — Feature Analysis**
```markdown
## Section 5 — Feature Analysis
```

**Cell 13 — Code: Per-shouter amplitude box plots**
```python
if len(df) == 0:
    print("[INFO] No data — skipping feature analysis.")
else:
    for sid in [1, 2, 3, 4]:
        amp_cols = [c for c in groups.get(f"s{sid}", []) if "_amp_" in c][:16]
        if not amp_cols:
            continue
        fig, ax = plt.subplots(figsize=(12, 3))
        df[amp_cols].plot.box(ax=ax, rot=90, showfliers=False)
        ax.set_title(f"Shouter {sid} — Listener-RX Amplitude (first 16 subcarriers)")
        ax.set_ylabel("Amplitude")
        plt.tight_layout()
        plt.show()
```

**Cell 14 — Code: Correlation heatmap (scalar features)**
```python
if len(df) > 0:
    all_scalar = []
    for sid in [1, 2, 3, 4]:
        for grp_key in [f"s{sid}", f"s{sid}_tx"]:
            all_scalar += [c for c in groups.get(grp_key, [])
                           if c.endswith("_rssi") or c.endswith("_noise_floor")]
    if all_scalar:
        corr = df[all_scalar].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, square=True, linewidths=0.5)
        ax.set_title("Correlation — Scalar Features (RSSI + Noise Floor)")
        plt.tight_layout()
        plt.show()
```

**Cell 15 — Code: Phase polar histograms**
```python
if len(df) > 0:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4),
                              subplot_kw={"projection": "polar"})
    for i, sid in enumerate([1, 2, 3, 4]):
        phase_vals = eda_utils.phase_polar_data(df, groups.get(f"s{sid}", []))
        if len(phase_vals) > 0:
            axes[i].hist(phase_vals, bins=64, density=True, color="#4C9BE8", alpha=0.8)
        axes[i].set_title(f"S{sid} Phase", pad=10)
    plt.suptitle("Phase Distribution — Listener-RX (polar histogram)")
    plt.tight_layout()
    plt.show()
```

**Cell 16 — Markdown: Section 6 — 3×3 Heatmap**
```markdown
## Section 6 — 3×3 Spatial Heatmap

Grid cell dimensions:
- Area dimensions parsed from filename (or MANUAL_DIMS override)
- `cell_width = total_width / 3`, `cell_depth = total_depth / 3`
```

**Cell 17 — Code: Print grid dimensions**
```python
if area_dims != (None, None):
    w, d = area_dims
    cw, cd = w / 3.0, d / 3.0
    print(f"Area: {w:.1f} m × {d:.1f} m")
    print(f"Cell: {cw:.2f} m × {cd:.2f} m  (width × depth)")
else:
    print("Area dimensions unknown — using cell index labels. "
          "Set MANUAL_DIMS = (width_m, depth_m) in Config to add physical labels.")
```

**Cell 18 — Code: Mean RSSI heatmap**
```python
if len(df) == 0:
    print("[INFO] No data — skipping heatmap.")
else:
    cell_stats = eda_utils.per_cell_stats(df)
    grid_rssi = np.full((3, 3), np.nan)
    for _, row in cell_stats.iterrows():
        grid_rssi[int(row["grid_row"]), int(row["grid_col"])] = row["mean_rssi"]

    fig = ghv2_viz.render_heatmap(
        grid_rssi, area_dims,
        title="Mean RSSI per Cell (dBm) — Listener-RX Average",
        mode="raw",
    )
    plt.show()
```

**Cell 19 — Code: Sample count heatmap**
```python
if len(df) > 0:
    cell_stats = eda_utils.per_cell_stats(df)   # re-compute; avoids cross-cell dependency
    grid_count = np.zeros((3, 3), dtype=float)
    for _, row in cell_stats.iterrows():
        grid_count[int(row["grid_row"]), int(row["grid_col"])] = row["count"]

    fig = ghv2_viz.render_heatmap(
        grid_count, area_dims,
        title="Sample Count per Cell",
        mode="raw",
        cmap_raw="Blues",
    )
    plt.show()
```

**Cell 20 — Markdown: Section 7 — Pairwise Relationships**
```markdown
## Section 7 — Pairwise Relationships (Scalar Features)
```

**Cell 21 — Code: Scatter matrix**
```python
if len(df) == 0:
    print("[INFO] No data — skipping pairwise plot.")
else:
    scalar_cols = [f"s{sid}_rssi" for sid in [1,2,3,4] if f"s{sid}_rssi" in df.columns]
    scalar_cols += [f"s{sid}_noise_floor" for sid in [1,2,3,4]
                    if f"s{sid}_noise_floor" in df.columns]
    if len(scalar_cols) >= 2:
        pd.plotting.scatter_matrix(df[scalar_cols], figsize=(10, 10),
                                   alpha=0.3, diagonal='kde')
        plt.suptitle("Pairwise Scatter — RSSI + Noise Floor (all shouters)")
        plt.tight_layout()
        plt.show()
```

**Cell 22 — Markdown: Section 8 — Recommendations**
```markdown
## Section 8 — Model & Labeling Recommendations
```

**Cell 23 — Code: Print recommendations**
```python
print("=" * 60)
print("MODEL SELECTION GUIDANCE")
print("=" * 60)
print(eda_utils.model_recommendation(df))
print()
print("=" * 60)
print(eda_utils.labeling_recommendation())
print("=" * 60)
```

- [ ] **Step 2: Verify notebook runs clean on empty data**

```
jupyter nbconvert --to notebook --execute eda.ipynb \
    --output eda_executed.ipynb --ExecutePreprocessor.timeout=120
```

Expected: exits 0, `eda_executed.ipynb` created. Each section prints `[INFO] No data` or similar — no exceptions.

If `jupyter` is not installed: `pip install jupyter nbconvert` first.

- [ ] **Step 3: Remove the executed output file**

```
rm eda_executed.ipynb
```

- [ ] **Step 4: Commit**

```
git add eda.ipynb
git commit -m "feat(eda): add eda.ipynb — 8-section EDA notebook with heatmap and recommendations"
```

---

## Final Verification

- [ ] **Run full test suite**

```
pytest tests/ -v --tb=short
```

Expected: all tests `PASS`

- [ ] **Verify imports are clean**

```
python -c "import ghv2_viz; import eda_utils; print('OK')"
```

Expected: `OK`

- [ ] **Final commit**

```
git add -A
git commit -m "chore: final integration — GHV2 EDA pipeline complete"
```
