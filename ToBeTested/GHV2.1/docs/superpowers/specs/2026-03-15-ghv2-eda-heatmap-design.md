# GHV2 EDA Pipeline & Live Heatmap Display — Design Spec

**Date:** 2026-03-15
**Project:** GHV2 (GlassHouse V2) — Search & Rescue WiFi CSI System
**Status:** Approved

---

## 1. Overview

Design a parameterized Jupyter EDA notebook backed by two Python helper modules that:

1. Analyses GHV2 CSV data to characterise WiFi CSI features and drive ML model selection.
2. Provides a reusable heatmap component (`ghv2_viz.py`) whose rendering logic is designed for reuse — the EDA notebook uses it for static analysis plots, and a future Raspberry Pi live display will use the same function for real-time occupancy updates. The Pi integration itself is **out of scope** for this spec (see Section 11).

The data does not yet exist; the pipeline is built against the known schema so it is ready to run the moment a CSV is captured.

---

## 2. Context

### 2.1 Physical Setup
- **4 ESP32 shouters** placed at the four corners of the search area.
- Area dimensions vary per deployment (search & rescue, not a fixed room).
- A **3×3 grid** divides the search area into 9 zones; cells are labelled by `(grid_row, grid_col)` where both range 0–2.
- Multiple people may occupy the same or different cells simultaneously.

### 2.2 Data Schema
Each row in the CSV represents one 200 ms bucket flush from `GlassHouseV2.py`.

**Feature count derivation** (from `csi_parser.build_feature_names`):

Per prefix (e.g. `s1` or `s1_tx`):
- 128 `_amp_` + 128 `_amp_norm_` + 128 `_phase_` + 128 `_snr_` = 512
- 127 `_pdiff_` (phase-difference; abbreviated `pdiff` in column names, NOT `phase_diff`)
- 1 `_rssi` + 1 `_noise_floor` = 2
- **Total per prefix: 641**

2 prefixes per shouter (listener-rx direction: `s{id}`; shouter-tx direction: `s{id}_tx`) × 4 shouters = **5,128 feature columns**.

| Column group | Count | Description |
|---|---|---|
| Meta | 5 | `timestamp_ms`, `label`, `zone_id`, `grid_row`, `grid_col` |
| CSI features | 641 × 2 prefixes × 4 shouters = 5,128 | 512 per-subcarrier + 127 pdiff + 2 scalar, ×2 dirs ×4 shouters |
| **Total** | **5,133** | |

**Column naming convention** (matches `csi_parser.build_feature_names` exactly):
- Listener-rx direction prefix: `s{id}` → e.g. `s1_amp_0`, `s1_pdiff_0`, `s1_rssi`
- Shouter-tx direction prefix: `s{id}_tx` → e.g. `s1_tx_amp_0`, `s1_tx_pdiff_0`, `s1_tx_rssi`
- Note: phase-difference columns use the abbreviated suffix `_pdiff_`, not `_phase_diff_`

### 2.3 ML Task
9 independent binary classifiers — one per grid cell — each predicting "person present in this cell?". This framing handles multi-occupancy naturally.

---

## 3. File Structure

```
GHV2/
├── eda.ipynb            # Primary deliverable — EDA notebook (GitHub-renderable)
├── eda_utils.py         # EDA-specific logic: loaders, stats, recommendations
├── ghv2_viz.py          # Heatmap rendering — reusable by notebook and future Pi display
├── InferenceV2.py       # Existing — unchanged by this spec
├── GlassHouseV2.py      # Existing — gains --width / --depth args + filename update
└── data/
    └── processed/
        └── capture_{W}x{D}m_{YYYY-MM-DD_HHMMSS}.csv
```

---

## 4. GlassHouseV2.py Changes

### 4.1 New CLI Arguments
```
--width   float   Width of search area in metres (default: None)
--depth   float   Depth of search area in metres (default: None)
```

The existing `--label` argument already accepts any string value and writes it verbatim to the CSV. Compound multi-person labels (e.g. `--label "r0c0+r2c2"`) are therefore already supported without code changes — the operator simply passes the compound string. No validation of label format is added to `GlassHouseV2.py`; format correctness is the operator's responsibility at collection time.

### 4.2 Output Filename Convention
When `--width` and `--depth` are provided:
```
capture_{W}x{D}m_{YYYY-MM-DD_HHMMSS}.csv
e.g. capture_6.0x4.0m_2026-03-15_143022.csv
```
When omitted, falls back to `capture_{YYYY-MM-DD_HHMMSS}.csv` (no dimensions embedded).

---

## 5. `ghv2_viz.py`

Single public function:

```python
from typing import Optional
import matplotlib.figure

def render_heatmap(
    grid_values,        # np.ndarray shape (3, 3); see mode parameter for expected range
    area_dims,          # tuple (width_m, depth_m) or (None, None) → see §5.3
    title,              # str
    mode="confidence",  # str: "confidence" | "raw" — controls rendering path
    threshold=0.70,     # float: only used when mode="confidence"
    cmap_lit="#FF6B35", # str: rescue orange; only used when mode="confidence"
    cmap_raw="YlOrRd",  # str: colormap for mode="raw" (RSSI, counts, etc.)
    ax=None,            # None → creates + returns new Figure
                        # provided → redraws in-place, returns None
):
    # type: (...) -> Optional[matplotlib.figure.Figure]
```

**Return value:**
- `ax=None`: creates a new `Figure`, returns it (EDA use — notebook captures and displays it).
- `ax` provided: clears and redraws into the supplied axes in-place, returns `None` (live Pi animation use — caller owns the Figure).

**Python compatibility:** Use `Optional[matplotlib.figure.Figure]` from `typing` (not `Figure | None`) to remain compatible with Python 3.8+.

### 5.1 Rendering Modes

**`mode="confidence"` — live occupancy display (threshold + glow):**

| Confidence | Display |
|---|---|
| `< threshold` | Near-black fill (cell dark / "clear") |
| `≥ threshold` | Solid rescue-orange fill (`#FF6B35`). When strictly above `threshold`, a glow overlay is **also** applied simultaneously: `glow_alpha = (confidence − threshold) / (1 − threshold)` — subtle, does not change fill colour. At exactly `threshold`, glow_alpha = 0 (no glow). Both solid fill and glow apply together for any confidence > threshold. |

Cell annotations show confidence as a percentage (e.g. `"83%"`).

**`mode="raw"` — EDA analysis plots (RSSI, counts, arbitrary floats):**

`grid_values` may hold any float range (e.g. negative dBm RSSI or positive integer counts). Values are normalised internally to `[0, 1]` using `(v − min) / (max − min + ε)` (where ε = 1e-9 prevents division by zero) before applying `cmap_raw`. No threshold logic is applied. Cell annotations show the raw value (formatted to 1 decimal place).

### 5.2 Cell Annotations
- Value printed inside each cell (format depends on mode — see §5.1).
- Cell boundaries clearly marked.
- Title shows metric and timestamp.

### 5.3 Axis Labels
- Tick convention: **cell-centre labels** (standard for heatmaps). For 3 columns of `cell_width` metres each, x-axis tick positions are `[cell_width/2, 3×cell_width/2, 5×cell_width/2]` with labels `["0 – {cell_width:.1f} m", "{cell_width:.1f} – {2×cell_width:.1f} m", ...]`. Y-axis similarly.
- When `area_dims` is `(None, None)`: axes show cell indices `[0, 1, 2]` with axis label "Cell index (col)" / "Cell index (row)".

---

## 6. `eda_utils.py`

### 6.1 Dimension Parsing
```python
def parse_dimensions(filename):
    # type: (str) -> tuple  # (float, float) or (None, None)
    """Extract (width_m, depth_m) from filename via regex r'(\d+\.?\d*)x(\d+\.?\d*)m'.
    Returns (None, None) if pattern not found.
    """
```

### 6.2 Grid Cell Calculation
```
cell_width  = total_width  / 3
cell_depth  = total_depth  / 3
cell (row, col) covers:
  x ∈ [col * cell_width,  (col+1) * cell_width]
  y ∈ [row * cell_depth,  (row+1) * cell_depth]
```
Only computed when `area_dims` is not `(None, None)`.

### 6.3 Column Grouping

Column prefixes follow `csi_parser` naming exactly:

```python
def group_columns(df) -> dict:
    """Returns {
        "meta":  ["timestamp_ms", "label", "zone_id", "grid_row", "grid_col"],
        "s1":    [col for col in df.columns if re.match(r'^s1_(?!tx_)', col)],
        "s1_tx": [col for col in df.columns if col.startswith("s1_tx_")],
        "s2":    [col for col in df.columns if re.match(r'^s2_(?!tx_)', col)],
        "s2_tx": [...],
        "s3":    [...],  "s3_tx": [...],
        "s4":    [...],  "s4_tx": [...],
    }
    # Pattern r'^s{id}_(?!tx_)' uses a negative lookahead to exclude s{id}_tx_ columns
    # from the listener-rx group. This is robust to any future two-digit shouter IDs.
    """
```

All notebook sections operate on one group at a time — no cell ever processes all 5,133 columns simultaneously.

### 6.4 Label Parser (Multi-Person Support)
```python
def parse_label(label: str, n_cells: int = 9, row_context: str = "") -> np.ndarray:
    """Decode compound label into (9,) binary target vector.
    Cell index = grid_row * 3 + grid_col.
    Examples:
      "r0c1"              → [0,1,0, 0,0,0, 0,0,0]
      "r0c0+r2c2"         → [1,0,0, 0,0,0, 0,0,1]
      "empty"             → [0,0,0, 0,0,0, 0,0,0]
      unrecognised string → [0,0,0, 0,0,0, 0,0,0]  (treated as empty)
         WARNING printed: f"Unrecognised label '{label}' at {row_context} — treated as empty"
         row_context is a caller-supplied string (e.g. "row 42, ts=12345ms") for debugging.
    """
```

Multi-person label format: `"r{row_a}c{col_a}+r{row_b}c{col_b}"` where `row_a`, `col_a`, `row_b`, `col_b` are integer digits 0–2. More than two people: extend with additional `+r{row}c{col}` segments. Labels are **case-sensitive** — `"r0C1"` is unrecognised and will print a warning.

### 6.5 CSV Loading & Validation

```python
def load_csv(path, manual_dims=None):
    # type: (str, Optional[tuple]) -> tuple  # (pd.DataFrame, tuple)
    """Load and validate the GHV2 CSV.

    Validation rules:
    - File must exist and be readable; raises FileNotFoundError otherwise.
    - CSV must contain at minimum the 5 meta columns
      ("timestamp_ms","label","zone_id","grid_row","grid_col");
      raises ValueError if any are missing.
    - If data has zero rows (header only), returns empty DataFrame with a warning
      printed to stdout — does not raise, allows the notebook to report "no data yet".
    - If actual column count differs from expected 5,133, prints a warning with the
      discrepancy (e.g. captured with fewer shouters) but does not raise.

    Dimension resolution order:
    1. manual_dims if not None  (from notebook Config cell)
    2. parse_dimensions(os.path.basename(path))
    3. (None, None) if neither yields a result

    Returns (df, area_dims).
    """
```

### 6.6 EDA Functions

| Function | Output |
|---|---|
| `load_csv(path, manual_dims)` | `(DataFrame, area_dims)` — see §6.5 |
| `describe_dataset(df)` | Shape, dtypes, missing % per column group |
| `outlier_summary(df, groups)` | IQR outlier counts per shouter group |
| `temporal_stats(df)` | Sampling rate (mean/std of `timestamp_ms` diffs), gap list |
| `per_cell_stats(df)` | Count and mean RSSI per `(grid_row, grid_col)`; RSSI = mean of `s1_rssi`, `s2_rssi`, `s3_rssi`, `s4_rssi` (listener-rx direction only) averaged across all 4 shouters per row |
| `correlation_matrix(df, group)` | Correlation DataFrame for one column group |
| `phase_polar_data(df, group)` | Phase values array for polar histogram (one group) |
| `model_recommendation(df)` | Formatted string — data-driven guidance |
| `labeling_recommendation()` | Formatted string — multi-person strategy |

**Gap detection rule** (used by `temporal_stats`):
The DataFrame is sorted by `timestamp_ms` before analysis (CSVWriter iterates a `set`, so row order is not guaranteed to be chronological). A gap is any consecutive pair of sorted rows where `timestamp_ms[i+1] − timestamp_ms[i] > 2 × BUCKET_MS` (i.e. > 400 ms, indicating at least 1 missed bucket). Returns list of `(timestamp_ms, gap_duration_ms)` tuples.

---

## 7. `eda.ipynb` — Notebook Structure

| Section | Content |
|---|---|
| **0. Config** | `CSV_PATH`, `MANUAL_DIMS = (None, None)`, `CONFIDENCE_THRESHOLD = 0.70`, `PLOT_DPI = 150` |
| **1. Data Loading & Schema** | Load CSV via `load_csv`, print shape, dtypes, first 5 rows, parsed dimensions |
| **2. Statistical Summary** | Descriptives table, missing values matrix, outlier counts per group |
| **3. Temporal Analysis** | RSSI time series ×4 shouters, sampling rate stats, gap list |
| **4. Spatial Analysis** | Per-cell sample counts, mean RSSI per cell, coverage scatter |
| **5. Feature Analysis** | Box plots (amplitude/phase/SNR per group), correlation heatmap |
| **6. 3×3 Heatmap** | Mean RSSI heatmap (`mode="raw"`) + sample-count heatmap (`mode="raw"`) via `ghv2_viz.render_heatmap` |
| **7. Pairwise Relationships** | Scatter matrix for scalar features (RSSI ×4 shouters, noise floor ×4 shouters) |
| **8. Recommendations** | Labeling strategy + model selection guidance |

### 7.1 Visualizations Produced

| # | Notebook Section | Plot | Function | Mode |
|---|---|---|---|---|
| 1 | §6 3×3 Heatmap | 3×3 heatmap — mean RSSI per cell | `ghv2_viz.render_heatmap` | `"raw"` |
| 2 | §6 3×3 Heatmap | 3×3 heatmap — sample count per cell | `ghv2_viz.render_heatmap` | `"raw"` |
| 3 | §5 Feature Analysis | Per-shouter amplitude box plots | `eda_utils` | — |
| 4 | §5 Feature Analysis | Correlation heatmap (scalar features) | `eda_utils` | — |
| 5 | §3 Temporal Analysis | RSSI time series ×4 shouters | `eda_utils` | — |
| 6 | §2 Statistical Summary | Missing value matrix | `eda_utils` | — |
| 7 | §4 Spatial Analysis | Per-cell sample count bar chart | `eda_utils` | — |
| 8 | §5 Feature Analysis | Phase polar histograms ×4 shouters | `eda_utils.phase_polar_data` | — |
| 9 | §7 Pairwise Relationships | Pairwise scatter (RSSI + noise floor) | `eda_utils` | — |

---

## 8. Labeling Strategy Recommendation

Captured in `eda_utils.labeling_recommendation()` and printed in Section 8 of the notebook:

- **Empty pass:** label = `"empty"` — one session, area clear.
- **Single-person passes:** label = `"r{row}c{col}"` — one session per cell (9 total).
- **Multi-person passes:** label = `"r{row_a}c{col_a}+r{row_b}c{col_b}"` — one session per occupied combination of interest.
- **Target encoding:** `parse_label()` decodes any label into a `(9,)` binary vector for training 9 independent binary classifiers.

---

## 9. Model Selection Guidance

Included in `eda_utils.model_recommendation(df)`, data-driven (adjusts messaging based on class balance and missing-value stats found in the actual data). **When `df` is empty (zero rows):** returns a fixed fallback string: `"No data available yet — recommendations will be generated once a CSV capture is loaded."` No analysis is attempted on an empty DataFrame.

| Step | Approach | Rationale |
|---|---|---|
| Dimensionality reduction | PCA per shouter group | 5,128 feature columns → manageable; per-group preserves spatial meaning |
| Per-cell classifier | Random Forest | Handles high-dim CSI, robust to noise, built-in feature importance |
| Baseline comparison | SVM (RBF kernel) | Strong precedent in WiFi CSI fingerprinting literature |
| Deployment format | `sklearn` pipeline → `joblib` `.pkl` | Already supported by `InferenceV2.load_model()` |

---

## 10. Dependencies

```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn   # for PCA / model recommendation examples
joblib         # model serialisation (already in InferenceV2)
```

No new dependencies beyond what a standard data science environment provides.

---

## 11. Future Considerations

- **Automatic area dimension detection:** In a future protocol extension, shouters should broadcast their inter-shouter distances (or absolute positions) to the Raspberry Pi via a new packet type or an extended `hello_pkt_t` field (e.g. `distance_cm`). The Pi live display would consume this and pass `area_dims` to `render_heatmap` automatically, removing the need for manual `--width`/`--depth` input. The operator would see real physical measurements on the heatmap edges without any configuration. The `MANUAL_DIMS` override in `eda.ipynb` and the `ghv2_viz.render_heatmap(area_dims=...)` interface are already structured to accept this when ready.

---

## 12. Out of Scope

- Training the ML models (EDA informs selection; training is a separate phase).
- Raspberry Pi live display driver / window management — `ghv2_viz.render_heatmap` is designed for reuse there (`ax` parameter, `mode="confidence"`, return-`None` path), but the Pi integration (animation loop, display window, inference wiring) is a future spec.
- Changes to `InferenceV2.py` — unchanged by this spec.
- Real-time serial ingestion in the notebook (notebook is post-collection analysis only).
