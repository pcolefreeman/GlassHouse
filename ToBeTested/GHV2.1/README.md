# GlassHouseV2.1

WiFi CSI-based occupancy detection system using ESP32 devices and a Raspberry Pi display.

---

## How It Works

- **Shouters** (ESP32) broadcast WiFi probe frames at ~50 ms intervals.
- **Listener** (ESP32) captures the incoming frames and measures the Channel State Information (CSI) for each one, then streams both over USB Serial at 921600 baud.
- **GlassHouseV2.py** reads the serial stream on your PC, correlates matched frame pairs, and saves them to a timestamped CSV.
- A trained model loaded by **InferenceV2.py** predicts occupancy in a 3×3 grid of cells.

---

## Hardware Setup

| Device | Role | Firmware |
|---|---|---|
| ESP32 (×4) | Shouter — transmits probe frames | `ShouterV2/ShouterV2.ino` |
| ESP32 (×1) | Listener — receives + streams CSI | `ListenerV2/ListenerV2.ino` |
| Raspberry Pi | Live display (heatmap) | runs `InferenceV2.py` |

Flash each shouter with `ShouterV2.ino` and the listener with `ListenerV2.ino` using the Arduino IDE. The listener connects to your PC via USB.

---

## Data Collection

### Basic capture

```bash
python GlassHouseV2.py --port COM3
```

Output is saved to `data/processed/capture_<timestamp>.csv`.

### With area dimensions (embeds in filename)

```bash
python GlassHouseV2.py --port COM3 --width 6.0 --depth 4.0
# → data/processed/capture_6.0x4.0m_2026-03-15_120000.csv
```

### With occupancy label

```bash
# Empty room baseline
python GlassHouseV2.py --port COM3 --label empty

# Single person in grid cell row 0, col 1
python GlassHouseV2.py --port COM3 --label r0c1 --row 0 --col 1

# Two people simultaneously
python GlassHouseV2.py --port COM3 --label "r0c0+r2c2"
```

Press **Ctrl+C** to stop recording.

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--port` | `COM3` | Serial port of the Listener ESP32 |
| `--output` | `data/processed/capture.csv` | Output directory/file (timestamp is appended) |
| `--label` | `unknown` | Occupancy label for this session |
| `--zone` | `0` | Zone ID metadata |
| `--row` | `0` | Grid row metadata |
| `--col` | `0` | Grid column metadata |
| `--width` | *(none)* | Area width in metres |
| `--depth` | *(none)* | Area depth in metres |

---

## Label Format

The grid is 3 rows × 3 columns. Labels use `r{row}c{col}` notation (0-indexed).

```
   col 0   col 1   col 2
row 0 [ r0c0 ] [ r0c1 ] [ r0c2 ]
row 1 [ r1c0 ] [ r1c1 ] [ r1c2 ]
row 2 [ r2c0 ] [ r2c1 ] [ r2c2 ]
```

- `empty` — no one in the area
- `r1c2` — one person in row 1, col 2
- `r0c0+r2c2` — two people (one in each cell)

Labels are **case-sensitive**.

---

## Recommended Training Collection

| Session | Command |
|---|---|
| Empty baseline | `--label empty` |
| Single person, cell (0,0) | `--label r0c0 --row 0 --col 0` |
| Single person, cell (0,1) | `--label r0c1 --row 0 --col 1` |
| *(repeat for all 9 cells)* | |
| Two-person passes | `--label "r0c0+r2c2"` |

Minimum recommended: 1 empty session + 9 single-person sessions.

---

## EDA and Analysis

Open `eda.ipynb` in Jupyter to explore collected data.

Key helper modules:

- **`eda_utils.py`** — load/validate CSVs, per-cell stats, temporal analysis, model recommendations
- **`ghv2_viz.py`** — render a 3×3 occupancy heatmap (used by both EDA and the Pi live display)
- **`csi_parser.py`** — low-level frame parsing and feature extraction (5,133 features per row)

Quick example:

```python
from eda_utils import load_csv, per_cell_stats
df, area_dims = load_csv("data/processed/capture_6.0x4.0m_2026-03-15_120000.csv")
print(per_cell_stats(df))
```

---

## Live Inference (Raspberry Pi)

```bash
# Dry-run (no model — verifies pipeline)
python InferenceV2.py --port /dev/ttyUSB0

# With trained model
python InferenceV2.py --port /dev/ttyUSB0 --model model.pkl
```

The model file must be a `joblib`-serialised sklearn pipeline.

---

## CSV Format

Each row represents one 200 ms bucket. The first 5 columns are metadata:

| Column | Description |
|---|---|
| `timestamp_ms` | Listener timestamp in milliseconds |
| `label` | Occupancy label string |
| `zone_id` | Zone metadata |
| `grid_row` | Row metadata |
| `grid_col` | Column metadata |

Followed by 5,128 CSI feature columns per shouter (amplitude, normalised amplitude, phase, SNR, phase difference across 128 subcarriers, plus RSSI and noise floor).

---

## Project Structure

```
GHV2.1/
├── GlassHouseV2.py        # Data collection entry point
├── csi_parser.py          # Frame parsing + feature extraction
├── eda_utils.py           # EDA helper functions
├── ghv2_viz.py            # Heatmap rendering
├── InferenceV2.py         # Live inference
├── eda.ipynb              # Exploratory data analysis notebook
├── ShouterV2/             # ESP32 shouter firmware (Arduino)
├── ListenerV2/            # ESP32 listener firmware (Arduino)
├── data/
│   ├── raw/               # Raw captures (if any)
│   └── processed/         # CSV files from GlassHouseV2.py
└── tests/                 # pytest test suite
```

---

## Running Tests

```bash
pytest tests/
```
