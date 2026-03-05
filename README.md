# 🏠 Project Glass House

> Indoor zone-level localization using Wi-Fi fingerprinting, ESP32 hardware, and a Scikit-learn classifier running on a Raspberry Pi.

**Senior Design Project — Computer Engineering**

---

## Overview

Project Glass House determines which of 9 zones in a room a person occupies — without cameras, ultrasound, or GPS. It works by fingerprinting the Wi-Fi signal environment using four ESP32 "Shouter" nodes placed at room corners and a single "Listener" ESP32 that captures RSSI (and CSI) data. A Raspberry Pi runs a trained machine learning model in real time to classify the zone.

```
[Shouter NW] ─┐
[Shouter NE] ─┤──► [Listener ESP32] ──USB──► [Raspberry Pi]
[Shouter SW] ─┤                               ├─ Preprocess
[Shouter SE] ─┘                               ├─ Train / Infer
                                               └─ Log predictions
```

---

## System Architecture

| Component | Role |
|---|---|
| 4× ESP32 Shouters | Continuously broadcast 2.4 GHz Wi-Fi beacons from room corners |
| 1× ESP32 Listener | Captures RSSI (+ CSI) from all 4 shouters, streams over USB Serial |
| Raspberry Pi | Runs the Python ML pipeline: ingest → features → predict |

### Zone Layout

Zones are numbered 1–9 in a 3×3 grid, top-left to bottom-right:

```
┌───────┬───────┬───────┐
│   1   │   2   │   3   │
├───────┼───────┼───────┤
│   4   │   5   │   6   │   ← Zone 5 (center) is hardest to classify
├───────┼───────┼───────┤
│   7   │   8   │   9   │
└───────┴───────┴───────┘
  Zone 0 = Empty room (no person present)
```

Shouters are placed at the four corners of the physical room. Each zone covers roughly 1/9 of the total floor area.

---

## Repository Structure

```
glass_house/
│
├── firmware/                  # ESP32 firmware (C/C++, Arduino framework)
│   ├── shouter/               # Broadcast logic for the 4 shouter nodes
│   └── listener/              # RSSI/CSI capture + serial transmission
│
├── ml/                        # Raspberry Pi ML pipeline (Python)
│   ├── config.py              # All paths, ports, and parameters — edit here first
│   ├── ingest.py              # Raw CSV loader and zone label parser
│   ├── features.py            # Sliding-window RSSI/CSI feature engineering
│   ├── preprocess.py          # Orchestrates ingest → features → features.csv
│   ├── train.py               # Model training, cross-validation, evaluation
│   └── inference.py           # Live serial inference and prediction logging
│
├── data/
│   ├── raw/                   # Collected run folders (gitignored — large)
│   └── processed/             # features.csv output (gitignored)
│
├── models/                    # Saved .joblib model files (gitignored)
├── logs/                      # Inference prediction logs (gitignored)
│
└── docs/
    ├── ML_Architecture.docx   # ML pipeline architecture reference
    └── TestProcess.docx       # Data collection test process document
```

---

## ML Pipeline

### Data Flow

```
data/raw/                      ← Collected by signal lead (run folders)
    └─ ingest.py               ← Parses folder names into labeled DataFrames
        └─ features.py         ← Sliding-window RSSI stats → feature vectors
            └─ preprocess.py   ← Outputs data/processed/features.csv
                └─ train.py    ← Trains classifier, saves models/glass_house_model.joblib
                    └─ inference.py  ← Loads model, reads serial, predicts live
```

### Folder Naming Convention

Raw data is organized by the collection script using a standardized naming scheme:

```
[RoomSize]Room_[GridState]_[Duration]Seconds_Run[NN]

Examples:
  24x24Room_Empty_10Seconds_Run01
  24x24Room_Grid5Occupied_10Seconds_Run03
  24x24Room_Grid1Seated_10Seconds_Run02
  24x24Room_Grid1-5Moving_30Seconds_Run01
```

### Feature Vector

Each training sample is a flat vector of RSSI statistics computed over a sliding window of readings. At minimum (RSSI only, 4 shouters × 5 statistics):

| Features | Per Shouter | Total (RSSI only) |
|---|---|---|
| mean, std, min, max, range | 5 | 20 |

> **CSI features** will be added to `features.py` once the signal lead finalizes the listener output format.

### Running the Pipeline

```bash
# 1. After data collection — build feature matrix
python ml/preprocess.py

# 2. Train the model
python ml/train.py

# 3. Run live inference on the Raspberry Pi
python ml/inference.py
```

### Model Candidates

| Model | Status | Notes |
|---|---|---|
| **Random Forest** | ✅ Recommended baseline | Fast inference, Pi-friendly, feature importances |
| KNN | Available | Simple, no training phase, slow at large dataset sizes |
| SVM | Available | Strong on clean data, no probability output by default |
| Gradient Boosting | Available | High accuracy, slower inference than RF |

Swap models by changing `SELECTED_MODEL_KEY` in `ml/train.py`.

---

## Configuration

All parameters are centralized in `ml/config.py`. Edit this file before running anything:

```python
SERIAL_PORT  = "/dev/ttyUSB0"   # USB port of Listener ESP32 on the Pi
BAUD_RATE    = 115200            # Must match ESP32 firmware
WINDOW_SIZE  = 20               # Readings per feature window — tune to sample rate
WINDOW_STEP  = 10               # Sliding window step (50% overlap)
TEST_SPLIT   = 0.20             # Held-out test fraction
CV_FOLDS     = 5                # Stratified k-fold cross-validation
```

---

## Hardware Setup

### Equipment

- 4× ESP32 development boards (Shouters)
- 1× ESP32 development board (Listener)
- 1× Raspberry Pi (3B+ or later recommended)
- USB cables + power supplies for all ESP32s
- Measuring tape + floor tape (mark 3×3 grid)

### Physical Setup

1. Measure room and mark the 3×3 grid on the floor with tape, cells labeled 1–9.
2. Mount one Shouter ESP32 at each corner at a consistent height (recommended: 1 meter).
3. Place the Listener ESP32 **outside** the grid boundary at a fixed documented position.
4. Connect the Listener to the Raspberry Pi via USB.
5. Power on all Shouters and allow **60 seconds** to stabilize before collecting data.

### Firmware

ESP32 firmware is written in C/C++ using the Arduino framework. See `firmware/shouter/` and `firmware/listener/` for build instructions.

> **CSI extraction** requires a custom ESP32 patch or library (e.g., ESP32-CSI-Tool). See `firmware/listener/README.md` for setup details.

---

## Data Collection

Data collection follows the process defined in `docs/TestProcess.docx`. Key requirements:

- **Baseline:** Minimum 20 × 10-second empty room captures
- **Per zone:** Minimum 10 runs per zone (center posture) + 5 runs for each alternate posture
- **Postures:** Occupied, Standing, Seated, Moving
- **Subjects:** At least 3 different test subjects to reduce person-specific bias
- **Rooms:** Multiple rooms recommended to improve generalization

Avoid collecting data during high Wi-Fi traffic periods. Test subjects must not carry phones or smartwatches during sessions.

---

## Open Items

| # | Item | Blocking | Owner |
|---|---|---|---|
| 1 | CSI output format from Listener ESP32 | `extract_csi_features()` in `features.py` | Signal Lead |
| 2 | Serial line format from Listener | `parse_serial_line()` in `inference.py` | Signal Lead + ML Lead |
| 3 | Final model selection | `SELECTED_MODEL_KEY` in `train.py` | ML Lead |
| 4 | `WINDOW_SIZE` tuning | Set after first data collection | ML Lead |

---

## Team

| Role | Responsibility |
|---|---|
| Signal Lead | ESP32 firmware, CSI extraction, serial output format, data collection |
| ML Lead | Feature engineering, model training, inference pipeline, Raspberry Pi deployment |
| *(+ 2 members)* | Architecture, documentation, integration, testing |

---

## Dependencies

### Raspberry Pi (Python)
```
scikit-learn
pandas
numpy
pyserial
joblib
```

Install:
```bash
pip install scikit-learn pandas numpy pyserial joblib
```

### ESP32 Firmware
- Arduino framework (via PlatformIO or Arduino IDE)
- ESP32-CSI-Tool (or equivalent CSI patch) — see `firmware/listener/`

---

## License

Academic project — Computer Engineering Senior Design. See individual source files for attribution.
