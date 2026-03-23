# ghv4 Python Package — Conventions & Gotchas
<!-- last verified: 2026-03-22 -->

## Constants Rule
All cross-module constants live in `ghv4/config.py` — the single source of truth.
If a value is used in more than one file, it belongs in config.py.
Module-private constants (used in only one file) stay local.
This pattern MUST be maintained for all future GlassHouse work.

## Data & Labels
- Bucket size: 200 ms (`ghv4.config.BUCKET_MS`)
- 9 class labels: `r0c0` … `r2c2` (row-major, `ghv4.config.CELL_LABELS`)
- Multi-cell labels supported: `r0c0+r2c2` etc. → multiple hot columns in y matrix
- `y` is an (N, 9) binary matrix from `ghv4.eda_utils.parse_label()`
- `spacing.json` format: `{"pairs": {"1-2": {"distance_m": 1.5}, ...}}`; 6 pairs

## GUI Port Selection
The executable has three COM port dropdowns (each with a Refresh button):
- **Serial Port** — main data capture
- **Listener Port** — listener debug monitor
- **Shouter Port** — shouter debug monitor

Refresh enumerates live ports via `serial.tools.list_ports`. COM3 is the
placeholder default; the user selects the actual port before connecting.

## ML Distance Estimation (implemented 2026-03-19)
- Spec/plan: `docs/superpowers/plans/2026-03-18-ml-distance-estimation.md`
- New files: `ghv4/distance_features.py`, `ghv4/distance_preprocess.py`, `ghv4/distance_train.py`, `ghv4/distance_inference.py`
- Entry points: `run_distance_preprocess.py`, `run_distance_train.py`
- Data: `distance_data/raw/` (CSVs), `distance_data/processed/` (numpy), `distance_models/` (pkl)
- 6 per-pair GBT/RF regressors, max 200 trees each, trained on PC, inference on Pi 4B
- Calibration phase: 30s snap collection → median prediction → spacing.json

## Pi LCD Display (implemented 2026-03-22)
- Module: `ghv4/pi_display.py` — pygame-based 3×3 grid operator display
- Entry point: `run_pi_display.py` (--port, --model, --fullscreen, --demo)
- InferenceThread reuses `inference.py` functions (load_model, load_spacing, etc.)
- DemoThread cycles cells without hardware for testing
- Colors match `viz.py` confidence mode (#FF6B35 rescue orange, #0d0d0d dark bg)
- Pi deployment: `pip install pygame>=2.5.0`; for headless use `SDL_VIDEODRIVER=kmsdrm`

## CSI Breathing Detection (implemented 2026-03-23)
- Spec: `docs/superpowers/specs/2026-03-23-csi-breathing-detection-design.md`
- Plan: `docs/superpowers/plans/2026-03-23-csi-breathing-detection.md` (9 tasks, 34 tests)
- Files: `ghv4/breathing.py`, `run_sar.py`, `tests/test_breathing.py`
- `_parse_csi_bytes` renamed to `parse_csi_bytes` (public API)
- Pure signal processing (no ML) — CSI ratio + FFT for zero-calibration human presence detection
- Console display only (`--display console`); `--display pygame` accepted but not implemented
- **Next step**: switch from shouter→listener paths to shouter↔shouter paths (CSI_SNAP `[0xEE][0xFF]` frames) for full 9-cell coverage with listener outside zone

## Gotchas

- **Inference requires preprocessing artifacts** — `inference.py` loads `feature_names.txt` and
  `scaler.pkl` from `--processed-dir` (default `data/processed/`). Without these, predictions
  use raw unscaled features. The `apply_preprocessing()` function mirrors `preprocess.py`'s
  column-drop + StandardScaler + phase/π transforms.
- **`_shouter_states` location** — lives on `ListenerDebugTab` in `ghv4/ui/debug_tab.py`,
  protected by `threading.Lock()`. The original spec incorrectly placed it in `spacing_tab.py`.
- **Garbled binary in Listener/Shouter log** — caused by ESP32 bootloader bytes at startup and CSI frame
  data leaking through the text filter. Current filter (`any byte > 0x7E → drop`) only blocks high bytes;
  CSI int8 amplitudes (0x00–0x7E) pass through as fake ASCII lines. The
  `if not line.lstrip('\r').startswith('['):` guard in `ListenerDebugThread._read_one` (in
  `ghv4/ui/debug_tab.py`) filters these — all legitimate `[LST]` lines start with `[`.
  ShouterDebugThread is unaffected (no binary frames on shouter serial).
- **SHOUTER DISTANCES cards stay `--` (Python cause)** —
  (1) `MIN_SAMPLES` must be 1 (already set).
  See also: firmware causes (2, 3) in `firmware/CLAUDE.md`.
- `NULL_PDIFF_INDICES` in `ghv4/config.py` (`{0,1,2,31,32,62,63,64,65}`) differs slightly from
  `NULL_SUBCARRIER_INDICES` — intentional
- `models/` directory is not created automatically — `ghv4/train.py` assumes it exists
- `ghv4/train.py` default `PROCESSED_DIR` points to `data/processed/` — pass
  `--processed-dir` for specific sessions
- `snr` column is dropped as collinear with `rssi` and `noise_floor`
- **`np.load` on string arrays needs `allow_pickle=True`** — session-ID groups saved as
  object-dtype `.npy` fail without it (e.g., `distance_train.py` loading `groups.npy`).
- **StandardScaler for distance models fits on 242 amp columns** — indices `[0:121]` (fwd
  amp_norm) + `[242:363]` (rev amp_norm) out of 484 total features. Test fixtures must
  match this shape or scaler transform will raise.
- **`META_COLS` must include all non-feature columns** — any new metadata column in CSVs
  (e.g. `activity`) must be added to `META_COLS` in `config.py` or preprocessing crashes
  with `ValueError: could not convert string to float`
- **`--raw-dir` must point to room subdirectory** — e.g. `test_coll_1/room2102/`, not `test_coll_1/`;
  the loader does not recurse into subdirectories
- **Training is slow with large feature sets** — 35K rows × 2,900 features: SVM takes 5-20 min,
  stacking even longer. Use `--model rf --skip-cv` to skip CV entirely, or `--model rf --fast` for quick iteration
- **Breathing replay requires raw CSI** — ML training CSVs store normalized `amp_norm`, destroying
  amplitude relationships needed for breathing FFT. Use `--port` (live serial) for real testing.
- **Breathing path map needs shouter-to-shouter CSI** — current `BREATHING_PATH_MAP` uses only
  shouter→listener paths (4 paths, 5 cells). Shouter↔shouter CSI already in firmware (`[0xEE][0xFF]`
  snap frames) but Python doesn't consume them yet. `SerialReader` parses snap frames but
  routes them only to `_music_estimator`/`_snap_callback`, NOT to `frame_queue`. Switching
  to 6 S↔S paths covers all 9 cells. Listener should be placed outside the grid zone.
  Firmware change required first: continuous beacons (spec written 2026-03-23).
- **debug_tab text parsing must match firmware text** — `ListenerDebugThread._read_one`
  parses `[LST]` text with regex (`_HELLO_RE`) and string matching (`'starting ranging'`).
  When firmware `Serial.printf` format changes, update `debug_tab.py` to match. Fixed 2026-03-22:
  HELLO regex updated for `(MAC-assigned)` insertion; ranging detection updated for new text.
