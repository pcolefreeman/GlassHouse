# ghv4 Python Package — Conventions & Gotchas
<!-- last verified: 2026-03-25 -->

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

## CSI Breathing Detection (implemented 2026-03-23, updated 2026-03-24)
- Spec: `docs/superpowers/specs/2026-03-24-continuous-snap-breathing-design.md`
- Plan: `docs/superpowers/plans/2026-03-24-continuous-snap-breathing.md` (10 tasks)
- Files: `ghv4/breathing.py`, `run_sar.py`, `tests/test_breathing.py`
- `_parse_csi_bytes` renamed to `parse_csi_bytes` (public API)
- Pure signal processing (no ML) — CSI ratio + FFT for zero-calibration human presence detection
- Uses shouter↔shouter CSI_SNAP frames (`[0xEE][0xFF]`), 6 paths covering all 9 grid cells
- `BREATHING_PATH_MAP` uses `(min_id, max_id)` tuple keys for undirected shouter pairs
- `BREATHING_SNAP_HZ=20`, `BREATHING_WINDOW_N=600` (20 Hz × 30s), `BREATHING_SLIDE_N=20`
- `SerialReader` enqueues snap frames as `('csi_snap', frame)` to `frame_queue`
- `BreathingDetector.feed_frame('csi_snap', frame)` with canonical `(min,max)` key routing
- Pygame heatmap display via `BreathingDisplay` + `BreathingThread`/`SARDemoThread`
- `run_sar.py` supports `--demo`, `--fullscreen`, `--display pygame|console`
- Snap frame dict key is `'csi'` (raw bytes), NOT `'csi_bytes'` (shouter frame key) — easy to confuse
- CSV replay removed from `run_sar.py` (ML CSVs lack raw CSI snap data)
- **Listener must stay inside the room** — moving listener outside behind a closed door causes ALL paths to show high variance simultaneously (empty room, no people). Likely cause: wall/door degrades WiFi AP signal to shouters, making CSI measurements unstable across all paths. Confirmed 2026-03-24.
- **Listener proximity to a path saturates `var_conf`** — when listener is inside but physically near a shouter pair path, its WiFi AP beacons act as a static RF scatterer on that path, driving `var_conf` to 0.99+. S1↔S4 was consistently saturated with listener near that wall. Not a board defect.
- **Deployment constraint** — listener must be inside the room, stationary, positioned away from all shouter pair paths during scans. Operator holding the listener must stand still.
- **A+C scoring implemented 2026-03-26** — replaces PCA Approach B (absolute gate + sigmoid). Two zero-calibration signals combined:
  - **(A) Inter-path contrast**: `_raw_amplitude_energy(window)` returns raw `snr_eig` (PCA eigenvalue ratio, no gate). `get_grid_scores()` normalises each path by `median(all_snr_eig)`. Contrast > 1 means path is elevated above group. `BREATHING_CONTRAST_CEILING=3.0` maps contrast to 0–1. Requires `BREATHING_MIN_PATHS_FOR_CONTRAST=3` ready paths; fewer → falls back to phase only.
  - **(C) Phase-based CSI ratio**: `_phase_score(window)` wires `CSIRatioExtractor` + `BreathingAnalyzer` (conjugate-multiply subcarrier pairs → FFT → breathing-band fraction). Returns 0–1 confidence. Phase is physically specific to path-length oscillation (breathing).
  - **Combined**: `confidence = max(contrast_score, phase_score)` per path. Either strong contrast OR strong phase triggers detection.
- **Why A+C works across rooms** — empty room: all paths have similar snr_eig (contrast ≈ 1 → 0 confidence). Person: nearby paths elevated 2–20× above others → high contrast. Different rooms: baseline may differ but ratio self-normalises. No absolute gate, no room calibration.
- **`BREATHING_SNR_GATE` removed** — replaced by inter-path contrast. No absolute thresholds remain in the scoring pipeline.
- **`_amplitude_score()` removed** — split into `_raw_amplitude_energy()` (returns raw snr_eig) and `_phase_score()` (returns CSI ratio phase confidence). `get_grid_scores()` combines both.
- **`run_sar.py --log-level DEBUG`** — now shows `snr_eig=`, `contrast=`, `phase=` per path per update cycle. Use to validate A+C behaviour on new hardware.
- **`_last_path_conf` cache** — `BreathingDetector` stores path confidences from last `get_grid_scores()` call. `BreathingThread` and console loop read this instead of recomputing.
- **S2↔S4 and S3↔S4 have low snap rates** (~1–4/s vs 8–19/s for S1 paths) — buffers may not fill reliably for these paths in current hardware config. With `BREATHING_MIN_PATHS_FOR_CONTRAST=3`, contrast scoring activates even if only 3 of 6 paths are ready.
- **Deployment constraint** — listener must be inside the room, stationary, positioned away from all shouter pair paths during scans. Operator holding the listener must stand still.

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
- **S1↔S4 and S2↔S3 have high background variance even in empty rooms** — confirmed 2026-03-24 with listener outside room. Cause unknown (may be hardware noise, multipath from walls, or listener AP signal attenuation through wall). These paths cannot be discriminated by variance alone. FFT-only detection is immune to this — steady-state noise has no 0.1–0.5 Hz periodicity. S3↔S4 path physically blocked by rubble — low snap rate expected.
- **Breathing requires raw CSI snap data** — ML training CSVs store normalized `amp_norm`, destroying
  amplitude relationships needed for breathing FFT. Use `--port` (live serial) or `--demo` mode.
- **debug_tab text parsing must match firmware text** — `ListenerDebugThread._read_one`
  parses `[LST]` text with regex (`_HELLO_RE`) and string matching (`'starting ranging'`).
  When firmware `Serial.printf` format changes, update `debug_tab.py` to match. Fixed 2026-03-22:
  HELLO regex updated for `(MAC-assigned)` insertion; ranging detection updated for new text.
