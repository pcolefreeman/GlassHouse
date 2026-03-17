# GlassHouseV3 — WiFi CSI Indoor Positioning

Senior capstone @ Georgia Southern University. Four ESP32 shouters at room
corners + one listener ESP32 collect CSI/RSSI across a 3×3 grid. A
scikit-learn classifier maps each 200 ms bucket of signal data to a grid cell.

## Version Control
This project is NOT a git repository. Do not attempt git commands (status, commit, diff, etc.).

## Hardware
- **Listener + Shouters**: ESP32-WROOM-UE (devkitC); baud rate 921600
- Raspberry Pi 4B — inference/deployment only; never used for training

## Commands
```bash
# Build distribution folder (exe + data dirs + README)
# Output: GHV3_Distro/GHV3_Collector.exe  +  data/raw/  data/processed/  README.txt
# Steps: pyinstaller GHV3_Collector.spec → copy dist/GHV3_Collector.exe → create GHV3_Distro/ tree

# Run GUI (select COM port via dropdown, click Refresh to enumerate ports)
python run_gui.py

# Build executable — run whenever any source file changes
pyinstaller GHV3_Collector.spec

# Preprocess raw CSVs → X.npy / y.npy / scaler.pkl
python run_preprocess.py
python run_preprocess.py --raw-dir data/raw/SESSION --out-dir data/processed/SESSION

# Train (always on PC)
python run_train.py                        # full pipeline (LogReg/KNN/SVM/GBT/RF + vote + stack)
python run_train.py --fast                 # lighter models optimized for Pi inference (train on PC)
python run_train.py --model rf             # force a specific model

# Live inference
python run_inference.py --port COM3 --model models/model.pkl --spacing data/raw/spacing.json
python run_inference.py --port COM3        # dry-run (no model needed)

# Tests
pytest tests/
```

## Architecture

### PC Software (Python — `ghv3_1/` package)
```
run_gui.py             — entry point: launches GUI
run_inference.py       — entry point: live inference
run_preprocess.py      — entry point: preprocessing pipeline
run_train.py           — entry point: model training

ghv3_1/
  __init__.py          — package root (__version__ = "3.1.0")
  config.py            — single source of truth for ALL cross-module constants
  csi_parser.py        — frame structs, feature extraction (shared by all modules)
  serial_io.py         — SerialReader (byte stream → frame_queue) + CSVWriter (frame_queue → CSV)
  spacing_estimator.py — consumes [0xCC][0xDD] ranging frames → EMA RSSI → spacing.json
  preprocess.py        — raw CSV → X.npy, y.npy, feature_names.txt, scaler.pkl
  train.py             — CV comparison → VotingClassifier → StackingClassifier → saves best
  inference.py         — live serial → feature extraction → model.predict()
  eda_utils.py         — label parsing, shared EDA helpers
  viz.py               — visualization widgets (heatmap, spacing overlay)
  cell_logic.py        — pure label/cell helper functions
  ui/
    __init__.py        — UI subpackage
    app.py             — application shell: window, tab routing, crash logger
    capture_tab.py     — data collection tab + BackgroundCaptureThread
    debug_tab.py       — ListenerDebugTab + ShouterDebugTab + debug threads
    spacing_tab.py     — SpacingCards distance display widget
    widgets.py         — PortDropdown, LogPanel, StatusLabel reusable components

tests/                 — pytest suite (conftest.py + per-module test files)
tools/                 — standalone utilities (log_listener, serial_frame_checker, poll_simulator)
data/raw/              — session CSVs + spacing.json per session
data/processed/        — X.npy, y.npy, scaler.pkl, feature_names.txt
models/                — saved model .pkl files (must be created manually)
GHV3_Distro/           — packaged distribution (exe + data dirs + README)
```

### ESP32 Firmware (Arduino/C++)
```
firmware/
  GHV3Protocol.h       — shared packet structs + magic byte constants (single copy)
  ListenerV3/
    ListenerV3.ino     — listener ESP32 firmware
                           WiFi AP (SSID: CSI_PRIVATE_AP, ch 6)
                           UDP server on port 3333; polls shouters on port 3334
                           CSI ring buffer (ISR → task) → emit_listener_frame [0xAA][0x55]
                           Handles HELLO, RANGE_RPT, RESP UDP packets
                           emit_shouter_frame [0xBB][0xDD] for each poll hit/miss
                           emit_ranging_frame [0xCC][0xDD] when ranging_rpt received
                           run_ranging_phase() — one-shot after all 4 shouters register
                           Text output: [LST] prefixed lines at 921600 baud
  ShouterV3/
    ShouterV3.ino      — shouter ESP32 firmware
                           Connects to listener AP as WiFi STA
                           Sends [BB][FA] HELLO on connect, re-sends on reconnect
                           Responds to polls ([BB][CC]) with CSI response ([BB][EE])
                           During ranging: beacons when instructed, reports RSSI as [BB][A3]
                           CSI ring buffer (ISR → task) → included in SHOUT response
                           Text output: [SHT] prefixed lines at 921600 baud (text only, no binary frames on serial)
```

## Constants Rule
All cross-module constants live in `ghv3_1/config.py` — the single source of truth.
If a value is used in more than one file, it belongs in config.py.
Module-private constants (used in only one file) stay local.
This pattern MUST be maintained for all future GlassHouse work.

### Serial Frame Types (listener COM port → PC)
```
[0xAA][0x55]  — listener CSI frame   magic(2) + 20-byte header + csi[N]
[0xBB][0xDD]  — shouter poll frame   magic(2) + 29-byte header + csi[N]
[0xCC][0xDD]  — ranging report       magic(2) + 12-byte payload (fixed, no CSI)
[0xEE][0xFF]  — CSI snapshot fwd     magic(2) + csi_snap_pkt_t payload (reporter, peer, seq, csi[N])
text          — [LST] debug lines    pure ASCII, newline-terminated
```

### UDP Packet Types (WiFi, not serial)
```
[BB][FA]  hello_pkt_t       shouter → listener  (10 bytes)
[BB][CC]  poll_pkt_t        listener → shouter  (108 bytes, includes 96-byte pad)
[BB][EE]  response_pkt_t    shouter → listener  (404 bytes, includes up to 384-byte CSI)
[BB][A0]  peer_info_pkt_t   listener → shouters (32 bytes, during ranging)
[BB][A1]  range_req_pkt_t   listener → shouter  (7 bytes, per-beacon-round)
[BB][A2]  range_bcn_pkt_t   shouter → broadcast (8 bytes, during ranging)
[BB][A3]  ranging_rpt_pkt_t shouter → listener  (14 bytes, RSSI report)
[BB][A4]  csi_snap_pkt_t    shouter → listener  (up to 392 bytes, one per CSI snapshot)
```

## GUI Port Selection
The executable has three COM port dropdowns (each with a Refresh button):
- **Serial Port** — main data capture
- **Listener Port** — listener debug monitor
- **Shouter Port** — shouter debug monitor

Refresh enumerates live ports via `serial.tools.list_ports`. COM3 is the
placeholder default; the user selects the actual port before connecting.

## Executable Rebuild Rule
**Rebuild the executable after any change to any file in `ghv3_1/` or `run_gui.py`:**
```bash
pyinstaller GHV3_Collector.spec
```

`GHV3_Collector.spec` was adapted from the V2 spec (`GHV2_Collector.spec` in the V2
project). Use that as the reference if the spec needs to be regenerated.

## Frame Protocol
- `[0xAA][0x55]` — listener frame (20-byte header after magic)
- `[0xBB][0xDD]` — shouter frame (29-byte header after magic)
- `[0xCC][0xDD]` — ranging frame (routed to SpacingEstimator)
- 128 subcarriers; **null indices `{0,1,2,32,63,64,65}` must be dropped** before feature extraction

## Data & Labels
- Bucket size: 200 ms (`ghv3_1.config.BUCKET_MS`)
- 9 class labels: `r0c0` … `r2c2` (row-major, `ghv3_1.config.CELL_LABELS`)
- `y` is an (N, 9) binary matrix from `ghv3_1.eda_utils.parse_label()`
- `spacing.json` format: `{"pairs": {"1-2": {"distance_m": 1.5}, ...}}`; 6 pairs

## Gotchas
- **`_shouter_states` location** — lives on `ListenerDebugTab` in `ghv3_1/ui/debug_tab.py`,
  protected by `threading.Lock()`. The original spec incorrectly placed it in `spacing_tab.py`.
- **`shouter_csi_cb` MAC matching always fails in STA mode** — `wifi_csi_info_t.mac` is the
  AP/listener BSSID, not the transmitting shouter's MAC. Peer RSSI is captured in
  `on_esp_now_recv` via `recv_info->rx_ctrl->rssi` — true P2P RSSI, not AP-relayed.
  Dispatch order in `loop()`: `[0xA0]` PEER_INFO → `[0xA1]` RANGE_REQ → `[0xBB][0xCC]` POLL.
  The `[0xA2]` UDP RANGE_BCN handler was removed (2026-03-16) when ESP-NOW replaced UDP beacons.
- **`GHV3_Distro/` not auto-updated** — `pyinstaller` only writes to `dist/GHV3_Collector.exe`;
  must manually copy to `GHV3_Distro/GHV3_Collector.exe` after each rebuild.
- **`ranging_done` resets on shouter disconnect** — after the 2026-03-16 firmware fix, the
  listener fires `ARDUINO_EVENT_WIFI_AP_STADISCONNECTED` on disconnect, sets `ranging_done = false`,
  and re-ranges once all 4 reconnect. `ranging_done` is `static volatile bool` at file scope
  (not `static bool` inside `loop()`). If re-ranging never fires, check the event handler is
  registered in `setup()` and that `shouter_mac[]` was populated by a prior HELLO.
- **Garbled binary in Listener/Shouter log** — caused by ESP32 bootloader bytes at startup and CSI frame
  data leaking through the text filter. Current filter (`any byte > 0x7E → drop`) only blocks high bytes;
  CSI int8 amplitudes (0x00–0x7E) pass through as fake ASCII lines. The
  `if not line.lstrip('\r').startswith('['):` guard in `ListenerDebugThread._read_one` (in
  `ghv3_1/ui/debug_tab.py`) filters these — all legitimate `[LST]` lines start with `[`.
  ShouterDebugThread is unaffected (no binary frames on shouter serial).
- **SHOUTER DISTANCES cards stay `--`** — Three known causes: (1) `MIN_SAMPLES` must be 1 (already set).
  (2) **All 4 shouters must run the same firmware** — bidirectional RSSI requires `min(count[i→j], count[j→i]) >= 1`;
  if any shouter runs old firmware it never sends `[BB][A3]`, its direction stays 0, every pair involving it stays `--`.
  (3) **`[BB][A3]` must be sent BEFORE CSI snapshots in ShouterV3.ino** (fixed 2026-03-17) — sending it after
  90 × ~392-byte snapshot packets overflows the listener's UDP RX queue and silently drops `[BB][A3]`.
  Fix already applied: ranging_rpt is now sent immediately after `[BB][EE]`, before the snap loop.
- **Ranging requires all 4 shouters — no timeout fallback** — `run_ranging_phase()` fires only
  when `registered_shouter_count == 4`. If one shouter permanently fails to associate (hardware
  fault, wrong `SHOUTER_ID` flashed), distances stay `--` indefinitely. Fix: resolve the hardware
  issue, then power-cycle the listener.
- **WiFi event handler lambdas in ESP32 Arduino use empty capture `[]`** — file-scope variables
  (`shouter_mac[]`, `shouter_ready[]`, `ranging_done`) are accessed directly, not captured.
  Using `[&]` or `[=]` for file-scope vars is a compile error. Cross-task bools must be
  `volatile` at file scope to prevent compiler register-caching across task boundaries.
- Shouter firmware sends ranging frames only during a discrete ranging phase (logged as `[LST] Starting ranging phase` / `[LST] Ranging phase complete`), not continuously
- Shouter serial port outputs **text only** (`[SHT]` lines) — no binary frames. Listener serial port outputs binary frames (`[0xAA][0x55]`, `[0xBB][0xDD]`, `[0xCC][0xDD]`) mixed with `[LST]` text.
- `NULL_PDIFF_INDICES` in `ghv3_1/config.py` (`{0,1,2,31,32,62,63,64,65}`) differs slightly from
  `NULL_SUBCARRIER_INDICES` — intentional
- `models/` directory is not created automatically — `ghv3_1/train.py` assumes it exists
- `ghv3_1/train.py` default `PROCESSED_DIR` points to `data/processed/` — pass
  `--processed-dir` for specific sessions
- `snr` column is dropped as collinear with `rssi` and `noise_floor`
- **ESP-NOW init sequence** — `esp_now_init()` must be called after `connect_and_register()`
  (WiFi STA fully connected). Do NOT call it again on WiFi dropout/reconnect — it persists.
  Broadcast MAC must be registered via `esp_now_add_peer` before any `esp_now_send` or send
  silently returns `ESP_ERR_ESPNOW_NOT_FOUND`. Use `bcast_peer.channel = 0` (not 6) to avoid
  `ESP_ERR_ESPNOW_CHAN`. `on_esp_now_recv` runs in WiFi task context (Core 0) — use
  `portENTER_CRITICAL` (not ISR variant). `ifidx = WIFI_IF_STA` required in STA mode.
- **Passive background beacons** — Shouters send 1 ESP-NOW beacon/second from `loop()` (after
  ranging phase) to keep peer RSSI estimates live. Causes ~7–10% miss rate increase vs 0%.
  Interval is `last_passive_bcn_ms >= 1000` in ShouterV3.ino; increase to 2000 if misses climb.
- **Test room geometry** — perfect 25ft square. Shouter corners: 1=bottom-left, 2=top-left,
  3=top-right, 4=bottom-right. Sides (7.62m): 1-2, 2-3, 3-4, 4-1. Diagonals (10.78m): 1-3, 2-4.
- **`ranging_completed_ms` must be in globals** — declaring it inside or after `run_ranging_phase()` in
  `firmware/ListenerV3/ListenerV3.ino` causes "not declared in this scope". Place it in the file-scope globals block near `ranging_done`.
- **RSSI-based ranging accuracy** — Log-distance path loss model; `ranging_config.json` hot-reloads
  on each frame (no restart needed). Current calibration: `n=2.16`, `rssi_ref_dbm=-26.2` (two-point,
  anchored on side=7.62m and diagonal=10.78m of 25ft test room, 2026-03-17). RSSI has ±1–2m indoor
  error regardless of calibration — values reflect relative ordering more than absolute meters.
- **CSI MUSIC ranging — fully implemented, not yet producing distances** — spec at
  `docs/superpowers/specs/2026-03-16-music-csi-ranging-design.md`; implementation plan at
  `docs/superpowers/plans/2026-03-16-music-csi-ranging.md`. Replaces RSSI scalar with
  MUSIC super-resolution CIR; offset-free (`d = c × τ`); bidirectional CFO cancellation via
  averaging τ_ij and τ_ji; `CSIMUSICEstimator` class in `ghv3_1/spacing_estimator.py`. MAC
  attribution solved by callback ordering: `shouter_csi_cb` (ISR, Core 0) always completes
  before `on_esp_now_recv` (WiFi task, Core 0) — do NOT move either callback off Core 0.
- **`CSI_SNAP_HEADER_SIZE = 6` in Python, not 8** — `offsetof(csi_snap_pkt_t, csi) = 8` in C
  (magic-inclusive), but `parse_csi_snap_frame` receives a buffer AFTER the 2 magic bytes are
  consumed by the dispatcher, so the pre-CSI header is only 6 bytes. The spec originally had 8;
  corrected in both spec and plan 2026-03-16.
- **ESP32 CSI byte format** — int8 imaginary first, then int8 real, per subcarrier (2 bytes
  each). 128 subcarriers × 2 = 256 bytes minimum for HT20. With `ltf_merge_en=true`,
  `info->len` may exceed 256 bytes; only the first 256 are needed for 128 subcarriers.
- **`bcn_seq=0xFF` passive beacon sentinel already in firmware** — no firmware change needed;
  the guard `if (bcn->bcn_seq == 0xFF) return;` in `on_esp_now_recv` is sufficient to skip
  background beacons during CSI snapshot collection.
- **`test_distance_at_ref_rssi` and `test_distance_formula` always fail** — the `estimator` fixture
  uses default `config_path="ranging_config.json"`, which loads the calibrated file (`n=2.16,
  rssi_ref=-26.2`) instead of generic defaults (`n=2.5, rssi_ref=-40.0`) the tests expect.
  Pre-existing; not a regression. Fix: pass a temp config with generic values to the fixture.
- **Shouter distances off by 3-5m** — RSSI log-distance model with current calibration
  (`n=2.16, rssi_ref=-26.2`) still has significant indoor error. RSSI is being phased out
  in favor of CSI MUSIC ranging (`d = c * tau`, no calibration needed). Diagnostics plan
  at `docs/superpowers/plans/2026-03-17-music-ranging-diagnostics.md`.
- **No Python logging handler configured** — `logging.basicConfig()` is not called anywhere.
  All `_log.info()` / `_log.warning()` calls are silently dropped until a handler is added
  (planned in `run_gui.py` per 2026-03-17 diagnostics plan).
- **Never call `Serial.printf` inside `portENTER_CRITICAL`** — `Serial.printf` is blocking I/O;
  calling it with interrupts disabled triggers ESP32 watchdog timeout. Capture values inside
  the critical section, print after `portEXIT_CRITICAL`.

## Behavioral Rules
- Ask for confirmation before modifying existing files unless told otherwise
- Ask for confirmation before any irreversible action (deleting data, overwriting models, flashing firmware)
- If a task is ambiguous, ask one clarifying question before proceeding
- Prefer incremental changes — one change at a time
- After any source file change, remind the user to rebuild the executable
- Before generating Word/Excel/PDF/PPTX output, check the relevant skill

## Skills
| Output type    | Skill                  |
|----------------|------------------------|
| Word document  | anthropic-skills:docx  |
| Spreadsheet    | anthropic-skills:xlsx  |
| PDF            | anthropic-skills:pdf   |
| Presentation   | anthropic-skills:pptx  |

## Session Prompt Template
Paste at session start and fill in the bracketed fields before sending.
Required: **Current State**, **Task This Session**, **Definition of Done**.
All other fields are optional but improve session quality.

```markdown
## GHV3 Session Start

**Date:** [YYYY-MM-DD]

### Current State
- Last working session: [date + what was accomplished]
- Executable status: [up to date | needs rebuild]
- Active data session dir: data/raw/[SESSION] | data/processed/[SESSION]
- Current model: models/[model.pkl] | none
- Known open issues: [list or "none"]

### Task This Session
Primary:

Sub-tasks (optional):
1.
2.

### Definition of Done
- [ ]
- [ ]

### Hardware / Ports (if relevant)
- Main data COM port: COM[X]
- Listener debug port: COM[X]
- Shouter debug port: COM[X]

### Constraints
-

### Autonomy Level
[ ] Default (confirm before any file edits)
[ ] Fast — confirm only before irreversible actions (data deletion, model overwrite, flash)
[ ] Hands-off — proceed unless destructive

### Output Artifacts Needed
[ ] Rebuilt executable (GHV3_Collector.exe)
[ ] Updated README / docs
[ ] Word doc → anthropic-skills:docx
[ ] Spreadsheet → anthropic-skills:xlsx
[ ] PDF → anthropic-skills:pdf
[ ] Presentation → anthropic-skills:pptx
[ ] None

### Notes / Context

```

## Keeping CLAUDE.md Current
- Press `#` mid-session to capture a learning immediately
- Run `/revise-claude-md` at end of session for a full audit and update

## Session Wrap-Up Trigger
When the user says **"Session over. Wrap up."** (or close variants like "session done", "wrap up", "end session"):
1. Run `/claude-md-management:revise-claude-md` to capture learnings
2. Write a pre-filled next session prompt using the template above, populated with:
   - Today's date as the last working session
   - Actual executable status based on what was changed this session
   - Active data session dir from this session
   - Current model if updated
   - Any open issues or TODOs surfaced during the session
   - Task This Session and Definition of Done left blank for the user to fill in
