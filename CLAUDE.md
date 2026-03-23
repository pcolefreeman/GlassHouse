# GlassHouseV4 — WiFi CSI Indoor Positioning

Senior capstone @ Georgia Southern University. Four ESP32 shouters at room
corners + one listener ESP32 collect CSI/RSSI across a 3×3 grid. A
scikit-learn classifier maps each 200 ms bucket of signal data to a grid cell.

## Quick Status
- Test count: 181 passing + 1 skipped (2026-03-23)
- Last exe rebuild: needs rebuild (inference.py + preprocess.py + debug_tab + serial_io + pi_display + config + breathing + run_sar changed)
- Unstaged changes: ML distance pipeline + [CC][DD] removal + inference scaling fix + spacing name fix + Pi LCD display + META_COLS activity fix + breathing detection + --skip-cv training + continuous shouter beacons firmware
- Active branch: main
- Done: Continuous shouter beacons firmware (2026-03-24) — ranging disabled, 10 Hz beacons added
- Done: Continuous snap breathing detection spec (2026-03-24) — design approved, spec reviewed
- Done: Continuous snap breathing implementation plan (2026-03-24) — reviewer approved, 10 tasks
- Done: Empty-room baseline validated 2026-03-24 — root cause of false positives identified (listener proximity to paths)
- Architecture decision: redesign `_variance_score()` to per-path rolling delta variance (not absolute) — pending implementation
- Deployment constraint confirmed 2026-03-24: listener must be stationary INSIDE room during scan — moving outside behind closed door causes all paths to saturate (WiFi link degradation)
- Pending: End-to-end firmware/serial/data-collection debugging plan
- Trained model: `models/rf_best.pkl` (RF, 99.93% train accuracy on 35K×2894 dataset)

## Version Control
Git repo: remote at https://github.com/pcolefreeman/GlassHouse.git, branch `main`.
V3 source lives in the `ghv4/` subdirectory (repo root also has GHV1/, Arduino_IDE/, etc.).
Never track exe binaries — ghv4_Collector.exe is ~100MB and hits GitHub's hard limit.
`.gitignore` lives at the repo root. Windows can silently save it as UTF-16 — git ignores UTF-16 .gitignore files entirely. Verify encoding with `xxd .gitignore | head -1` (should NOT start with `ff fe`).

## Hardware
- **Listener + Shouters**: ESP32-WROOM-UE (devkitC); baud rate 921600
- Raspberry Pi 4B — inference/deployment only; never used for training

## Architecture

### PC Software (Python — `ghv4/` package)
```
run_gui.py             — entry point: launches GUI
run_inference.py       — entry point: live inference
run_pi_display.py      — entry point: Pi LCD operator display (pygame)
run_preprocess.py      — entry point: preprocessing pipeline
run_train.py           — entry point: model training
run_sar.py             — entry point: SAR breathing detection (planned, not yet implemented)

ghv4/
  __init__.py          — package root (__version__ = "3.1.0")
  config.py            — single source of truth for ALL cross-module constants
  csi_parser.py        — frame structs, feature extraction (shared by all modules)
  serial_io.py         — SerialReader (byte stream → frame_queue) + CSVWriter (frame_queue → CSV)
  spacing_estimator.py — MUSIC distance estimation via [0xEE][0xFF] CSI snapshots → spacing.json
  preprocess.py        — raw CSV → X.npy, y.npy, feature_names.txt, scaler.pkl
  train.py             — CV comparison → VotingClassifier → StackingClassifier → saves best
  inference.py         — live serial → feature extraction → model.predict()
  pi_display.py        — pygame 3×3 grid display for Pi LCD (operator view)
  eda_utils.py         — label parsing, shared EDA helpers
  viz.py               — visualization widgets (heatmap, spacing overlay)
  cell_logic.py        — pure label/cell helper functions
  breathing.py         — CSI breathing/micro-motion detection (planned, not yet implemented)
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
ghv4_Distro/           — packaged distribution (exe + data dirs + README)
```

### ESP32 Firmware (Arduino/C++)
```
firmware/
  ghv4Protocol.h       — shared packet structs + magic byte constants (single copy)
  ListenerV4/
    ListenerV4.ino     — listener ESP32 firmware
                           WiFi AP (SSID: CSI_PRIVATE_AP, ch 6)
                           UDP server on port 3333; polls shouters on port 3334
                           CSI ring buffer (ISR → task) → emit_listener_frame [0xAA][0x55]
                           Handles HELLO, CSI_SNAP, RESP UDP packets
                           emit_shouter_frame [0xBB][0xDD] for each poll hit/miss
                           emit_csi_snap_frame [0xEE][0xFF] for MUSIC CSI snapshots
                           Non-blocking ranging state machine (advance_ranging())
                           Broadcast polling with staggered responses (#if USE_BROADCAST_POLL)
                           MAC-based runtime ID assignment (no compile-time SHOUTER_ID)
                           Text output: [LST] prefixed lines at 921600 baud
  ShouterV4/
    ShouterV4.ino      — shouter ESP32 firmware
                           Connects to listener AP as WiFi STA
                           Sends [BB][FA] HELLO on connect, re-sends on reconnect
                           Responds to polls ([BB][CC]) with CSI response ([BB][EE])
                           During ranging: beacons when instructed, sends CSI snapshots [BB][A4]
                           CSI ring buffer (ISR → task) → included in SHOUT response
                           Text output: [SHT] prefixed lines at 921600 baud (text only, no binary frames on serial)
```

## Subsystem Details
Detailed gotchas, protocols, and conventions are in local CLAUDE.md files
(auto-loaded when working in those directories):
- **`GHV4/CLAUDE.md`** — build commands, exe rebuild rule, distribution
- **`GHV4/firmware/CLAUDE.md`** — serial/UDP frame protocol, ESP32 gotchas, ranging, CSI format
- **`GHV4/ghv4/CLAUDE.md`** — constants rule, data/labels, GUI ports, ML distance pipeline, parser gotchas
- **`GHV4/tests/CLAUDE.md`** — known test failures, test fixture requirements

## Behavioral Rules
- Do NOT run any git commands (commit, add, push, pull, checkout) — user handles all git operations
- Ask for confirmation before moving or deleting files/data
- If a task is ambiguous, ask one clarifying question before proceeding
- Prefer incremental changes — one change at a time
- After any source file change, remind the user to rebuild the executable
- Before generating Word/Excel/PDF/PPTX output, check the relevant skill

## Keeping Subsystem Docs Current
When you modify behavior described in a subsystem CLAUDE.md, update that file in the same session.
When adding a new gotcha, place it in the nearest subsystem CLAUDE.md, not the root.
If a subsystem CLAUDE.md becomes stale, update its `<!-- last verified -->` date after review.

## Skills
| Output type    | Skill                  |
|----------------|------------------------|
| Word document  | anthropic-skills:docx  |
| Spreadsheet    | anthropic-skills:xlsx  |
| PDF            | anthropic-skills:pdf   |
| Presentation   | anthropic-skills:pptx  |

## Model Recommendation Guide
- **Opus** — broad cross-module reasoning, architecture changes, multi-file refactors, debugging complex interactions
- **Sonnet** — focused single-module work, feature implementation, test writing, routine edits
- **Haiku** — quick lookups, simple questions, formatting, typo fixes

## Session Prompt Template
Paste at session start and fill in the bracketed fields before sending.
Required: **Current State**, **Task This Session**, **Definition of Done**.
All other fields are optional but improve session quality.

```markdown
## ghv4 Session Start

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

### Files Likely Touched (helps Claude load context faster)
-

### Related Sessions (memory file names, if any)
-

### Model Recommendation from Last Session
[Opus | Sonnet | Haiku] — [reason]

### Constraints
-

### Autonomy Level
[ ] Default (confirm before any file edits)
[ ] Fast — confirm only before irreversible actions (data deletion, model overwrite, flash)
[ ] Hands-off — proceed unless destructive

### Output Artifacts Needed
[ ] Rebuilt executable (ghv4_Collector.exe)
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
2. Update the **Quick Status** section at the top of this file with current test count, exe status, unstaged changes, and branch
3. Recommend a Claude model for the next session (Opus/Sonnet/Haiku) based on likely next task
4. Write a pre-filled next session prompt using the template above, populated with:
   - Today's date as the last working session
   - Actual executable status based on what was changed this session
   - Active data session dir from this session
   - Current model if updated
   - Any open issues or TODOs surfaced during the session
   - The model recommendation in the "Model Recommendation from Last Session" field
   - Task This Session and Definition of Done left blank for the user to fill in
