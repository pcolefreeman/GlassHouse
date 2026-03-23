# GHV4 — Build, Commands & Distribution
<!-- last verified: 2026-03-19 -->

## Commands
```bash
# Build distribution folder (exe + data dirs + README)
# Output: ghv4_Distro/ghv4_Collector.exe  +  data/raw/  data/processed/  README.txt
# Steps: pyinstaller ghv4_Collector.spec → copy dist/ghv4_Collector.exe → create ghv4_Distro/ tree

# Run GUI (select COM port via dropdown, click Refresh to enumerate ports)
python run_gui.py

# Build executable — run whenever any source file changes
pyinstaller ghv4_Collector.spec

# Preprocess raw CSVs → X.npy / y.npy / scaler.pkl
python run_preprocess.py
python run_preprocess.py --raw-dir data/raw/SESSION --out-dir data/processed/SESSION

# Train (always on PC)
python run_train.py                        # full pipeline (LogReg/KNN/SVM/GBT/RF + vote + stack)
python run_train.py --fast                 # lighter models optimized for Pi inference (train on PC)
python run_train.py --model rf             # force a specific model
python run_train.py --model rf --skip-cv   # skip CV comparison, train RF directly (minutes vs hours)

# Live inference
python run_inference.py --port COM3 --model models/model.pkl --spacing data/raw/spacing.json
python run_inference.py --port COM3        # dry-run (no model needed)

# Pi LCD display (operator zone tracker — pygame-based)
python run_pi_display.py --port /dev/ttyUSB0 --model models/model.pkl --fullscreen
python run_pi_display.py --demo            # cycle cells without hardware

# Tests
pytest tests/
```

## Executable Rebuild Rule
**Rebuild the executable after any change to any file in `ghv4/` or `run_gui.py`:**
```bash
pyinstaller ghv4_Collector.spec
```

`ghv4_Collector.spec` was adapted from the V2 spec (`GHV2_Collector.spec` in the V2
project). Use that as the reference if the spec needs to be regenerated.

## Gotchas

- **`ghv4_Distro/` not auto-updated** — `pyinstaller` only writes to `dist/ghv4_Collector.exe`;
  must manually copy to `ghv4_Distro/ghv4_Collector.exe` after each rebuild.
