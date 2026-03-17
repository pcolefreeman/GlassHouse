GlassHouseV3 — Data Collection Tool
=====================================

QUICK START
-----------
1. Flash ListenerV3.ino to the listener ESP32 (baud 921600).
2. Flash ShouterV3.ino to each of the four shouter ESP32s.
3. Connect the listener ESP32 via USB.
4. Run GHV3_Collector.exe.
5. Use the Serial Port dropdown to select the correct COM port, then click Connect.
6. Enter the grid cell label (e.g. r0c0) and click Start Capture.
7. Raw session CSVs are saved to:  data\raw\<SESSION>\

PORT DROPDOWNS
--------------
- Serial Port   — main data capture (listener ESP32)
- Listener Port — listener debug monitor (optional)
- Shouter Port  — shouter debug monitor (optional)
Click Refresh next to any dropdown to re-enumerate available COM ports.

FOLDER STRUCTURE
----------------
GHV3_Collector.exe   — main application
data\
  raw\               — session CSVs land here (auto-created per session)
  processed\         — output of preprocess.py (run on research PC)

GRID LABELS
-----------
9-cell 3x3 grid, row-major:
  r0c0  r0c1  r0c2
  r1c0  r1c1  r1c2
  r2c0  r2c1  r2c2

NOTES
-----
- Crash log written to ghv2_ui_crash.log (same folder as exe) if app fails to start.
- Baud rate: 921600 (fixed in firmware).
- Do NOT use the Raspberry Pi for training — inference only.
