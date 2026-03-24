# Tests — Known Failures & Fixture Requirements
<!-- last verified: 2026-03-24 -->

## Known Test Failures

- **`test_viz.py` and `test_debug_tab.py` fail without display** — `matplotlib` and
  `customtkinter` require a display/GUI environment; these tests fail in headless or
  minimal environments. Pre-existing, not regressions.
- **`test_pi_display.py` requires pygame** — uses `pytest.importorskip("pygame")` to
  auto-skip if pygame is not installed. Mocks pygame display init so no actual window opens.
- **pygame does not build on Python 3.14** — `pip install pygame` fails. On Pi (Python 3.11/3.12)
  use `sudo apt install python3-pygame`. Tests use `importorskip` to auto-skip gracefully.
- **Python 3.14 subprocess bug** — `subprocess.run` may hang in test discovery on 3.14. Use 3.12 or 3.13 for reliable test runs.

## Test Files (SAR vital sign detector — added 2026-03-24)

- **`test_signal_hardening.py`** — 11 tests: Hampel filter (4), coherence gate (3), subcarrier selection (4)
- **`test_heartrate.py`** — 14 tests: HeartRateAnalyzer (5), PresenceScorer (3), DualBandFusion (2), FullPipelineSynthetic (2)
- **`test_breathing.py`** — 4 tests updated to match BreathingDetector rewrite (PresenceScorer, BreathingAnalyzer tests replace old `_raw_amplitude_energy` / `_phase_score` tests)

## Removed Tests

- **8 tests removed (2026-03-22)** — `[CC][DD]` ranging frame tests deleted from
  `test_serial_io.py` (4) and `test_debug_tab.py` (4) after dead code removal.

## Fixture Requirements

- **BytesIO needs `.timeout` attribute for SerialReader** — `_read_one_frame` accesses
  `self._ser.timeout` on [0xEE][0xFF] frames. Use `class _BytesIOWithTimeout(BytesIO):
  timeout = 1.0` in tests.
- **`META_COLS` in test fixtures must include `activity`** — `eda_utils.load_csv` validates against `config.META_COLS` which includes 6 columns (timestamp_ms, label, zone_id, grid_row, grid_col, activity). Test fixtures that use META_COLS must match.
