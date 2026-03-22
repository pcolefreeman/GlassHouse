# Tests — Known Failures & Fixture Requirements
<!-- last verified: 2026-03-19 -->

## Known Test Failures

- **`test_viz.py` and `test_debug_tab.py` fail without display** — `matplotlib` and
  `customtkinter` require a display/GUI environment; these tests fail in headless or
  minimal environments. Pre-existing, not regressions.
- **`test_pi_display.py` requires pygame** — uses `pytest.importorskip("pygame")` to
  auto-skip if pygame is not installed. Mocks pygame display init so no actual window opens.
- **pygame does not build on Python 3.14** — `pip install pygame` fails. On Pi (Python 3.11/3.12)
  use `sudo apt install python3-pygame`. Tests use `importorskip` to auto-skip gracefully.

## Removed Tests

- **8 tests removed (2026-03-22)** — `[CC][DD]` ranging frame tests deleted from
  `test_serial_io.py` (4) and `test_debug_tab.py` (4) after dead code removal.

## Fixture Requirements

- **BytesIO needs `.timeout` attribute for SerialReader** — `_read_one_frame` accesses
  `self._ser.timeout` on [0xEE][0xFF] frames. Use `class _BytesIOWithTimeout(BytesIO):
  timeout = 1.0` in tests.
