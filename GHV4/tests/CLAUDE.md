# Tests — Known Failures & Fixture Requirements
<!-- last verified: 2026-03-19 -->

## Known Test Failures

- **`test_viz.py` and `test_debug_tab.py` fail without display** — `matplotlib` and
  `customtkinter` require a display/GUI environment; these tests fail in headless or
  minimal environments. Pre-existing, not regressions.

## Fixture Requirements

- **BytesIO needs `.timeout` attribute for SerialReader** — `_read_one_frame` accesses
  `self._ser.timeout` on [0xEE][0xFF] frames. Use `class _BytesIOWithTimeout(BytesIO):
  timeout = 1.0` in tests.
