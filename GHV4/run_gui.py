"""Launch the GHV4 data collection GUI."""
import logging
import os
import sys
import time

# Write logs to a file next to the exe (visible when running as frozen exe)
if getattr(sys, "frozen", False):
    _log_dir = os.path.dirname(sys.executable)
else:
    _log_dir = os.path.dirname(os.path.abspath(__file__))
_log_path = os.path.join(_log_dir, f"ghv4_debug_{time.strftime('%H%M%S')}.log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(_log_path, encoding="utf-8"),
    ],
)

from ghv4.ui.app import main

if __name__ == "__main__":
    main()
