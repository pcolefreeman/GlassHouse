"""Launch the GHV3.1 data collection GUI."""
import logging

from ghv3_1.ui.app import main

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    main()
