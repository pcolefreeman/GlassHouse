"""Entry point: preprocess raw distance CSVs into ML-ready arrays."""
import argparse
import logging

from ghv4.distance_preprocess import run


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Preprocess distance training data")
    parser.add_argument(
        "--raw-dir", default="distance_data/raw", help="Raw CSV directory"
    )
    parser.add_argument(
        "--out-dir", default="distance_data/processed", help="Output directory"
    )
    args = parser.parse_args()
    run(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
