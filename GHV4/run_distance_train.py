"""Entry point: train per-pair distance regressors."""
import argparse
import logging

from ghv4.distance_train import run


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train distance models")
    parser.add_argument(
        "--processed-dir",
        default="distance_data/processed",
        help="Preprocessed data directory",
    )
    parser.add_argument(
        "--model-dir",
        default="distance_models",
        help="Output model directory",
    )
    args = parser.parse_args()
    run(args.processed_dir, args.model_dir)


if __name__ == "__main__":
    main()
