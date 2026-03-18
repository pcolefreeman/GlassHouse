"""Run GHV4 preprocessing pipeline."""
from ghv4.preprocess import run
import argparse
from ghv4.config import DATA_RAW_DIR, DATA_PROCESSED_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(DATA_RAW_DIR))
    parser.add_argument("--out-dir", default=str(DATA_PROCESSED_DIR))
    args = parser.parse_args()
    run(args.raw_dir, args.out_dir)
