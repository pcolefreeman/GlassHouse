"""Run GHV4 model training pipeline."""
from ghv4.train import run
import argparse
from ghv4.config import DATA_PROCESSED_DIR, MODELS_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default=str(DATA_PROCESSED_DIR))
    parser.add_argument("--out-dir", default=str(MODELS_DIR))
    parser.add_argument("--model", default=None,
                        help="Force a specific model key: "
                             "logreg | knn | svm | gbt | rf | voting | stacking")
    parser.add_argument("--fast", action="store_true",
                        help="Reduce tree counts for slower hardware (Pi 4B)")
    parser.add_argument("--skip-cv", action="store_true",
                        help="Skip CV comparison, train --model directly (much faster)")
    args = parser.parse_args()
    run(args.processed_dir, args.out_dir,
        force_model=args.model, fast=args.fast, skip_cv=args.skip_cv)
