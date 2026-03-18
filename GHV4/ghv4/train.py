"""train.py — GHV2 model selection → voting → stacking pipeline.

Stage 1 : Individual CV comparison  (LogReg, KNN, SVM, GBT, RF)
Stage 2 : Soft VotingClassifier      (all base models)
Stage 3 : StackingClassifier         (base models + LogReg meta-learner)

The best scorer by val F1-macro is saved as the final model.

Usage:
    python train.py                        # full pipeline, default data dir
    python train.py --fast                 # lighter models for Raspberry Pi
    python train.py --model rf             # force a specific model key
    python train.py --processed-dir p/to/processed --out-dir p/to/models
"""
import argparse
import os

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

from ghv4.config import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    CELL_LABELS,
    GRID_POS,
)

PROCESSED_DIR = str(DATA_PROCESSED_DIR)
CV_FOLDS = 5


# ── SAR spatial metrics ────────────────────────────────────────────────────────
def cell_dist(i: int, j: int) -> int:
    """Chebyshev distance between two cell indices on the 3x3 grid.
    0 = exact, 1 = adjacent (incl. diagonal), 2 = far (opposite corner etc.)
    """
    r1, c1 = GRID_POS[i]
    r2, c2 = GRID_POS[j]
    return max(abs(r1 - r2), abs(c1 - c2))


def zone_accuracy(y_true, y_pred) -> tuple[float, float, float]:
    """Returns (exact_acc, zone_acc, far_error_rate).
    zone_acc  : fraction where prediction is correct cell OR adjacent cell (dist ≤ 1)
    far_error : fraction where prediction is 2+ cells away — the dangerous ones
    """
    dists      = np.array([cell_dist(t, p) for t, p in zip(y_true, y_pred)])
    exact_acc  = (dists == 0).mean()
    zone_acc   = (dists <= 1).mean()
    far_rate   = (dists >= 2).mean()
    return exact_acc, zone_acc, far_rate


# ── Model factory ─────────────────────────────────────────────────────────────
def make_base_estimators(fast: bool) -> list[tuple[str, object]]:
    """
    Returns (name, estimator) pairs used as base models everywhere.
    `fast=True` cuts tree counts for slower hardware (Pi 4B).
    """
    rf_trees  = 100 if fast else 300
    gbt_trees = 50  if fast else 200

    return [
        ("logreg", Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=2000, C=1.0,
                                          solver="lbfgs", random_state=42)),
        ])),
        ("knn", Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier(n_neighbors=5, weights="distance",
                                             metric="euclidean")),
        ])),
        ("svm", Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                           decision_function_shape="ovr", random_state=42)),
        ])),
        ("gbt", GradientBoostingClassifier(
            n_estimators=gbt_trees, learning_rate=0.1,
            max_depth=4, random_state=42,
        )),
        ("rf", RandomForestClassifier(
            n_estimators=rf_trees, max_features="sqrt",
            min_samples_leaf=2, n_jobs=-1, random_state=42,
        )),
    ]


def make_all_candidates(fast: bool) -> list[tuple[str, str, object]]:
    """
    Returns (display_name, save_key, estimator) for every model that will
    be evaluated, including the voting and stacking ensembles.
    """
    base = make_base_estimators(fast)

    # ── Stage 1 : individuals ─────────────────────────────────────────────────
    candidates = [
        ("Logistic Regression",       "logreg", base[0][1]),
        ("K-Nearest Neighbours (k=5)","knn",    base[1][1]),
        ("SVM (RBF)",                 "svm",    base[2][1]),
        ("Gradient Boosting",         "gbt",    base[3][1]),
        ("Random Forest",             "rf",     base[4][1]),
    ]

    # ── Stage 2 : soft voting ─────────────────────────────────────────────────
    # SVC already has probability=True above so soft voting works.
    voting = VotingClassifier(estimators=base, voting="soft", n_jobs=-1)
    candidates.append(("Voting (soft, all 5)", "voting", voting))

    # ── Stage 3 : stacking ────────────────────────────────────────────────────
    # Meta-learner sees out-of-fold predicted probabilities from each base model.
    # passthrough=True also feeds the original features to the meta-learner.
    meta = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                               random_state=42)
    stacking = StackingClassifier(
        estimators=base,
        final_estimator=meta,
        cv=CV_FOLDS,          # same folds as outer CV for consistency
        stack_method="predict_proba",
        passthrough=True,     # give meta-learner raw features too
        n_jobs=-1,
    )
    candidates.append(("Stacking (LR meta)", "stacking", stacking))

    return candidates


# ── CV comparison ─────────────────────────────────────────────────────────────
def run_selection(X, y_cls, cv, candidates) -> dict:
    results = {}

    print(f"{'Model':<32}  {'Val Acc':>8}  {'± Acc':>6}  {'Val F1':>8}  {'± F1':>6}")
    print("-" * 72)

    for stage, (name, key, estimator) in enumerate(candidates):
        # Print a stage header at the right boundaries
        if key == "voting":
            print()
            print("  -- Stage 2: Ensemble (Voting) --")
        elif key == "stacking":
            print()
            print("  -- Stage 3: Ensemble (Stacking) --")
        elif stage == 0:
            print("  -- Stage 1: Individual models --")

        cv_res = cross_validate(
            estimator, X, y_cls, cv=cv,
            scoring=["accuracy", "f1_macro"],
            return_train_score=True,
            n_jobs=-1,
        )
        acc_v = cv_res["test_accuracy"]
        f1_v  = cv_res["test_f1_macro"]
        results[key] = dict(
            name=name,
            estimator=estimator,
            acc_mean=acc_v.mean(), acc_std=acc_v.std(),
            f1_mean=f1_v.mean(),  f1_std=f1_v.std(),
            train_acc_mean=cv_res["train_accuracy"].mean(),
        )
        print(f"  {name:<30}  {acc_v.mean():.4f}  ±{acc_v.std():.4f}"
              f"  {f1_v.mean():.4f}  ±{f1_v.std():.4f}")

    best_key = max(results, key=lambda k: results[k]["f1_mean"])
    print(f"\n  Winner → {results[best_key]['name']}  "
          f"(F1 {results[best_key]['f1_mean']:.4f})\n")
    return results, best_key


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_selection(results: dict, out_dir: str):
    keys  = list(results.keys())
    names = [results[k]["name"] for k in keys]
    f1_m  = np.array([results[k]["f1_mean"]  for k in keys])
    f1_s  = np.array([results[k]["f1_std"]   for k in keys])
    acc_m = np.array([results[k]["acc_mean"] for k in keys])

    # Colour-code stages
    colours = (["#4C9BE8"] * 5) + ["#F5A623"] + ["#7ED321"]

    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, f1_m, yerr=f1_s, capsize=5, color=colours, alpha=0.85,
           label="Val F1-macro")
    ax.plot(x, acc_m, "o--", color="#E84C4C", label="Val Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Model Selection — {CV_FOLDS}-fold CV\n"
                 f"Blue=individual  Orange=voting  Green=stacking")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "model_selection.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(clf, X, y_cls, name, f1_mean, f1_std, out_dir):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y_cls, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CELL_LABELS).plot(
        ax=ax, colorbar=True, cmap="Blues"
    )
    ax.set_title(f"Confusion Matrix — {name}\n"
                 f"(Val F1 {f1_mean:.3f} ± {f1_std:.3f})")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return y_pred


def plot_spatial_confusion(y_true, y_pred, name, exact_acc, zone_acc,
                           far_rate, out_dir):
    """Confusion matrix where cell background encodes spatial error severity.
    Exact match  → blue   (operationally perfect)
    Adjacent     → yellow (operationally acceptable)
    Far (dist≥2) → red    (dangerous mis-direction)
    """
    cm = confusion_matrix(y_true, y_pred)
    n  = len(CELL_LABELS)

    # Build per-cell colour based on distance
    bg = np.empty((n, n), dtype=object)
    for true_i in range(n):
        for pred_j in range(n):
            d = cell_dist(true_i, pred_j)
            if d == 0:
                bg[true_i, pred_j] = "#D6EAF8"   # light blue  — exact
            elif d == 1:
                bg[true_i, pred_j] = "#FEF9E7"   # light yellow — adjacent
            else:
                bg[true_i, pred_j] = "#FADBD8"   # light red   — far

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.invert_yaxis()

    for true_i in range(n):
        for pred_j in range(n):
            count = cm[true_i, pred_j]
            ax.add_patch(plt.Rectangle(
                (pred_j, true_i), 1, 1,
                facecolor=bg[true_i, pred_j], edgecolor="#AAAAAA", linewidth=0.8
            ))
            if count > 0:
                ax.text(pred_j + 0.5, true_i + 0.5, str(count),
                        ha="center", va="center", fontsize=10,
                        fontweight="bold" if true_i == pred_j else "normal")

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(CELL_LABELS, fontsize=9)
    ax.set_yticklabels(CELL_LABELS, fontsize=9)
    ax.set_xlabel("Predicted cell")
    ax.set_ylabel("True cell")
    ax.set_title(
        f"Spatial Confusion — {name}\n"
        f"Exact {exact_acc:.1%}  |  Zone (±1 cell) {zone_acc:.1%}  |  Far errors {far_rate:.1%}\n"
        f"Blue=exact  Yellow=adjacent (OK)  Red=far (dangerous)"
    )
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "spatial_confusion.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_importance(clf, feat_names, name, out_dir):
    raw = clf
    if hasattr(raw, "steps"):
        raw = raw.steps[-1][1]
    # For stacking, try the final_estimator's coef
    if hasattr(raw, "final_estimator_"):
        raw = raw.final_estimator_

    if hasattr(raw, "feature_importances_"):
        imp     = raw.feature_importances_
        n_feats = min(len(feat_names), len(imp))
        TOP_N   = min(20, n_feats)
        top_idx = np.argsort(imp)[::-1][:TOP_N]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(range(TOP_N), imp[top_idx][::-1], color="#4C9BE8")
        ax.set_yticks(range(TOP_N))
        ax.set_yticklabels([feat_names[i] for i in top_idx][::-1], fontsize=8)
        ax.set_xlabel("Feature importance (Gini)")
        ax.set_title(f"Top {TOP_N} Feature Importances — {name}")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "feature_importance.png"),
                    dpi=130, bbox_inches="tight")
        plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def run(processed_dir: str, out_dir: str,
        force_model: str | None = None, fast: bool = False):

    # Load
    X = np.load(os.path.join(processed_dir, "X.npy"))
    y = np.load(os.path.join(processed_dir, "y.npy"))
    with open(os.path.join(processed_dir, "feature_names.txt")) as f:
        feat_names = f.read().splitlines()

    y_cls = y.argmax(axis=1)
    print(f"Loaded  X={X.shape}  y={y.shape}")
    print(f"Class distribution: { {CELL_LABELS[i]: int((y_cls==i).sum()) for i in range(9)} }")
    if fast:
        print("  [--fast mode: reduced tree counts for Pi 4B]\n")
    else:
        print()

    cv         = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    candidates = make_all_candidates(fast)
    os.makedirs(out_dir, exist_ok=True)

    # Selection
    print(f"=== Model Selection ({CV_FOLDS}-fold stratified CV) ===\n")
    results, best_key = run_selection(X, y_cls, cv, candidates)
    plot_selection(results, out_dir)

    if force_model:
        lookup = {k: est for _, k, est in candidates}
        if force_model not in lookup:
            raise ValueError(f"--model must be one of: {list(lookup)}")
        best_key = force_model
        print(f"  (Overriding winner with --model {force_model})\n")

    chosen    = results[best_key]
    final_clf = chosen["estimator"]

    # Final fit
    print(f"=== Final fit: {chosen['name']} ===\n")
    final_clf.fit(X, y_cls)
    y_pred    = final_clf.predict(X)
    train_acc = accuracy_score(y_cls, y_pred)

    exact_acc, zone_acc, far_rate = zone_accuracy(y_cls, y_pred)

    print(f"  Train accuracy (full) : {train_acc:.4f}")
    print(f"  Exact cell accuracy   : {exact_acc:.4f}")
    print(f"  Zone accuracy (±1)    : {zone_acc:.4f}  ← SAR operational metric")
    print(f"  Far error rate (≥2)   : {far_rate:.4f}  ← dangerous mis-directions\n")
    print(classification_report(y_cls, y_pred, target_names=CELL_LABELS))

    plot_confusion(final_clf, X, y_cls,
                   chosen["name"], chosen["f1_mean"], chosen["f1_std"], out_dir)
    plot_spatial_confusion(y_cls, y_pred, chosen["name"],
                           exact_acc, zone_acc, far_rate, out_dir)
    plot_importance(final_clf, feat_names, chosen["name"], out_dir)

    # Save
    model_path = os.path.join(str(out_dir), f"{best_key}_best.pkl")
    joblib.dump(final_clf, model_path)

    print(f"\nSaved to {out_dir}:")
    print(f"  {best_key}_best.pkl")
    print(f"  model_selection.png")
    print(f"  confusion_matrix.png")
    print(f"  spatial_confusion.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default=PROCESSED_DIR)
    parser.add_argument("--out-dir",       default=str(MODELS_DIR))
    parser.add_argument("--model",         default=None,
                        help="Force a specific model key: "
                             "logreg | knn | svm | gbt | rf | voting | stacking")
    parser.add_argument("--fast", action="store_true",
                        help="Reduce tree counts for slower hardware (Pi 4B)")
    args = parser.parse_args()
    run(args.processed_dir, args.out_dir,
        force_model=args.model, fast=args.fast)
