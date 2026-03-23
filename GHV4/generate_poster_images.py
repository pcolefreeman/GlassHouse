"""Generate poster-ready PNGs showing the GlassHouse V4 UI and heatmap.

Produces:
  models/ui_demo.png         — simulated Pi LCD grid display with a prediction
  models/confidence_heatmap.png — per-cell mean confidence heatmap from trained model

Usage:
    python generate_poster_images.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI needed
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib

# ---------------------------------------------------------------------------
# Config (mirrors ghv4/config.py colors)
# ---------------------------------------------------------------------------
BG = "#0d0d0d"
ACTIVE = "#FF6B35"
INACTIVE = "#1a1a1a"
BORDER = "#444444"
TEXT_ACTIVE = "white"
TEXT_INACTIVE = "#666666"
CYAN = "#00C8C8"

CELL_LABELS = [f"r{r}c{c}" for r in range(3) for c in range(3)]

MODELS_DIR = Path(__file__).parent / "models"
PROCESSED_DIR = Path(__file__).parent / "data" / "processed"


# ---------------------------------------------------------------------------
# 1) Simulated UI Screenshot
# ---------------------------------------------------------------------------
def draw_ui_demo(active_cell="r1c1", confidence=0.97, save_path=None):
    """Render a matplotlib replica of the Pi LCD grid display."""
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Title
    ax.text(1.5, 3.55, "GlassHouse V4 — Zone Tracker",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color=TEXT_ACTIVE, fontfamily="monospace")

    # Grid
    for row in range(3):
        for col in range(3):
            label = f"r{row}c{col}"
            x, y = col, 2 - row  # row 0 at top
            is_active = (label == active_cell)

            fill = ACTIVE if is_active else INACTIVE
            rect = patches.FancyBboxPatch(
                (x + 0.03, y + 0.03), 0.94, 0.94,
                boxstyle="round,pad=0.02",
                facecolor=fill, edgecolor=BORDER, linewidth=2,
            )
            ax.add_patch(rect)

            text_color = TEXT_ACTIVE if is_active else TEXT_INACTIVE

            if is_active:
                # Label + confidence
                ax.text(x + 0.5, y + 0.58, label,
                        ha="center", va="center", fontsize=20,
                        fontweight="bold", color=text_color, fontfamily="monospace")
                ax.text(x + 0.5, y + 0.35, f"{confidence:.0%}",
                        ha="center", va="center", fontsize=14,
                        color=text_color, fontfamily="monospace")
                # Glow overlay
                glow = patches.FancyBboxPatch(
                    (x + 0.03, y + 0.03), 0.94, 0.94,
                    boxstyle="round,pad=0.02",
                    facecolor="white", alpha=0.08, edgecolor="none",
                )
                ax.add_patch(glow)
            else:
                ax.text(x + 0.5, y + 0.5, label,
                        ha="center", va="center", fontsize=18,
                        fontweight="bold", color=text_color, fontfamily="monospace")

    # Shouter markers at corners
    shouter_pos = {2: (0, 3), 3: (3, 3), 1: (0, 0), 4: (3, 0)}
    offsets = {2: (-0.2, 0.15), 3: (0.2, 0.15), 1: (-0.2, -0.15), 4: (0.2, -0.15)}
    for sid, (sx, sy) in shouter_pos.items():
        circle = patches.Circle((sx, sy), 0.12, facecolor=CYAN,
                                edgecolor="white", linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        ox, oy = offsets[sid]
        ax.text(sx + ox, sy + oy, f"S{sid}", ha="center", va="center",
                fontsize=9, color=CYAN, fontweight="bold", fontfamily="monospace")

    # Status bar
    ax.text(1.5, -0.35, "Demo mode  |  Last: r1c1 @ 14:32:07",
            ha="center", va="center", fontsize=11,
            color=TEXT_INACTIVE, fontfamily="monospace")
    ax.plot([-0.3, 3.3], [-0.15, -0.15], color=BORDER, linewidth=1)

    ax.set_xlim(-0.4, 3.4)
    ax.set_ylim(-0.55, 3.75)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, facecolor=fig.get_facecolor(),
                    bbox_inches="tight", pad_inches=0.2)
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2) Per-cell Confidence Heatmap from Trained Model
# ---------------------------------------------------------------------------
def draw_confidence_heatmap(save_path=None):
    """Load trained model + processed data, compute per-cell confidence, render heatmap."""
    model_path = MODELS_DIR / "rf_best.pkl"
    X_path = PROCESSED_DIR / "X.npy"
    y_path = PROCESSED_DIR / "y.npy"

    if not model_path.exists():
        print(f"  Skipping heatmap: {model_path} not found")
        return
    if not X_path.exists() or not y_path.exists():
        print(f"  Skipping heatmap: processed data not found in {PROCESSED_DIR}")
        return

    print("  Loading model and data...")
    model = joblib.load(model_path)
    X = np.load(X_path)
    y = np.load(y_path)

    # Predict probabilities
    print(f"  Predicting on {X.shape[0]} samples...")
    proba = model.predict_proba(X)  # (N, 9)

    # For each sample, get its true cell and the model's confidence for that cell
    true_cells = y.argmax(axis=1)  # (N,)

    # Per-cell mean confidence (model's confidence for the correct class)
    grid = np.zeros((3, 3))
    grid_count = np.zeros((3, 3), dtype=int)
    for i in range(len(true_cells)):
        cell_idx = true_cells[i]
        row, col = cell_idx // 3, cell_idx % 3
        grid[row, col] += proba[i, cell_idx]
        grid_count[row, col] += 1

    # Average confidence per cell
    mask = grid_count > 0
    grid[mask] /= grid_count[mask]

    # Render
    fig, ax = plt.subplots(figsize=(7, 7), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Color cells by confidence
    for row in range(3):
        for col in range(3):
            conf = grid[row, col]
            x, y_pos = col, 2 - row

            # Interpolate color: low confidence → dark, high → rescue orange
            if conf >= 0.7:
                fill = ACTIVE
                alpha_glow = (conf - 0.7) / 0.3 * 0.3
            else:
                fill = INACTIVE
                alpha_glow = 0.0

            rect = patches.Rectangle(
                (x, y_pos), 1, 1,
                facecolor=fill, edgecolor="white", linewidth=1.5,
            )
            ax.add_patch(rect)

            if alpha_glow > 0:
                glow = patches.Rectangle(
                    (x, y_pos), 1, 1,
                    facecolor="white", alpha=alpha_glow, edgecolor="none",
                )
                ax.add_patch(glow)

            # Labels
            label_color = TEXT_ACTIVE if conf >= 0.7 else TEXT_INACTIVE
            cell_label = f"r{row}c{col}"
            ax.text(x + 0.5, y_pos + 0.6, cell_label,
                    ha="center", va="center", fontsize=12,
                    color=label_color, fontweight="bold")
            ax.text(x + 0.5, y_pos + 0.38, f"{conf:.1%}",
                    ha="center", va="center", fontsize=10, color=label_color)
            ax.text(x + 0.5, y_pos + 0.18, f"n={grid_count[row, col]}",
                    ha="center", va="center", fontsize=8, color="#888888")

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal")

    # Axis labels
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(["Col 0", "Col 1", "Col 2"], color="white", fontsize=10)
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["Row 2", "Row 1", "Row 0"], color="white", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    ax.set_title("Per-Cell Mean Prediction Confidence\n(Random Forest — 35,877 samples)",
                 color="white", fontsize=14, pad=12)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, facecolor=fig.get_facecolor(),
                    bbox_inches="tight", pad_inches=0.2)
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    MODELS_DIR.mkdir(exist_ok=True)

    print("Generating poster images...")

    ui_path = MODELS_DIR / "ui_demo.png"
    draw_ui_demo(active_cell="r1c1", confidence=0.97, save_path=ui_path)

    heatmap_path = MODELS_DIR / "confidence_heatmap.png"
    draw_confidence_heatmap(save_path=heatmap_path)

    print("Done!")


if __name__ == "__main__":
    main()
