"""ghv2_viz.py — GHV2 heatmap rendering, shared by EDA notebook and Pi live display.

Public API:
    render_heatmap(grid_values, area_dims, title, mode, threshold,
                   cmap_lit, cmap_raw, ax) -> Optional[Figure]
"""
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def render_heatmap(
    grid_values,            # np.ndarray shape (3, 3)
    area_dims,              # (width_m, depth_m) or (None, None)
    title,                  # str
    mode="confidence",      # "confidence" | "raw"
    threshold=0.70,         # float: confidence threshold (mode="confidence" only)
    cmap_lit="#FF6B35",     # rescue orange (mode="confidence" only)
    cmap_raw="YlOrRd",      # colormap for raw values
    ax=None,                # None → new Figure; provided → redraw in-place
):
    # type: (...) -> Optional[matplotlib.figure.Figure]
    """Render a 3×3 occupancy or analysis heatmap.

    Returns a new Figure when ax=None (EDA use).
    Returns None when ax is provided (live Pi animation — caller owns Figure).
    """
    owns_fig = (ax is None)
    if owns_fig:
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor('#0d0d0d' if mode == "confidence" else 'white')
    else:
        fig = None
        ax.cla()
        ax.figure.patch.set_facecolor('#0d0d0d' if mode == "confidence" else 'white')

    grid = np.array(grid_values, dtype=float)

    if mode == "confidence":
        _draw_confidence(ax, grid, threshold, cmap_lit)
    else:
        _draw_raw(ax, grid, cmap_raw)

    _set_axis_labels(ax, area_dims, mode)
    ax.set_title(title, color='white' if mode == "confidence" else 'black', pad=10)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')

    if owns_fig:
        plt.tight_layout()
        return fig
    return None


def _draw_confidence(ax, grid, threshold, cmap_lit):
    """Confidence mode: dark below threshold, rescue-orange fill + glow above."""
    ax.set_facecolor('#0d0d0d')
    for row in range(3):
        for col in range(3):
            conf = float(grid[row, col])
            # row 0 is displayed at top → y = 2 - row
            x, y = col, 2 - row

            if conf >= threshold:
                # Solid rescue-orange fill
                rect = patches.Rectangle(
                    (x, y), 1, 1,
                    facecolor=cmap_lit, edgecolor='white', linewidth=1.5,
                )
                ax.add_patch(rect)
                # Glow overlay (stacked on top; only when strictly above threshold).
                # Spec formula: glow_alpha = (conf - threshold) / (1 - threshold) → [0, 1].
                # Scaled by 0.3 as an implementation detail to keep it subtle and
                # prevent the white overlay from washing out the rescue-orange fill.
                if conf > threshold:
                    glow_alpha = (conf - threshold) / (1.0 - threshold) * 0.3
                    glow = patches.Rectangle(
                        (x, y), 1, 1,
                        facecolor='white', alpha=glow_alpha, edgecolor='none',
                    )
                    ax.add_patch(glow)
                label_color = 'white'
            else:
                rect = patches.Rectangle(
                    (x, y), 1, 1,
                    facecolor='#1a1a1a', edgecolor='#444444', linewidth=1.5,
                )
                ax.add_patch(rect)
                label_color = '#666666'

            pct = int(conf * 100)
            ax.text(
                x + 0.5, y + 0.5, f"{pct}%",
                ha='center', va='center',
                fontsize=12, color=label_color, fontweight='bold',
            )


def _draw_raw(ax, grid, cmap_raw):
    """Raw mode: normalised colormap applied to arbitrary float values."""
    ax.set_facecolor('white')
    valid = grid[~np.isnan(grid)]
    if valid.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(valid.min()), float(valid.max())
    eps  = 1e-9
    norm_grid = (grid - vmin) / (vmax - vmin + eps)

    # matplotlib 3.7+ removed plt.get_cmap(); use matplotlib.colormaps (3.5+)
    cmap = matplotlib.colormaps.get_cmap(cmap_raw)
    for row in range(3):
        for col in range(3):
            norm_val = float(norm_grid[row, col])
            color    = cmap(norm_val)
            x, y     = col, 2 - row

            rect = patches.Rectangle(
                (x, y), 1, 1,
                facecolor=color, edgecolor='white', linewidth=1.5,
            )
            ax.add_patch(rect)

            raw_val    = float(grid[row, col])
            text_color = 'black' if norm_val > 0.5 else 'white'
            ax.text(
                x + 0.5, y + 0.5, f"{raw_val:.1f}",
                ha='center', va='center',
                fontsize=11, color=text_color,
            )


def _set_axis_labels(ax, area_dims, mode):
    """Set cell-centre tick labels — physical metres or cell indices."""
    tick_color = 'white' if mode == "confidence" else 'black'

    # Ticks are always at cell-centre positions in plot coordinates [0, 3].
    # Cell centres are at 0.5, 1.5, 2.5 (col 0, 1, 2 respectively).
    # Labels are physical metres when area_dims is known, cell indices otherwise.
    if area_dims is not None and area_dims[0] is not None:
        width_m, depth_m = float(area_dims[0]), float(area_dims[1])
        cw = width_m / 3.0
        cd = depth_m / 3.0

        # Cell centres in plot coords = [0.5, 1.5, 2.5]; labels show physical range
        x_labels = [f"{i * cw:.1f}–{(i + 1) * cw:.1f} m" for i in range(3)]
        ax.set_xticks([0.5, 1.5, 2.5])
        ax.set_xticklabels(x_labels, fontsize=9, color=tick_color)
        ax.set_xlabel(f"Width ({width_m:.1f} m)", color=tick_color)

        # Row 0 is at top (y=2 in plot); row 2 is at bottom (y=0 in plot).
        # y-tick centres: row 2 → y=0.5, row 1 → y=1.5, row 0 → y=2.5
        y_labels = [
            f"{(2 - i) * cd:.1f}–{(3 - i) * cd:.1f} m"   # row at y-centre i+0.5
            for i in range(3)                               # i=0 → bottom, i=2 → top
        ]
        ax.set_yticks([0.5, 1.5, 2.5])
        ax.set_yticklabels(y_labels, fontsize=9, color=tick_color)
        ax.set_ylabel(f"Depth ({depth_m:.1f} m)", color=tick_color)
    else:
        ax.set_xticks([0.5, 1.5, 2.5])
        ax.set_xticklabels(['0', '1', '2'], color=tick_color)
        ax.set_xlabel("Cell index (col)", color=tick_color)

        ax.set_yticks([0.5, 1.5, 2.5])
        ax.set_yticklabels(['2', '1', '0'], color=tick_color)
        ax.set_ylabel("Cell index (row)", color=tick_color)

    ax.tick_params(colors=tick_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(tick_color)
