# GHV2.1/tests/test_viz.py
"""Tests for ghv5.viz shouter overlay rendering."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest


_FULL_SPACING = {
    "1-2": 2.5, "1-3": 3.8, "1-4": 2.4,
    "2-3": 2.3, "2-4": 3.9, "3-4": 2.6,
}

_PARTIAL_SPACING = {"1-2": 2.5}   # only one pair available


def _get_ax_patches_and_texts(fig):
    ax = fig.axes[0]
    return ax.patches, ax.texts


def test_render_heatmap_no_spacing_still_works():
    """shouter_spacing=None must produce same output as ghv2_viz (no overlay)."""
    from ghv5.viz import render_heatmap
    grid = [[0.9, 0.1, 0.2], [0.3, 0.8, 0.1], [0.2, 0.1, 0.7]]
    fig = render_heatmap(grid, (None, None), "Test", shouter_spacing=None)
    assert fig is not None
    plt.close(fig)


def test_render_heatmap_with_spacing_returns_figure():
    """render_heatmap with shouter_spacing dict must return a Figure."""
    from ghv5.viz import render_heatmap
    grid = [[0.9, 0.1, 0.2], [0.3, 0.8, 0.1], [0.2, 0.1, 0.7]]
    fig = render_heatmap(grid, (None, None), "Test", shouter_spacing=_FULL_SPACING)
    assert fig is not None
    plt.close(fig)


def test_overlay_adds_node_markers():
    """4 shouter node markers (cyan circles) must be added to the axes."""
    from ghv5.viz import render_heatmap
    grid = [[0.9, 0.1, 0.2], [0.3, 0.8, 0.1], [0.2, 0.1, 0.7]]
    fig = render_heatmap(grid, (None, None), "Test", shouter_spacing=_FULL_SPACING)
    ax = fig.axes[0]
    # Node markers are drawn as Circle patches; count them
    circles = [p for p in ax.patches if hasattr(p, 'radius')]
    assert len(circles) == 4
    plt.close(fig)


def test_overlay_adds_6_lines():
    """6 lines (4 edges + 2 diagonals) must be added when spacing is provided."""
    from ghv5.viz import render_heatmap
    grid = [[0.9, 0.1, 0.2], [0.3, 0.8, 0.1], [0.2, 0.1, 0.7]]
    fig = render_heatmap(grid, (None, None), "Test", shouter_spacing=_FULL_SPACING)
    ax = fig.axes[0]
    lines = ax.lines
    assert len(lines) == 6
    plt.close(fig)


def test_overlay_labels_show_distance():
    """Distance labels must show 'X.Xm' format for known pairs."""
    from ghv5.viz import render_heatmap
    grid = [[0.9, 0.1, 0.2], [0.3, 0.8, 0.1], [0.2, 0.1, 0.7]]
    fig = render_heatmap(grid, (None, None), "Test", shouter_spacing=_FULL_SPACING)
    ax = fig.axes[0]
    label_texts = [t.get_text() for t in ax.texts]
    # Every pair should have a "X.Xm" label
    dist_labels = [t for t in label_texts if t.endswith('m') and '.' in t]
    assert len(dist_labels) == 6
    plt.close(fig)


def test_overlay_partial_shows_ellipsis_for_unknown():
    """Pairs not in spacing dict must show '…' as their label."""
    from ghv5.viz import render_heatmap
    grid = [[0.9, 0.1, 0.2], [0.3, 0.8, 0.1], [0.2, 0.1, 0.7]]
    fig = render_heatmap(grid, (None, None), "Test", shouter_spacing=_PARTIAL_SPACING)
    ax = fig.axes[0]
    label_texts = [t.get_text() for t in ax.texts]
    ellipsis_labels = [t for t in label_texts if t == '…']
    assert len(ellipsis_labels) == 5   # 6 pairs - 1 known = 5 unknown
    plt.close(fig)


def test_diagonals_are_dashed():
    """Diagonal lines (1-3, 2-4) must be dashed; edge lines must be solid."""
    from ghv5.viz import render_heatmap
    grid = [[0.9, 0.1, 0.2], [0.3, 0.8, 0.1], [0.2, 0.1, 0.7]]
    fig = render_heatmap(grid, (None, None), "Test", shouter_spacing=_FULL_SPACING)
    ax = fig.axes[0]
    lines = ax.lines
    dashed = [l for l in lines if l.get_linestyle() in ('--', 'dashed')]
    solid  = [l for l in lines if l.get_linestyle() in ('-',  'solid')]
    assert len(dashed) == 2
    assert len(solid)  == 4
    plt.close(fig)
