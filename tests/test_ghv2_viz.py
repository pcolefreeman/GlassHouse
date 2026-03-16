import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless for CI
import matplotlib.pyplot as plt
import matplotlib.figure
import pytest

# ── import guard (file doesn't exist yet — tests will fail to import) ──────────
import ghv2_viz


GRID_CONF = np.array([
    [0.9, 0.3, 0.8],
    [0.2, 0.75, 0.1],
    [0.6, 0.4, 0.95],
], dtype=float)

GRID_RAW = np.array([
    [-65.0, -72.0, -58.0],
    [-80.0, -61.0, -75.0],
    [-55.0, -68.0, -71.0],
], dtype=float)

AREA = (6.0, 4.0)


# ── return type ────────────────────────────────────────────────────────────────

def test_render_heatmap_no_ax_returns_figure():
    fig = ghv2_viz.render_heatmap(GRID_CONF, AREA, "Test", mode="confidence")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')


def test_render_heatmap_with_ax_returns_none():
    fig_ext, ax = plt.subplots()
    result = ghv2_viz.render_heatmap(GRID_CONF, AREA, "Test", mode="confidence", ax=ax)
    assert result is None
    plt.close('all')


def test_raw_mode_returns_figure():
    fig = ghv2_viz.render_heatmap(GRID_RAW, AREA, "RSSI", mode="raw")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')


def test_raw_mode_with_ax_returns_none():
    fig_ext, ax = plt.subplots()
    result = ghv2_viz.render_heatmap(GRID_RAW, AREA, "RSSI", mode="raw", ax=ax)
    assert result is None
    plt.close('all')


def test_no_area_dims_does_not_raise():
    """(None, None) area_dims should produce cell-index labels without error."""
    fig = ghv2_viz.render_heatmap(GRID_CONF, (None, None), "No dims")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')


def test_all_below_threshold_does_not_raise():
    grid = np.zeros((3, 3), dtype=float)  # all confidence = 0
    fig = ghv2_viz.render_heatmap(grid, AREA, "All dark", threshold=0.70)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')


def test_all_above_threshold_does_not_raise():
    grid = np.ones((3, 3), dtype=float)  # all confidence = 1
    fig = ghv2_viz.render_heatmap(grid, AREA, "All lit", threshold=0.70)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')


def test_raw_mode_uniform_values_does_not_raise():
    """Uniform grid (max == min) should not divide by zero."""
    grid = np.full((3, 3), -70.0)
    fig = ghv2_viz.render_heatmap(grid, AREA, "Uniform", mode="raw")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')
