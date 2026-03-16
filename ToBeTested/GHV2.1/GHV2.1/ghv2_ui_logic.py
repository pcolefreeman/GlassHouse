# ghv2_ui_logic.py
"""Pure helper functions for the GHV2 data collection UI.

No UI imports — safe to test without a display.
"""
from __future__ import annotations


def _validate_positive_float(s: str) -> float | None:
    """Return positive float, or None if s is empty, zero, negative, or non-numeric."""
    try:
        v = float(s)
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


def validate_width(s: str) -> float | None:
    return _validate_positive_float(s)


def validate_depth(s: str) -> float | None:
    return _validate_positive_float(s)


def validate_zone(s: str) -> int | None:
    """Return non-negative int, or None if s is empty, negative, or non-integer."""
    try:
        v = int(s)          # raises ValueError if s is "1.5" or ""
        return v if v >= 0 else None
    except (ValueError, TypeError):
        return None


def build_label(selected: set[tuple[int, int]]) -> str:
    """Build the occupancy label string from a set of (row, col) tuples.

    Returns 'empty' for an empty set. Otherwise returns '+'-joined cell names
    sorted in row-major order, e.g. 'r0c0+r2c2'.
    """
    if not selected:
        return "empty"
    return "+".join(f"r{r}c{c}" for r, c in sorted(selected))


def first_cell(selected: set[tuple[int, int]]) -> tuple[int, int]:
    """Return the first selected cell in row-major order, or (0, 0) if none."""
    if not selected:
        return (0, 0)
    return min(selected)
