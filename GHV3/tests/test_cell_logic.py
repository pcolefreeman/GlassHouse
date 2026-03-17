# tests/test_cell_logic.py
"""Tests for ghv3_1.cell_logic — pure helpers with no UI dependency."""
import pytest
from ghv3_1.cell_logic import (
    validate_width, validate_depth, validate_zone,
    build_label, first_cell,
)


# ── validate_width ──────────────────────────────────────────────────────────

def test_validate_width_valid_float():
    assert validate_width("6.0") == 6.0

def test_validate_width_valid_int_string():
    assert validate_width("4") == 4.0

def test_validate_width_zero_returns_none():
    assert validate_width("0") is None

def test_validate_width_negative_returns_none():
    assert validate_width("-1.5") is None

def test_validate_width_empty_returns_none():
    assert validate_width("") is None

def test_validate_width_letters_returns_none():
    assert validate_width("abc") is None

def test_validate_width_whitespace_returns_none():
    assert validate_width("   ") is None


# ── validate_depth ──────────────────────────────────────────────────────────

def test_validate_depth_valid():
    assert validate_depth("4.0") == 4.0

def test_validate_depth_zero_returns_none():
    assert validate_depth("0") is None

def test_validate_depth_negative_returns_none():
    assert validate_depth("-0.5") is None

def test_validate_depth_empty_returns_none():
    assert validate_depth("") is None

def test_validate_depth_whitespace_returns_none():
    assert validate_depth("   ") is None


# ── validate_zone ───────────────────────────────────────────────────────────

def test_validate_zone_zero():
    assert validate_zone("0") == 0

def test_validate_zone_positive():
    assert validate_zone("5") == 5

def test_validate_zone_negative_returns_none():
    assert validate_zone("-1") is None

def test_validate_zone_float_string_returns_none():
    assert validate_zone("1.5") is None

def test_validate_zone_empty_returns_none():
    assert validate_zone("") is None


# ── build_label ─────────────────────────────────────────────────────────────

def test_build_label_empty_set():
    assert build_label(set()) == "empty"

def test_build_label_single_cell():
    assert build_label({(1, 0)}) == "r1c0"

def test_build_label_two_cells_sorted():
    assert build_label({(2, 2), (0, 0)}) == "r0c0+r2c2"

def test_build_label_row_major_sort():
    # (2,0) comes after (0,1) because row 0 < row 2
    assert build_label({(2, 0), (0, 1)}) == "r0c1+r2c0"

def test_build_label_all_nine_cells():
    cells = {(r, c) for r in range(3) for c in range(3)}
    label = build_label(cells)
    parts = label.split("+")
    assert len(parts) == 9
    assert parts[0] == "r0c0"
    assert parts[-1] == "r2c2"


# ── first_cell ──────────────────────────────────────────────────────────────

def test_first_cell_empty_returns_sentinel():
    assert first_cell(set()) == (-1, -1)

def test_first_cell_single():
    assert first_cell({(1, 2)}) == (1, 2)

def test_first_cell_multi_picks_smallest_row_then_col():
    # (0,2) has smaller row than (1,0) and (2,1)
    assert first_cell({(2, 1), (0, 2), (1, 0)}) == (0, 2)

def test_first_cell_same_row_picks_smallest_col():
    assert first_cell({(1, 2), (1, 0)}) == (1, 0)
