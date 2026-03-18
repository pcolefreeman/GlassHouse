"""Tests for ghv4.config — validates internal consistency of constants."""
from math import comb

from ghv4.config import (
    ACTIVE_SHOUTER_IDS,
    CELL_LABELS,
    GRID_POS,
    NULL_PDIFF_INDICES,
    NULL_SUBCARRIER_INDICES,
    PAIR_KEYS,
    SUBCARRIERS,
    META_COLS,
    EXPECTED_COLS,
    BAUD_RATE,
    MAGIC_LISTENER,
    MAGIC_SHOUTER,
    MAGIC_RANGING,
    MAGIC_CSI_SNAP,
    LISTENER_HDR_SIZE,
    SHOUTER_HDR_SIZE,
    RANGING_PAYLOAD_SIZE,
    CSI_SNAP_HDR_SIZE,
    BUCKET_MS,
    MAX_LOG_LINES,
)


def test_null_subs_subset_of_null_pdiff():
    """NULL_SUBCARRIER_INDICES must be a subset of NULL_PDIFF_INDICES."""
    assert NULL_SUBCARRIER_INDICES.issubset(NULL_PDIFF_INDICES)


def test_pair_keys_count():
    """len(PAIR_KEYS) must equal C(len(ACTIVE_SHOUTER_IDS), 2)."""
    expected = comb(len(ACTIVE_SHOUTER_IDS), 2)
    assert len(PAIR_KEYS) == expected


def test_cell_labels_count():
    """9 cells for the 3x3 grid."""
    assert len(CELL_LABELS) == 9


def test_cell_labels_format():
    """Each label matches r{0-2}c{0-2}."""
    import re
    for label in CELL_LABELS:
        assert re.fullmatch(r"r[0-2]c[0-2]", label), f"Bad label: {label}"


def test_grid_pos_covers_all_cells():
    """GRID_POS must have entries 0-8 mapping to valid (row, col)."""
    assert len(GRID_POS) == 9
    for i in range(9):
        r, c = GRID_POS[i]
        assert 0 <= r <= 2 and 0 <= c <= 2


def test_pair_keys_format():
    """Each pair key is 'X-Y' where X < Y and both are in ACTIVE_SHOUTER_IDS."""
    for key in PAIR_KEYS:
        parts = key.split("-")
        assert len(parts) == 2
        a, b = int(parts[0]), int(parts[1])
        assert a < b
        assert a in ACTIVE_SHOUTER_IDS and b in ACTIVE_SHOUTER_IDS


def test_null_subcarrier_indices_within_range():
    """All null subcarrier indices must be < SUBCARRIERS."""
    for idx in NULL_SUBCARRIER_INDICES:
        assert 0 <= idx < SUBCARRIERS


def test_null_pdiff_indices_within_range():
    """All null pdiff indices must be < SUBCARRIERS."""
    for idx in NULL_PDIFF_INDICES:
        assert 0 <= idx < SUBCARRIERS


def test_magic_bytes_are_two_bytes():
    """All magic byte constants must be exactly 2 bytes."""
    for name, val in [
        ("MAGIC_LISTENER", MAGIC_LISTENER),
        ("MAGIC_SHOUTER", MAGIC_SHOUTER),
        ("MAGIC_RANGING", MAGIC_RANGING),
        ("MAGIC_CSI_SNAP", MAGIC_CSI_SNAP),
    ]:
        assert len(val) == 2, f"{name} is {len(val)} bytes, expected 2"


def test_meta_cols_count():
    """META_COLS has exactly 5 entries."""
    assert len(META_COLS) == 5


def test_baud_rate_positive():
    assert BAUD_RATE > 0


def test_header_sizes_positive():
    assert LISTENER_HDR_SIZE > 0
    assert SHOUTER_HDR_SIZE > 0
    assert RANGING_PAYLOAD_SIZE > 0
    assert CSI_SNAP_HDR_SIZE > 0
