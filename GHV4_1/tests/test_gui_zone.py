"""
Tests for gui_zone module — verifies importability, constants, and
class structure without requiring pygame to be initialized.
"""

from __future__ import annotations

import importlib
import sys
import types


# ---------------------------------------------------------------------------
# Importability tests (no pygame needed for constants)
# ---------------------------------------------------------------------------


class TestGuiZoneImport:
    """Module-level constants should be importable without pygame init."""

    def test_import_module(self):
        """gui_zone can be imported."""
        import gui_zone
        assert gui_zone is not None

    def test_window_constants(self):
        from gui_zone import WINDOW_WIDTH, WINDOW_HEIGHT
        assert isinstance(WINDOW_WIDTH, int) and WINDOW_WIDTH > 0
        assert isinstance(WINDOW_HEIGHT, int) and WINDOW_HEIGHT > 0

    def test_area_constants(self):
        from gui_zone import AREA_LEFT, AREA_TOP, AREA_WIDTH, AREA_HEIGHT
        assert AREA_LEFT >= 0
        assert AREA_TOP >= 0
        assert AREA_WIDTH > 0
        assert AREA_HEIGHT > 0

    def test_area_fits_in_window(self):
        from gui_zone import (
            WINDOW_WIDTH, WINDOW_HEIGHT,
            AREA_LEFT, AREA_TOP, AREA_WIDTH, AREA_HEIGHT,
        )
        assert AREA_LEFT + AREA_WIDTH < WINDOW_WIDTH
        assert AREA_TOP + AREA_HEIGHT < WINDOW_HEIGHT


class TestColorConstants:
    """Color tuples should be well-formed RGB or RGBA."""

    def _check_rgb(self, color: tuple, allow_alpha: bool = False):
        expected_len = 4 if allow_alpha else 3
        if allow_alpha:
            assert len(color) in (3, 4), f"Expected RGB or RGBA, got {color}"
        else:
            assert len(color) == 3, f"Expected RGB, got {color}"
        for c in color[:3]:
            assert 0 <= c <= 255, f"Color channel out of range: {c}"

    def test_bg_color(self):
        from gui_zone import COLOR_BG
        self._check_rgb(COLOR_BG)

    def test_text_color(self):
        from gui_zone import COLOR_TEXT
        self._check_rgb(COLOR_TEXT)

    def test_node_color(self):
        from gui_zone import COLOR_NODE
        self._check_rgb(COLOR_NODE)

    def test_occupied_color(self):
        from gui_zone import COLOR_OCCUPIED
        self._check_rgb(COLOR_OCCUPIED)

    def test_empty_color(self):
        from gui_zone import COLOR_EMPTY
        self._check_rgb(COLOR_EMPTY)

    def test_zone_active_has_alpha(self):
        from gui_zone import COLOR_ZONE_ACTIVE
        assert len(COLOR_ZONE_ACTIVE) == 4, "Active zone color should have alpha"
        for c in COLOR_ZONE_ACTIVE:
            assert 0 <= c <= 255

    def test_link_idle_color(self):
        from gui_zone import COLOR_LINK_IDLE
        self._check_rgb(COLOR_LINK_IDLE)

    def test_link_motion_color(self):
        from gui_zone import COLOR_LINK_MOTION
        self._check_rgb(COLOR_LINK_MOTION)


class TestLayoutConstants:
    """Layout definitions should be correct."""

    def test_node_names(self):
        from gui_zone import NODE_NAMES
        assert NODE_NAMES == ["A", "B", "C", "D"]

    def test_link_defs_count(self):
        from gui_zone import LINK_DEFS
        assert len(LINK_DEFS) == 6, "Should have 6 links for 4 nodes"

    def test_link_to_id_matches_defs(self):
        from gui_zone import LINK_DEFS, LINK_TO_ID
        for pair in LINK_DEFS:
            assert pair in LINK_TO_ID, f"Missing ID for link {pair}"

    def test_link_ids_match_presence_detector(self):
        from gui_zone import LINK_TO_ID
        from presence_detector import LINK_IDS
        gui_link_ids = set(LINK_TO_ID.values())
        detector_link_ids = set(LINK_IDS)
        assert gui_link_ids == detector_link_ids

    def test_zone_labels_all_four(self):
        from gui_zone import ZONE_LABELS
        assert set(ZONE_LABELS.keys()) == {"Q1", "Q2", "Q3", "Q4"}

    def test_zone_label_positions_in_unit_range(self):
        from gui_zone import ZONE_LABELS
        for name, (fx, fy) in ZONE_LABELS.items():
            assert 0.0 < fx < 1.0, f"{name} x fraction out of range"
            assert 0.0 < fy < 1.0, f"{name} y fraction out of range"


class TestComputeLayout:
    """Node positions should be at the corners of the area rectangle."""

    def test_returns_all_four_nodes(self):
        from gui_zone import _compute_layout
        positions = _compute_layout()
        assert set(positions.keys()) == {"A", "B", "C", "D"}

    def test_a_is_top_left(self):
        from gui_zone import _compute_layout, AREA_LEFT, AREA_TOP
        positions = _compute_layout()
        assert positions["A"] == (AREA_LEFT, AREA_TOP)

    def test_b_is_top_right(self):
        from gui_zone import _compute_layout, AREA_LEFT, AREA_TOP, AREA_WIDTH
        positions = _compute_layout()
        assert positions["B"] == (AREA_LEFT + AREA_WIDTH, AREA_TOP)

    def test_c_is_bottom_left(self):
        from gui_zone import _compute_layout, AREA_LEFT, AREA_TOP, AREA_HEIGHT
        positions = _compute_layout()
        assert positions["C"] == (AREA_LEFT, AREA_TOP + AREA_HEIGHT)

    def test_d_is_bottom_right(self):
        from gui_zone import (
            _compute_layout, AREA_LEFT, AREA_TOP, AREA_WIDTH, AREA_HEIGHT,
        )
        positions = _compute_layout()
        assert positions["D"] == (AREA_LEFT + AREA_WIDTH, AREA_TOP + AREA_HEIGHT)


class TestZoneDisplayClass:
    """ZoneDisplay class should exist and have the expected interface."""

    def test_class_exists(self):
        from gui_zone import ZoneDisplay
        assert ZoneDisplay is not None

    def test_has_update_method(self):
        from gui_zone import ZoneDisplay
        assert callable(getattr(ZoneDisplay, "update", None))

    def test_has_close_method(self):
        from gui_zone import ZoneDisplay
        assert callable(getattr(ZoneDisplay, "close", None))

    def test_init_signature_accepts_title(self):
        import inspect
        from gui_zone import ZoneDisplay
        sig = inspect.signature(ZoneDisplay.__init__)
        params = list(sig.parameters.keys())
        assert "title" in params


class TestSidebarConstants:
    """Sidebar layout constants should be valid."""

    def test_sidebar_left_after_area(self):
        from gui_zone import SIDEBAR_LEFT, AREA_LEFT, AREA_WIDTH
        assert SIDEBAR_LEFT > AREA_LEFT + AREA_WIDTH

    def test_sidebar_width_positive(self):
        from gui_zone import SIDEBAR_WIDTH
        assert SIDEBAR_WIDTH > 0

    def test_sidebar_fits_in_window(self):
        from gui_zone import SIDEBAR_LEFT, SIDEBAR_WIDTH, WINDOW_WIDTH
        assert SIDEBAR_LEFT + SIDEBAR_WIDTH <= WINDOW_WIDTH
