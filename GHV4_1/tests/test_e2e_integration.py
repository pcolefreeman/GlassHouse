"""
End-to-end integration tests — verifies the complete system wires together
from CSI parsing through zone detection to GUI-ready state.

These tests validate cross-module boundaries that no individual slice's
tests cover.  All tests use synthetic data — no hardware required.
"""

from __future__ import annotations

import inspect
import math
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Project structure constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PYTHON_DIR = _PROJECT_ROOT / "python"
_FIRMWARE_DIR_COORDINATOR = _PROJECT_ROOT / "coordinator"
_FIRMWARE_DIR_PERIMETER = _PROJECT_ROOT / "perimeter_node"


# ---------------------------------------------------------------------------
# Synthetic data helpers (aligned with test_zone_integration.py)
# ---------------------------------------------------------------------------

_NUM_SUBCARRIERS = 64


def _make_stable_bytes(base_amplitude: float = 10.0) -> list[int]:
    real_val = int(base_amplitude)
    bytes_out: list[int] = []
    for _ in range(_NUM_SUBCARRIERS):
        bytes_out.append(0)
        bytes_out.append(real_val)
    return bytes_out


def _make_disturbed_bytes(frame_index: int) -> list[int]:
    bytes_out: list[int] = []
    phase = frame_index * 1.5
    frame_mod = 1.0 + 0.5 * math.sin(frame_index * 0.8)
    for sc_idx in range(_NUM_SUBCARRIERS):
        variation = 8.0 * math.sin(sc_idx * 0.7 + phase)
        amp = max(1.0, 10.0 + variation * frame_mod)
        real_val = int(amp)
        imag_val = int(8.0 * math.cos(sc_idx * 1.1 + phase) * frame_mod)
        bytes_out.append(max(-127, min(127, imag_val)))
        bytes_out.append(max(-127, min(127, real_val)))
    return bytes_out


def _make_s02_line(link_id: str, seq: int, raw_bytes: list[int]) -> str:
    tx, rx = link_id[0], link_id[1]
    byte_str = " ".join(str(b) for b in raw_bytes)
    return f"CSI_DATA,{seq},{tx},{rx},{link_id},-50,{len(raw_bytes)},{byte_str}"


# ---------------------------------------------------------------------------
# Import chain tests
# ---------------------------------------------------------------------------

class TestImportChain:
    """Verify the complete import chain works without hardware."""

    def test_serial_csi_reader_import(self):
        from serial_csi_reader import parse_csi_line, compute_amplitudes
        assert callable(parse_csi_line)
        assert callable(compute_amplitudes)

    def test_csi_features_import(self):
        from csi_features import select_subcarriers, compute_turbulence
        assert callable(select_subcarriers)
        assert callable(compute_turbulence)

    def test_presence_detector_import(self):
        from presence_detector import (
            PresenceEngine, RoomState, LinkState, LinkDetector, LINK_IDS,
        )
        assert PresenceEngine is not None
        assert len(LINK_IDS) == 6

    def test_zone_detector_import(self):
        from zone_detector import (
            ZoneDetector, Zone, ZoneResult, LINK_ZONE_WEIGHTS,
        )
        assert ZoneDetector is not None
        assert len(Zone) == 4

    def test_gui_zone_import(self):
        from gui_zone import (
            ZoneDisplay, WINDOW_WIDTH, WINDOW_HEIGHT,
            COLOR_BG, NODE_NAMES, LINK_DEFS, ZONE_LABELS,
        )
        assert ZoneDisplay is not None
        assert len(NODE_NAMES) == 4

    def test_main_gui_import(self):
        from main_gui import process_frame_gui, build_parser
        assert callable(process_frame_gui)
        assert callable(build_parser)

    def test_main_presence_import(self):
        from main_presence import process_line, format_link_detail
        assert callable(process_line)
        assert callable(format_link_detail)

    def test_main_zone_import(self):
        from main_zone import process_line_zone, format_zone_detail
        assert callable(process_line_zone)
        assert callable(format_zone_detail)


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """Verify the complete data pipeline from raw line to GUI state."""

    def test_parse_to_gui_state(self):
        """Single line through the complete pipeline."""
        from serial_csi_reader import parse_csi_line, compute_amplitudes
        from csi_features import select_subcarriers, compute_turbulence
        from presence_detector import PresenceEngine, RoomState
        from zone_detector import ZoneDetector

        engine = PresenceEngine()
        detector = ZoneDetector(engine)

        raw_bytes = _make_stable_bytes()
        line = _make_s02_line("AB", 1, raw_bytes)

        # Step 1: parse
        parsed = parse_csi_line(line)
        assert parsed is not None
        assert parsed["link_id"] == "AB"

        # Step 2: amplitudes
        amps = compute_amplitudes(parsed["raw_bytes"])
        assert len(amps) > 0

        # Step 3: features
        selected = select_subcarriers(amps)
        turb = compute_turbulence(selected)
        assert isinstance(turb, float)

        # Step 4: presence engine
        state = engine.update("AB", turb)
        assert isinstance(state, RoomState)

        # Step 5: zone detector
        result = detector.estimate()
        assert result is not None
        assert hasattr(result, "zone")
        assert hasattr(result, "scores")
        assert hasattr(result, "confidence")

        # Step 6: link states for GUI
        link_states = engine.get_link_states()
        assert "AB" in link_states

    def test_process_frame_gui_complete_flow(self):
        """process_frame_gui wires the entire pipeline correctly."""
        from main_gui import process_frame_gui
        from presence_detector import PresenceEngine, RoomState, LINK_IDS
        from zone_detector import ZoneDetector, Zone

        engine = PresenceEngine(window_size=10)
        detector = ZoneDetector(engine)

        # Phase 1: stable data → EMPTY
        for frame in range(25):
            raw = _make_stable_bytes()
            for lid in LINK_IDS:
                line = _make_s02_line(lid, frame, raw)
                result = process_frame_gui(line, engine, detector)
                assert result is not None

        assert engine.room_state == RoomState.EMPTY

        # Phase 2: disturb AB+AC → OCCUPIED/Q1
        for frame in range(25, 50):
            for lid in LINK_IDS:
                if lid in ("AB", "AC"):
                    raw = _make_disturbed_bytes(frame)
                else:
                    raw = _make_stable_bytes()
                line = _make_s02_line(lid, frame, raw)
                result = process_frame_gui(line, engine, detector)

        assert engine.room_state == RoomState.OCCUPIED
        zone_result = detector.estimate()
        assert zone_result.zone == Zone.Q1
        assert zone_result.confidence > 1.0

        # Verify link states match expected pattern
        link_states = engine.get_link_states()
        assert link_states["AB"]["state"] == "MOTION"
        assert link_states["AC"]["state"] == "MOTION"
        assert link_states["CD"]["state"] == "IDLE"

    def test_all_four_zones_through_gui_pipeline(self):
        """Each quadrant can be identified through process_frame_gui."""
        from main_gui import process_frame_gui
        from presence_detector import PresenceEngine, LINK_IDS
        from zone_detector import ZoneDetector, Zone

        zone_links = {
            Zone.Q1: ("AB", "AC"),
            Zone.Q2: ("AB", "BD"),
            Zone.Q3: ("CD", "AC"),
            Zone.Q4: ("CD", "BD"),
        }

        for expected_zone, disturbed_pair in zone_links.items():
            engine = PresenceEngine(window_size=10)
            detector = ZoneDetector(engine)

            for frame in range(30):
                for lid in LINK_IDS:
                    if lid in disturbed_pair:
                        raw = _make_disturbed_bytes(frame)
                    else:
                        raw = _make_stable_bytes()
                    line = _make_s02_line(lid, frame, raw)
                    process_frame_gui(line, engine, detector)

            result = detector.estimate()
            assert result.zone == expected_zone, (
                f"Expected {expected_zone} but got {result.zone} "
                f"for disturbed links {disturbed_pair}"
            )


# ---------------------------------------------------------------------------
# Project structure verification
# ---------------------------------------------------------------------------

class TestProjectStructure:
    """Verify all expected files exist."""

    def test_firmware_coordinator_exists(self):
        assert (_FIRMWARE_DIR_COORDINATOR / "coordinator.ino").exists()

    def test_firmware_perimeter_exists(self):
        assert (_FIRMWARE_DIR_PERIMETER / "perimeter_node.ino").exists()

    def test_python_modules_exist(self):
        expected = [
            "serial_csi_reader.py",
            "csi_features.py",
            "presence_detector.py",
            "zone_detector.py",
            "gui_zone.py",
            "main_gui.py",
            "main_presence.py",
            "main_zone.py",
        ]
        for fname in expected:
            assert (_PYTHON_DIR / fname).exists(), f"Missing: python/{fname}"

    def test_requirements_txt_exists(self):
        req_file = _PYTHON_DIR / "requirements.txt"
        assert req_file.exists()

    def test_requirements_has_pyserial(self):
        req_file = _PYTHON_DIR / "requirements.txt"
        content = req_file.read_text()
        assert "pyserial" in content

    def test_requirements_has_pygame(self):
        req_file = _PYTHON_DIR / "requirements.txt"
        content = req_file.read_text()
        assert "pygame" in content

    def test_test_files_exist(self):
        tests_dir = _PROJECT_ROOT / "tests"
        expected = [
            "test_csi_parser.py",
            "test_csi_features.py",
            "test_presence_detector.py",
            "test_presence_integration.py",
            "test_zone_detector.py",
            "test_zone_integration.py",
            "test_gui_zone.py",
            "test_gui_integration.py",
            "test_e2e_integration.py",
        ]
        for fname in expected:
            assert (tests_dir / fname).exists(), f"Missing: tests/{fname}"


# ---------------------------------------------------------------------------
# API surface consistency
# ---------------------------------------------------------------------------

class TestAPISurface:
    """Verify key APIs are consistent across modules."""

    def test_link_ids_consistent(self):
        """LINK_IDS in presence_detector matches link definitions in gui_zone."""
        from presence_detector import LINK_IDS
        from gui_zone import LINK_TO_ID
        from zone_detector import LINK_ZONE_WEIGHTS

        gui_ids = set(LINK_TO_ID.values())
        zone_ids = set(LINK_ZONE_WEIGHTS.keys())
        det_ids = set(LINK_IDS)

        assert gui_ids == det_ids, f"GUI link IDs {gui_ids} != detector {det_ids}"
        assert zone_ids == det_ids, f"Zone link IDs {zone_ids} != detector {det_ids}"

    def test_zone_enum_consistent(self):
        """Zone enum in zone_detector matches zone labels in gui_zone."""
        from zone_detector import Zone
        from gui_zone import ZONE_LABELS

        zone_names = {z.value for z in Zone}
        label_names = set(ZONE_LABELS.keys())
        assert zone_names == label_names

    def test_room_state_enum_values(self):
        """RoomState enum values are the strings used in GUI display."""
        from presence_detector import RoomState
        assert RoomState.EMPTY.value == "EMPTY"
        assert RoomState.OCCUPIED.value == "OCCUPIED"

    def test_link_state_enum_values(self):
        """LinkState enum values match what get_link_states() returns."""
        from presence_detector import LinkState
        assert LinkState.IDLE.value == "IDLE"
        assert LinkState.MOTION.value == "MOTION"

    def test_process_frame_gui_return_schema(self):
        """process_frame_gui return dict has the right keys."""
        from main_gui import process_frame_gui
        from presence_detector import PresenceEngine, RoomState
        from zone_detector import ZoneDetector, ZoneResult

        engine = PresenceEngine()
        detector = ZoneDetector(engine)
        line = _make_s02_line("AB", 1, _make_stable_bytes())
        result = process_frame_gui(line, engine, detector)

        assert result is not None
        expected_keys = {"room_state", "zone_result", "link_states", "link_id", "turbulence"}
        assert set(result.keys()) == expected_keys
