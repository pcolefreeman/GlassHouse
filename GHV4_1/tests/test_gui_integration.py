"""
Integration tests for main_gui module — verifies importability,
process_frame_gui pure function, and pipeline wiring without
requiring pygame or pyserial.
"""

from __future__ import annotations

import math
import random

_NUM_SUBCARRIERS = 64


# ---------------------------------------------------------------------------
# Synthetic data helpers (aligned with test_zone_integration.py pattern)
# ---------------------------------------------------------------------------


def _make_stable_bytes(base_amplitude: float = 10.0) -> list[int]:
    """Stable CSI bytes — all subcarriers have nearly identical amplitude."""
    real_val = int(base_amplitude)
    bytes_out: list[int] = []
    for _ in range(_NUM_SUBCARRIERS):
        bytes_out.append(0)          # imag
        bytes_out.append(real_val)   # real
    return bytes_out


def _make_disturbed_bytes(
    frame_index: int,
    base_amplitude: float = 10.0,
    variation_scale: float = 8.0,
) -> list[int]:
    """Disturbed CSI bytes — high amplitude variation across subcarriers."""
    bytes_out: list[int] = []
    phase = frame_index * 1.5
    frame_mod = 1.0 + 0.5 * math.sin(frame_index * 0.8)
    for sc_idx in range(_NUM_SUBCARRIERS):
        variation = variation_scale * math.sin(sc_idx * 0.7 + phase)
        amp = max(1.0, base_amplitude + variation * frame_mod)
        real_val = int(amp)
        imag_val = int(variation_scale * math.cos(sc_idx * 1.1 + phase) * frame_mod)
        bytes_out.append(max(-127, min(127, imag_val)))
        bytes_out.append(max(-127, min(127, real_val)))
    return bytes_out


_TX_RX = {
    "AB": ("A", "B"), "AC": ("A", "C"), "AD": ("A", "D"),
    "BC": ("B", "C"), "BD": ("B", "D"), "CD": ("C", "D"),
}


def _make_s02_line(
    link_id: str, seq: int = 1, raw_bytes: list[int] | None = None,
    rssi: int = -50,
) -> str:
    """Build an S02-format CSV line from components."""
    tx, rx = _TX_RX[link_id]
    if raw_bytes is None:
        raw_bytes = _make_stable_bytes()
    byte_str = " ".join(str(b) for b in raw_bytes)
    return f"CSI_DATA,{seq},{tx},{rx},{link_id},{rssi},{len(raw_bytes)},{byte_str}"


# ---------------------------------------------------------------------------
# Importability tests
# ---------------------------------------------------------------------------

class TestMainGuiImport:
    """Module should be importable without pygame or pyserial."""

    def test_import_process_frame_gui(self):
        from main_gui import process_frame_gui
        assert callable(process_frame_gui)

    def test_import_build_parser(self):
        from main_gui import build_parser
        assert callable(build_parser)


class TestBuildParser:
    """CLI argument parser should have expected arguments."""

    def test_parser_has_port_arg(self):
        from main_gui import build_parser
        parser = build_parser()
        args = parser.parse_args(["--port", "COM3"])
        assert args.port == "COM3"

    def test_parser_defaults(self):
        from main_gui import build_parser
        parser = build_parser()
        args = parser.parse_args(["--port", "COM3"])
        assert args.baud == 921600
        assert args.threshold == 0.005
        assert args.window == 20

    def test_parser_custom_values(self):
        from main_gui import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "--port", "/dev/ttyUSB0",
            "--baud", "115200",
            "--threshold", "0.01",
            "--window", "30",
        ])
        assert args.port == "/dev/ttyUSB0"
        assert args.baud == 115200
        assert args.threshold == 0.01
        assert args.window == 30


# ---------------------------------------------------------------------------
# process_frame_gui tests
# ---------------------------------------------------------------------------

class TestProcessFrameGui:
    """Pure function process_frame_gui should work without pygame/serial."""

    def test_returns_none_for_non_csi(self):
        from main_gui import process_frame_gui
        from presence_detector import PresenceEngine
        from zone_detector import ZoneDetector

        engine = PresenceEngine()
        detector = ZoneDetector(engine)
        result = process_frame_gui("not a csi line", engine, detector)
        assert result is None

    def test_returns_none_for_s01_format(self):
        from main_gui import process_frame_gui
        from presence_detector import PresenceEngine
        from zone_detector import ZoneDetector

        engine = PresenceEngine()
        detector = ZoneDetector(engine)
        # S01 format — no link_id
        line = "CSI_DATA,1,AA:BB:CC:DD:EE:FF,-40,4,10 20 30 40"
        result = process_frame_gui(line, engine, detector)
        assert result is None

    def test_returns_none_for_unknown_link(self):
        from main_gui import process_frame_gui
        from presence_detector import PresenceEngine
        from zone_detector import ZoneDetector

        engine = PresenceEngine()
        detector = ZoneDetector(engine)
        line = "CSI_DATA,1,X,Y,XY,-40,4,10 20 30 40"
        result = process_frame_gui(line, engine, detector)
        assert result is None

    def test_returns_dict_for_valid_s02(self):
        from main_gui import process_frame_gui
        from presence_detector import PresenceEngine, RoomState
        from zone_detector import ZoneDetector, ZoneResult

        engine = PresenceEngine()
        detector = ZoneDetector(engine)
        line = _make_s02_line("AB")
        result = process_frame_gui(line, engine, detector)

        assert result is not None
        assert "room_state" in result
        assert "zone_result" in result
        assert "link_states" in result
        assert "link_id" in result
        assert "turbulence" in result
        assert isinstance(result["room_state"], RoomState)
        assert isinstance(result["zone_result"], ZoneResult)
        assert result["link_id"] == "AB"

    def test_result_has_all_link_states(self):
        from main_gui import process_frame_gui
        from presence_detector import PresenceEngine, LINK_IDS
        from zone_detector import ZoneDetector

        engine = PresenceEngine()
        detector = ZoneDetector(engine)
        line = _make_s02_line("AB")
        result = process_frame_gui(line, engine, detector)
        assert result is not None
        for lid in LINK_IDS:
            assert lid in result["link_states"]


class TestProcessFrameGuiPipeline:
    """Full pipeline through process_frame_gui should produce correct state."""

    def test_stable_data_stays_empty(self):
        from main_gui import process_frame_gui
        from presence_detector import PresenceEngine, RoomState, LINK_IDS
        from zone_detector import ZoneDetector

        engine = PresenceEngine(window_size=10)
        detector = ZoneDetector(engine)

        # Feed 30 stable frames per link
        for frame in range(30):
            raw = _make_stable_bytes()
            for lid in LINK_IDS:
                line = _make_s02_line(lid, seq=frame, raw_bytes=raw)
                process_frame_gui(line, engine, detector)

        assert engine.room_state == RoomState.EMPTY

    def test_disturbed_data_detects_occupied(self):
        from main_gui import process_frame_gui
        from presence_detector import PresenceEngine, RoomState, LINK_IDS
        from zone_detector import ZoneDetector

        engine = PresenceEngine(window_size=10)
        detector = ZoneDetector(engine)

        # Feed 30 frames — AB and AC disturbed, others stable
        for frame in range(30):
            for lid in LINK_IDS:
                if lid in ("AB", "AC"):
                    raw = _make_disturbed_bytes(frame)
                else:
                    raw = _make_stable_bytes()
                line = _make_s02_line(lid, seq=frame, raw_bytes=raw)
                process_frame_gui(line, engine, detector)

        # Check engine state directly — any-link-OR aggregation
        assert engine.room_state == RoomState.OCCUPIED

    def test_disturbed_ab_ac_gives_q1(self):
        from main_gui import process_frame_gui
        from presence_detector import PresenceEngine, RoomState, LINK_IDS
        from zone_detector import ZoneDetector, Zone

        engine = PresenceEngine(window_size=10)
        detector = ZoneDetector(engine)

        for frame in range(30):
            for lid in LINK_IDS:
                if lid in ("AB", "AC"):
                    raw = _make_disturbed_bytes(frame)
                else:
                    raw = _make_stable_bytes()
                line = _make_s02_line(lid, seq=frame, raw_bytes=raw)
                process_frame_gui(line, engine, detector)

        # Check engine + detector state directly
        assert engine.room_state == RoomState.OCCUPIED
        assert detector.estimate().zone == Zone.Q1

    def test_full_lifecycle(self):
        """EMPTY → OCCUPIED/Q1 → EMPTY lifecycle through process_frame_gui."""
        from main_gui import process_frame_gui
        from presence_detector import PresenceEngine, RoomState, LINK_IDS
        from zone_detector import ZoneDetector, Zone

        engine = PresenceEngine(window_size=10)
        detector = ZoneDetector(engine)

        # Phase 1: stable → EMPTY
        for frame in range(30):
            raw = _make_stable_bytes()
            for lid in LINK_IDS:
                line = _make_s02_line(lid, seq=frame, raw_bytes=raw)
                process_frame_gui(line, engine, detector)

        assert engine.room_state == RoomState.EMPTY

        # Phase 2: disturb AB+AC → OCCUPIED/Q1
        for frame in range(30, 60):
            for lid in LINK_IDS:
                if lid in ("AB", "AC"):
                    raw = _make_disturbed_bytes(frame)
                else:
                    raw = _make_stable_bytes()
                line = _make_s02_line(lid, seq=frame, raw_bytes=raw)
                process_frame_gui(line, engine, detector)

        assert engine.room_state == RoomState.OCCUPIED
        assert detector.estimate().zone == Zone.Q1

        # Phase 3: stable again → EMPTY
        for frame in range(60, 90):
            raw = _make_stable_bytes()
            for lid in LINK_IDS:
                line = _make_s02_line(lid, seq=frame, raw_bytes=raw)
                process_frame_gui(line, engine, detector)

        assert engine.room_state == RoomState.EMPTY

    def test_turbulence_is_float(self):
        from main_gui import process_frame_gui
        from presence_detector import PresenceEngine
        from zone_detector import ZoneDetector

        engine = PresenceEngine()
        detector = ZoneDetector(engine)
        line = _make_s02_line("AB")
        result = process_frame_gui(line, engine, detector)
        assert result is not None
        assert isinstance(result["turbulence"], float)
