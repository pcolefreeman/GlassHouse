"""
Integration tests for the CSI zone detection pipeline.

Feeds synthetic S02-format CSV lines through the full pipeline:
    parse_csi_line → compute_amplitudes → select_subcarriers →
    compute_turbulence → PresenceEngine.update() → ZoneDetector.estimate()

Proves per-quadrant identification, zone transitions, and full lifecycle
(EMPTY → OCCUPIED/Q1 → EMPTY) without hardware.
"""

from __future__ import annotations

import math
import os
import sys

# Ensure the python/ directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from serial_csi_reader import parse_csi_line, compute_amplitudes
from csi_features import select_subcarriers, compute_turbulence
from presence_detector import (
    PresenceEngine,
    RoomState,
    LINK_IDS,
    DEFAULT_WINDOW_SIZE,
)
from zone_detector import ZoneDetector, Zone, ZoneResult
from main_zone import process_line_zone, format_zone_detail


# ---------------------------------------------------------------------------
# Helpers — synthetic CSI data generation
# (Copied from test_presence_integration.py — test utilities, not prod code)
# ---------------------------------------------------------------------------

_NUM_SUBCARRIERS = 64
_NUM_BYTES = _NUM_SUBCARRIERS * 2  # 128 bytes


def _make_stable_bytes(base_amplitude: float = 10.0) -> list[int]:
    """Generate raw CSI bytes where all subcarriers have nearly identical amplitude.

    Stable amplitudes → low turbulence (CV ≈ 0) → IDLE.
    """
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
    """Generate raw CSI bytes with high amplitude variation across subcarriers.

    Disturbed amplitudes → high turbulence (CV >> 0) → MOTION.
    """
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


def _make_s02_line(
    seq: int,
    tx: str,
    rx: str,
    link_id: str,
    raw_bytes: list[int],
    rssi: int = -50,
) -> str:
    """Build an S02-format CSV line from components."""
    byte_str = " ".join(str(b) for b in raw_bytes)
    return f"CSI_DATA,{seq},{tx},{rx},{link_id},{rssi},{len(raw_bytes)},{byte_str}"


_TX_RX = {
    "AB": ("A", "B"), "AC": ("A", "C"), "AD": ("A", "D"),
    "BC": ("B", "C"), "BD": ("B", "D"), "CD": ("C", "D"),
}


def _feed_stable_frames(
    engine: PresenceEngine,
    num_frames: int,
    link_ids: list[str] | None = None,
    seq_start: int = 0,
) -> list[RoomState]:
    """Feed stable (low turbulence) frames through the pipeline."""
    if link_ids is None:
        link_ids = LINK_IDS
    states: list[RoomState] = []
    seq = seq_start
    for _ in range(num_frames):
        raw_bytes = _make_stable_bytes()
        for lid in link_ids:
            tx, rx = _TX_RX[lid]
            line = _make_s02_line(seq, tx, rx, lid, raw_bytes)
            parsed = parse_csi_line(line)
            assert parsed is not None
            amps = compute_amplitudes(parsed["raw_bytes"])
            selected = select_subcarriers(amps)
            turb = compute_turbulence(selected)
            state = engine.update(lid, turb)
            seq += 1
        states.append(state)
    return states


def _feed_disturbed_frames(
    engine: PresenceEngine,
    num_frames: int,
    disturbed_links: list[str],
    all_links: list[str] | None = None,
    seq_start: int = 1000,
) -> list[RoomState]:
    """Feed frames where some links are disturbed, others stable."""
    if all_links is None:
        all_links = LINK_IDS
    states: list[RoomState] = []
    seq = seq_start
    for frame_idx in range(num_frames):
        for lid in all_links:
            tx, rx = _TX_RX[lid]
            if lid in disturbed_links:
                raw_bytes = _make_disturbed_bytes(frame_index=frame_idx)
            else:
                raw_bytes = _make_stable_bytes()
            line = _make_s02_line(seq, tx, rx, lid, raw_bytes)
            parsed = parse_csi_line(line)
            assert parsed is not None
            amps = compute_amplitudes(parsed["raw_bytes"])
            selected = select_subcarriers(amps)
            turb = compute_turbulence(selected)
            state = engine.update(lid, turb)
            seq += 1
        states.append(state)
    return states


# ---------------------------------------------------------------------------
# Per-quadrant identification through full pipeline
# ---------------------------------------------------------------------------


class TestPerQuadrantIdentification:
    """Disturbing the 2 edge links adjacent to a quadrant should identify that zone."""

    # Room layout:
    #   A ---AB--- B        Q1=top-left(A)  Q2=top-right(B)
    #   |         |
    #   AC        BD        Q3=bot-left(C)  Q4=bot-right(D)
    #   |         |
    #   C ---CD--- D

    def _identify_zone(self, disturbed_links: list[str]) -> Zone | None:
        """Establish baseline, disturb specified links, return detected zone."""
        engine = PresenceEngine()
        detector = ZoneDetector(engine)

        # Phase 1: stable baseline
        _feed_stable_frames(engine, num_frames=25)
        assert engine.room_state == RoomState.EMPTY

        # Phase 2: disturb target links
        states = _feed_disturbed_frames(
            engine, num_frames=25, disturbed_links=disturbed_links
        )
        assert RoomState.OCCUPIED in states, (
            f"Never reached OCCUPIED with {disturbed_links}"
        )

        result = detector.estimate()
        return result.zone

    def test_q1_ab_ac(self):
        """Disturbing AB + AC → Q1 (top-left, near node A)."""
        zone = self._identify_zone(["AB", "AC"])
        assert zone == Zone.Q1, f"Expected Q1, got {zone}"

    def test_q2_ab_bd(self):
        """Disturbing AB + BD → Q2 (top-right, near node B)."""
        zone = self._identify_zone(["AB", "BD"])
        assert zone == Zone.Q2, f"Expected Q2, got {zone}"

    def test_q3_cd_ac(self):
        """Disturbing CD + AC → Q3 (bot-left, near node C)."""
        zone = self._identify_zone(["CD", "AC"])
        assert zone == Zone.Q3, f"Expected Q3, got {zone}"

    def test_q4_cd_bd(self):
        """Disturbing CD + BD → Q4 (bot-right, near node D)."""
        zone = self._identify_zone(["CD", "BD"])
        assert zone == Zone.Q4, f"Expected Q4, got {zone}"


# ---------------------------------------------------------------------------
# Full lifecycle: EMPTY → OCCUPIED(Q1) → EMPTY
# ---------------------------------------------------------------------------


class TestZoneLifecycle:
    """Prove the complete EMPTY/None → OCCUPIED/Q1 → EMPTY/None lifecycle."""

    def test_full_lifecycle_with_zone(self):
        """Stable → EMPTY/None → disturb AB+AC → OCCUPIED/Q1 → stable → EMPTY/None."""
        engine = PresenceEngine()
        detector = ZoneDetector(engine)

        # Phase 1: Stable baseline — EMPTY, no zone
        _feed_stable_frames(engine, num_frames=25)
        assert engine.room_state == RoomState.EMPTY
        result = detector.estimate()
        # Zone should be None (all IDLE) or scores near zero
        assert result.zone is None, f"Expected None zone during EMPTY, got {result.zone}"

        # Phase 2: Disturb AB + AC → should go OCCUPIED with zone Q1
        states = _feed_disturbed_frames(
            engine, num_frames=25, disturbed_links=["AB", "AC"]
        )
        assert RoomState.OCCUPIED in states
        assert engine.room_state == RoomState.OCCUPIED

        result = detector.estimate()
        assert result.zone == Zone.Q1, f"Expected Q1, got {result.zone}"
        assert result.confidence > 1.0, "Q1 confidence should be > 1.0"

        # Phase 3: Return to stable → EMPTY, zone None
        _feed_stable_frames(engine, num_frames=30, seq_start=2000)
        assert engine.room_state == RoomState.EMPTY

        result = detector.estimate()
        assert result.zone is None, f"Expected None zone after EMPTY, got {result.zone}"


# ---------------------------------------------------------------------------
# Zone transition: Q1 → Q4
# ---------------------------------------------------------------------------


class TestZoneTransition:
    """Prove zone changes when disturbance pattern shifts."""

    def test_zone_change_q1_to_q4(self):
        """Start with AB+AC disturbed (Q1), switch to CD+BD disturbed (Q4)."""
        engine = PresenceEngine()
        detector = ZoneDetector(engine)

        # Baseline
        _feed_stable_frames(engine, num_frames=25)

        # Phase 1: Q1 disturbance
        _feed_disturbed_frames(
            engine, num_frames=25, disturbed_links=["AB", "AC"]
        )
        assert engine.room_state == RoomState.OCCUPIED
        result1 = detector.estimate()
        assert result1.zone == Zone.Q1, f"Phase 1 expected Q1, got {result1.zone}"

        # Transition: restore AB+AC to stable, disturb CD+BD
        # We need enough frames to flush the old pattern and establish the new one
        _feed_disturbed_frames(
            engine, num_frames=30, disturbed_links=["CD", "BD"], seq_start=3000
        )
        assert engine.room_state == RoomState.OCCUPIED

        result2 = detector.estimate()
        assert result2.zone == Zone.Q4, f"Phase 2 expected Q4, got {result2.zone}"


# ---------------------------------------------------------------------------
# process_line_zone wiring tests
# ---------------------------------------------------------------------------


class TestProcessLineZone:
    """Test the process_line_zone function from main_zone.py."""

    def test_process_line_zone_with_s02_data(self):
        """process_line_zone correctly feeds S02 CSI data through the pipeline."""
        engine = PresenceEngine()
        detector = ZoneDetector(engine)
        raw_bytes = _make_stable_bytes()
        line = _make_s02_line(1, "A", "B", "AB", raw_bytes)

        new_state, new_zone, was_csi = process_line_zone(
            line, engine, detector, RoomState.EMPTY, None,
        )

        assert was_csi is True
        assert new_state == RoomState.EMPTY  # single frame, window not full
        assert new_zone is None

    def test_process_line_zone_ignores_non_csi(self):
        """process_line_zone returns False for non-CSI lines."""
        engine = PresenceEngine()
        detector = ZoneDetector(engine)
        state, zone, was_csi = process_line_zone(
            "boot message", engine, detector, RoomState.EMPTY, None,
        )
        assert was_csi is False
        assert state == RoomState.EMPTY
        assert zone is None

    def test_process_line_zone_ignores_unknown_link(self):
        """process_line_zone handles unknown link IDs gracefully."""
        engine = PresenceEngine()
        detector = ZoneDetector(engine)
        raw_bytes = _make_stable_bytes()
        line = _make_s02_line(1, "X", "Y", "XY", raw_bytes)
        state, zone, was_csi = process_line_zone(
            line, engine, detector, RoomState.EMPTY, None,
        )
        assert was_csi is False

    def test_process_line_zone_full_pipeline(self):
        """process_line_zone drives full pipeline: EMPTY/None → OCCUPIED/Q2."""
        engine = PresenceEngine()
        detector = ZoneDetector(engine)
        prev_state = RoomState.EMPTY
        prev_zone: Zone | None = None

        # Phase 1: Stable baseline
        for seq in range(25):
            raw = _make_stable_bytes()
            for lid in LINK_IDS:
                tx, rx = _TX_RX[lid]
                line = _make_s02_line(seq, tx, rx, lid, raw)
                prev_state, prev_zone, _ = process_line_zone(
                    line, engine, detector, prev_state, prev_zone,
                )
        assert prev_state == RoomState.EMPTY
        assert prev_zone is None

        # Phase 2: Disturb AB + BD → should get OCCUPIED with Q2
        saw_occupied = False
        for seq in range(25):
            for lid in LINK_IDS:
                tx, rx = _TX_RX[lid]
                if lid in ("AB", "BD"):
                    raw = _make_disturbed_bytes(frame_index=seq)
                else:
                    raw = _make_stable_bytes()
                line = _make_s02_line(1000 + seq, tx, rx, lid, raw)
                prev_state, prev_zone, _ = process_line_zone(
                    line, engine, detector, prev_state, prev_zone,
                )
                if prev_state == RoomState.OCCUPIED:
                    saw_occupied = True
        assert saw_occupied, "Never reached OCCUPIED"
        assert prev_zone == Zone.Q2, f"Expected Q2, got {prev_zone}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestZoneEdgeCases:
    """Edge cases: single-link disturbance, all-links disturbance."""

    def test_single_link_disturbance_ab(self):
        """Disturbing only AB → zone should be Q1 or Q2 (tiebreaker: Q1)."""
        engine = PresenceEngine()
        detector = ZoneDetector(engine)

        _feed_stable_frames(engine, num_frames=25)
        states = _feed_disturbed_frames(
            engine, num_frames=25, disturbed_links=["AB"]
        )
        assert RoomState.OCCUPIED in states

        result = detector.estimate()
        # AB has weight 1.0 for Q1 and Q2, 0.0 for Q3/Q4
        # Tied → tiebreaker picks Q1 (lowest ordinal)
        assert result.zone in (Zone.Q1, Zone.Q2), f"Expected Q1 or Q2, got {result.zone}"
        # With only AB disturbed, Q1 and Q2 are tied → Q1 wins tiebreaker
        assert result.zone == Zone.Q1, f"AB-only tiebreaker should be Q1, got {result.zone}"

    def test_all_links_disturbed_equally(self):
        """All 6 links disturbed equally → all zones score similarly → Q1 (tiebreaker)."""
        engine = PresenceEngine()
        detector = ZoneDetector(engine)

        _feed_stable_frames(engine, num_frames=25)
        states = _feed_disturbed_frames(
            engine, num_frames=25, disturbed_links=LINK_IDS,
        )
        assert RoomState.OCCUPIED in states

        result = detector.estimate()
        assert result.zone is not None
        # All links disturbed with same synthetic data → scores should be similar
        # Tiebreaker picks Q1
        assert result.zone == Zone.Q1, (
            f"All-equal tiebreaker should be Q1, got {result.zone}. "
            f"Scores: {result.scores}"
        )


# ---------------------------------------------------------------------------
# format_zone_detail
# ---------------------------------------------------------------------------


class TestFormatZoneDetail:
    """Test the format_zone_detail display function."""

    def test_format_includes_zone_and_scores(self):
        """Detail line includes zone name, confidence, and all 4 zone scores."""
        engine = PresenceEngine()
        detector = ZoneDetector(engine)

        # Establish baseline and disturb AB+AC → Q1
        _feed_stable_frames(engine, num_frames=25)
        _feed_disturbed_frames(
            engine, num_frames=25, disturbed_links=["AB", "AC"]
        )

        detail = format_zone_detail(detector)
        assert "Q1" in detail
        assert "confidence" in detail
        assert "Q2=" in detail
        assert "Q3=" in detail
        assert "Q4=" in detail

    def test_format_empty_room(self):
        """Detail line shows -- for empty room."""
        engine = PresenceEngine()
        detector = ZoneDetector(engine)

        detail = format_zone_detail(detector)
        assert "--" in detail
        assert "no estimate" in detail

    def test_format_after_baseline(self):
        """After stable baseline (all IDLE), zone should be None / no estimate."""
        engine = PresenceEngine()
        detector = ZoneDetector(engine)

        _feed_stable_frames(engine, num_frames=25)
        detail = format_zone_detail(detector)
        assert "--" in detail
