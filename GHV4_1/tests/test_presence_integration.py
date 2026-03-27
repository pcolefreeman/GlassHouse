"""
Integration tests for the CSI presence detection pipeline.

Feeds synthetic S02-format CSV lines through the full pipeline:
    parse_csi_line → compute_amplitudes → select_subcarriers →
    compute_turbulence → PresenceEngine.update()

Proves EMPTY → OCCUPIED → EMPTY state transitions without hardware.
"""

from __future__ import annotations

import math
import os
import sys

# Ensure the python/ directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from serial_csi_reader import parse_csi_line, compute_amplitudes
from csi_features import select_subcarriers, compute_turbulence, SELECTED_SUBCARRIERS
from presence_detector import (
    PresenceEngine,
    RoomState,
    LinkState,
    LINK_IDS,
    DEFAULT_WINDOW_SIZE,
)
from main_presence import process_line, format_link_detail


# ---------------------------------------------------------------------------
# Helpers — synthetic CSI data generation
# ---------------------------------------------------------------------------

# We need at least max(SELECTED_SUBCARRIERS)+1 = 53 subcarriers.
# Each subcarrier = 2 bytes (imag, real).  So we need 106 bytes minimum.
_NUM_SUBCARRIERS = 64
_NUM_BYTES = _NUM_SUBCARRIERS * 2  # 128 bytes


def _make_stable_bytes(base_amplitude: float = 10.0) -> list[int]:
    """Generate raw CSI bytes where all subcarriers have nearly identical amplitude.

    Stable amplitudes → low turbulence (CV ≈ 0) → IDLE.
    All subcarriers get the same (imag=0, real=base_amplitude) pair,
    producing amplitude = base_amplitude for every subcarrier.
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
    Each subcarrier gets a different amplitude based on its index and the
    frame_index.  The frame-dependent scaling factor ensures that the
    *turbulence value itself varies between frames* — which is what the
    detector's moving variance measures.  Without frame-to-frame variation
    in turbulence, the variance stays below threshold even though
    individual frames have high CV.
    """
    bytes_out: list[int] = []
    # Frame-dependent scaling creates turbulence variation across frames
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


# ---------------------------------------------------------------------------
# Verify stable data actually produces low turbulence
# ---------------------------------------------------------------------------


class TestSyntheticDataSanity:
    """Verify our synthetic data generators produce the expected signal properties."""

    def test_stable_bytes_produce_low_turbulence(self):
        """Stable bytes should yield CV ≈ 0 (all amplitudes identical)."""
        raw = _make_stable_bytes(10.0)
        amps = compute_amplitudes(raw)
        selected = select_subcarriers(amps)
        turb = compute_turbulence(selected)
        # All selected amplitudes should be 10.0 → std=0 → CV=0
        assert turb < 0.001, f"Stable data turbulence too high: {turb}"

    def test_disturbed_bytes_produce_high_turbulence(self):
        """Disturbed bytes should yield CV >> 0 (varied amplitudes)."""
        raw = _make_disturbed_bytes(frame_index=5)
        amps = compute_amplitudes(raw)
        selected = select_subcarriers(amps)
        turb = compute_turbulence(selected)
        # High variation across subcarriers → high CV
        assert turb > 0.05, f"Disturbed data turbulence too low: {turb}"

    def test_disturbed_frames_have_varying_turbulence(self):
        """Different frame indices should produce different turbulence values."""
        turbulences = []
        for frame_idx in range(10):
            raw = _make_disturbed_bytes(frame_index=frame_idx)
            amps = compute_amplitudes(raw)
            selected = select_subcarriers(amps)
            turb = compute_turbulence(selected)
            turbulences.append(turb)
        # Not all identical (frames vary)
        assert len(set(round(t, 6) for t in turbulences)) > 1


# ---------------------------------------------------------------------------
# Full pipeline integration: EMPTY → OCCUPIED → EMPTY
# ---------------------------------------------------------------------------


class TestPresenceLifecycle:
    """Prove the full EMPTY → OCCUPIED → EMPTY detection lifecycle."""

    def _feed_stable_frames(
        self,
        engine: PresenceEngine,
        num_frames: int,
        link_ids: list[str] | None = None,
    ) -> list[RoomState]:
        """Feed stable (low turbulence) frames through the full pipeline."""
        if link_ids is None:
            link_ids = LINK_IDS
        states: list[RoomState] = []
        seq = 0
        tx_rx = {"AB": ("A", "B"), "AC": ("A", "C"), "AD": ("A", "D"),
                 "BC": ("B", "C"), "BD": ("B", "D"), "CD": ("C", "D")}
        for _ in range(num_frames):
            raw_bytes = _make_stable_bytes()
            for lid in link_ids:
                tx, rx = tx_rx[lid]
                line = _make_s02_line(seq, tx, rx, lid, raw_bytes)
                parsed = parse_csi_line(line)
                assert parsed is not None
                assert parsed["link_id"] == lid
                amps = compute_amplitudes(parsed["raw_bytes"])
                selected = select_subcarriers(amps)
                turb = compute_turbulence(selected)
                state = engine.update(lid, turb)
                seq += 1
            states.append(state)
        return states

    def _feed_disturbed_frames(
        self,
        engine: PresenceEngine,
        num_frames: int,
        disturbed_links: list[str],
        all_links: list[str] | None = None,
    ) -> list[RoomState]:
        """Feed frames where some links are disturbed, others stable."""
        if all_links is None:
            all_links = LINK_IDS
        states: list[RoomState] = []
        seq = 1000
        tx_rx = {"AB": ("A", "B"), "AC": ("A", "C"), "AD": ("A", "D"),
                 "BC": ("B", "C"), "BD": ("B", "D"), "CD": ("C", "D")}
        for frame_idx in range(num_frames):
            for lid in all_links:
                tx, rx = tx_rx[lid]
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

    def test_initial_state_is_empty(self):
        """Engine starts in EMPTY state before any data."""
        engine = PresenceEngine()
        assert engine.room_state == RoomState.EMPTY

    def test_stable_data_stays_empty(self):
        """25 frames of stable data on all 6 links → remains EMPTY."""
        engine = PresenceEngine()
        states = self._feed_stable_frames(engine, num_frames=25)
        # After window fills (20 frames), all frames should report EMPTY
        for state in states[DEFAULT_WINDOW_SIZE:]:
            assert state == RoomState.EMPTY

    def test_disturbed_data_transitions_to_occupied(self):
        """After stable baseline, disturbed data on 2 links → OCCUPIED."""
        engine = PresenceEngine()
        # Phase 1: establish stable baseline
        self._feed_stable_frames(engine, num_frames=25)
        assert engine.room_state == RoomState.EMPTY

        # Phase 2: disturb 2 links
        states = self._feed_disturbed_frames(
            engine, num_frames=25, disturbed_links=["AB", "CD"]
        )
        # Should transition to OCCUPIED once the disturbed variance
        # exceeds the threshold after window fills with new data
        assert RoomState.OCCUPIED in states, (
            f"Never transitioned to OCCUPIED. States: {[s.value for s in states]}"
        )

    def test_full_empty_occupied_empty_lifecycle(self):
        """Complete EMPTY → OCCUPIED → EMPTY lifecycle."""
        engine = PresenceEngine()

        # Phase 1: Stable (EMPTY)
        states_1 = self._feed_stable_frames(engine, num_frames=25)
        assert engine.room_state == RoomState.EMPTY
        # After window fills, should be EMPTY
        for s in states_1[DEFAULT_WINDOW_SIZE:]:
            assert s == RoomState.EMPTY

        # Phase 2: Disturbed on AB, BC, CD (OCCUPIED)
        states_2 = self._feed_disturbed_frames(
            engine, num_frames=30, disturbed_links=["AB", "BC", "CD"]
        )
        assert RoomState.OCCUPIED in states_2, "Failed to transition to OCCUPIED"
        # Last state should be OCCUPIED (disturbance ongoing)
        assert states_2[-1] == RoomState.OCCUPIED

        # Phase 3: Return to stable (EMPTY)
        states_3 = self._feed_stable_frames(engine, num_frames=30)
        assert RoomState.EMPTY in states_3, "Failed to transition back to EMPTY"
        # After enough stable frames to flush the window, should be EMPTY
        assert states_3[-1] == RoomState.EMPTY

    def test_single_link_disturbance_triggers_occupied(self):
        """Even one disturbed link should trigger OCCUPIED (any-OR rule)."""
        engine = PresenceEngine()
        self._feed_stable_frames(engine, num_frames=25)
        assert engine.room_state == RoomState.EMPTY

        states = self._feed_disturbed_frames(
            engine, num_frames=25, disturbed_links=["AD"]
        )
        assert RoomState.OCCUPIED in states, "Single link disturbance should trigger OCCUPIED"

    def test_link_states_reflect_disturbance_pattern(self):
        """After disturbance, link states should show MOTION only on disturbed links."""
        engine = PresenceEngine()
        self._feed_stable_frames(engine, num_frames=25)

        # Disturb only AB and CD
        self._feed_disturbed_frames(
            engine, num_frames=25, disturbed_links=["AB", "CD"]
        )

        link_states = engine.get_link_states()
        # Disturbed links should be MOTION
        assert link_states["AB"]["state"] == "MOTION"
        assert link_states["CD"]["state"] == "MOTION"
        # Undisturbed links should be IDLE
        assert link_states["AC"]["state"] == "IDLE"
        assert link_states["AD"]["state"] == "IDLE"
        assert link_states["BC"]["state"] == "IDLE"
        assert link_states["BD"]["state"] == "IDLE"


# ---------------------------------------------------------------------------
# process_line integration (main_presence.py wiring)
# ---------------------------------------------------------------------------


class TestProcessLine:
    """Test the process_line function from main_presence.py."""

    def test_process_line_with_s02_data(self):
        """process_line correctly feeds S02 CSI data through the pipeline."""
        engine = PresenceEngine()
        raw_bytes = _make_stable_bytes()
        line = _make_s02_line(1, "A", "B", "AB", raw_bytes)

        new_state, was_csi = process_line(line, engine, RoomState.EMPTY)

        assert was_csi is True
        assert new_state == RoomState.EMPTY  # single frame, window not full

    def test_process_line_ignores_non_csi(self):
        """process_line returns False for non-CSI lines."""
        engine = PresenceEngine()
        state, was_csi = process_line("boot message", engine, RoomState.EMPTY)
        assert was_csi is False
        assert state == RoomState.EMPTY

    def test_process_line_ignores_s01_format(self):
        """process_line skips S01 lines (no link_id)."""
        engine = PresenceEngine()
        line = "CSI_DATA,42,24:6F:28:AA:BB:CC,-55,8,3 -4 10 20 -1 0 7 -8"
        state, was_csi = process_line(line, engine, RoomState.EMPTY)
        assert was_csi is False

    def test_process_line_ignores_unknown_link(self):
        """process_line handles unknown link IDs gracefully."""
        engine = PresenceEngine()
        raw_bytes = _make_stable_bytes()
        line = _make_s02_line(1, "X", "Y", "XY", raw_bytes)
        state, was_csi = process_line(line, engine, RoomState.EMPTY)
        assert was_csi is False

    def test_process_line_full_lifecycle(self):
        """process_line drives EMPTY → OCCUPIED → EMPTY through CSV lines."""
        engine = PresenceEngine()
        prev_state = RoomState.EMPTY
        tx_rx = {"AB": ("A", "B"), "AC": ("A", "C"), "AD": ("A", "D"),
                 "BC": ("B", "C"), "BD": ("B", "D"), "CD": ("C", "D")}

        # Phase 1: Stable frames
        for seq in range(25):
            raw = _make_stable_bytes()
            for lid in LINK_IDS:
                tx, rx = tx_rx[lid]
                line = _make_s02_line(seq, tx, rx, lid, raw)
                prev_state, _ = process_line(line, engine, prev_state)
        assert prev_state == RoomState.EMPTY

        # Phase 2: Disturbed frames on AB, CD
        saw_occupied = False
        for seq in range(25):
            for lid in LINK_IDS:
                tx, rx = tx_rx[lid]
                if lid in ("AB", "CD"):
                    raw = _make_disturbed_bytes(frame_index=seq)
                else:
                    raw = _make_stable_bytes()
                line = _make_s02_line(1000 + seq, tx, rx, lid, raw)
                prev_state, _ = process_line(line, engine, prev_state)
                if prev_state == RoomState.OCCUPIED:
                    saw_occupied = True
        assert saw_occupied, "Never reached OCCUPIED"

        # Phase 3: Back to stable
        for seq in range(30):
            raw = _make_stable_bytes()
            for lid in LINK_IDS:
                tx, rx = tx_rx[lid]
                line = _make_s02_line(2000 + seq, tx, rx, lid, raw)
                prev_state, _ = process_line(line, engine, prev_state)
        assert prev_state == RoomState.EMPTY


# ---------------------------------------------------------------------------
# format_link_detail
# ---------------------------------------------------------------------------


class TestFormatLinkDetail:
    """Test the format_link_detail display function."""

    def test_format_includes_all_links(self):
        """Detail line includes all 6 canonical links."""
        engine = PresenceEngine()
        detail = format_link_detail(engine)
        for lid in LINK_IDS:
            assert lid in detail

    def test_format_shows_state_and_variance(self):
        """Detail line shows state label and variance for each link."""
        engine = PresenceEngine()
        detail = format_link_detail(engine)
        assert "IDLE" in detail
        assert "0.0000" in detail
