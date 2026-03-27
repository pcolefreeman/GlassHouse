"""Unit tests for presence detection — LinkDetector and PresenceEngine.

All tests use synthetic turbulence values — no hardware or serial port required.
"""

from __future__ import annotations

import math
import os
import sys

# Ensure the python/ directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from presence_detector import (
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW_SIZE,
    LINK_IDS,
    LinkDetector,
    LinkState,
    PresenceEngine,
    RoomState,
)


# ---------------------------------------------------------------------------
# LinkDetector tests
# ---------------------------------------------------------------------------


class TestLinkDetectorInitialization:
    """Tests for LinkDetector initial state."""

    def test_initial_state_idle(self):
        """New detector starts in IDLE state."""
        det = LinkDetector("AB")
        assert det.state == LinkState.IDLE

    def test_initial_variance_zero(self):
        """New detector has zero variance."""
        det = LinkDetector("AB")
        assert det.variance == 0.0

    def test_initial_window_not_full(self):
        """Window is not full until enough samples are ingested."""
        det = LinkDetector("AB")
        assert det.window_full is False

    def test_stores_link_id(self):
        """Link ID is stored for diagnostics."""
        det = LinkDetector("CD")
        assert det.link_id == "CD"

    def test_custom_window_size(self):
        """Window size is configurable."""
        det = LinkDetector("AB", window_size=10)
        assert det.window_size == 10

    def test_custom_threshold(self):
        """Threshold is configurable."""
        det = LinkDetector("AB", threshold=0.01)
        assert det.threshold == 0.01

    def test_defaults_match_constants(self):
        """Default values match module-level constants."""
        det = LinkDetector("AB")
        assert det.window_size == DEFAULT_WINDOW_SIZE
        assert det.threshold == DEFAULT_THRESHOLD


class TestLinkDetectorWindowFilling:
    """Tests for buffer filling behavior and variance suppression."""

    def test_window_fills_at_exact_size(self):
        """window_full becomes True after exactly window_size samples."""
        det = LinkDetector("AB", window_size=5)
        for i in range(4):
            det.update(0.001)
            assert det.window_full is False, f"Should not be full at {i + 1} samples"
        det.update(0.001)
        assert det.window_full is True

    def test_state_stays_idle_while_filling(self):
        """State must remain IDLE while the window is not yet full,
        even if turbulence values would trigger MOTION."""
        det = LinkDetector("AB", window_size=5, threshold=0.001)
        # Feed high-variance values, but window isn't full yet
        for val in [0.0, 0.5, 0.0, 0.5]:
            state = det.update(val)
            assert state == LinkState.IDLE, (
                "Must not transition while window is filling"
            )

    def test_variance_computed_during_filling(self):
        """Variance is computed even while filling, but no state change."""
        det = LinkDetector("AB", window_size=5)
        det.update(0.0)
        det.update(1.0)
        # Variance of [0.0, 1.0] = 0.25 (population)
        assert det.variance > 0.0, "Variance should be computed during filling"
        assert det.window_full is False

    def test_single_sample_zero_variance(self):
        """A single sample produces zero variance (need >= 2 for variation)."""
        det = LinkDetector("AB", window_size=5)
        det.update(0.5)
        assert det.variance == 0.0


class TestLinkDetectorStateTransitions:
    """Tests for IDLE → MOTION → IDLE state transitions."""

    def _fill_with_constant(self, det: LinkDetector, value: float) -> None:
        """Fill the detector's window with a constant turbulence value."""
        for _ in range(det.window_size):
            det.update(value)

    def test_idle_with_low_variance(self):
        """Constant low-turbulence values → IDLE state."""
        det = LinkDetector("AB", window_size=5, threshold=0.005)
        self._fill_with_constant(det, 0.01)
        assert det.state == LinkState.IDLE
        assert det.variance < det.threshold

    def test_motion_with_high_variance(self):
        """Alternating turbulence values → high variance → MOTION."""
        det = LinkDetector("AB", window_size=4, threshold=0.001)
        # Alternate between 0.0 and 0.1 → variance = 0.0025
        for val in [0.0, 0.1, 0.0, 0.1]:
            det.update(val)
        assert det.window_full is True
        assert det.variance > det.threshold
        assert det.state == LinkState.MOTION

    def test_idle_to_motion_transition(self):
        """Stable period followed by disturbance → IDLE → MOTION."""
        det = LinkDetector("AB", window_size=4, threshold=0.001)
        # Fill with stable values
        self._fill_with_constant(det, 0.01)
        assert det.state == LinkState.IDLE

        # Now push high-variance values to fill the window
        for val in [0.0, 0.2, 0.0, 0.2]:
            det.update(val)
        assert det.state == LinkState.MOTION

    def test_motion_to_idle_transition(self):
        """After MOTION, feeding stable values returns to IDLE."""
        det = LinkDetector("AB", window_size=4, threshold=0.001)
        # First push high-variance to reach MOTION
        for val in [0.0, 0.2, 0.0, 0.2]:
            det.update(val)
        assert det.state == LinkState.MOTION

        # Now push constant values to push out the variance
        self._fill_with_constant(det, 0.01)
        assert det.state == LinkState.IDLE

    def test_update_returns_current_state(self):
        """update() return value matches .state property."""
        det = LinkDetector("AB", window_size=3, threshold=0.001)
        for _ in range(3):
            result = det.update(0.01)
        assert result == det.state

    def test_threshold_boundary_below(self):
        """Variance exactly at threshold → should be IDLE (not strictly greater)."""
        det = LinkDetector("AB", window_size=2, threshold=0.25)
        # Values [0, 1]: variance = ((0-0.5)^2 + (1-0.5)^2) / 2 = 0.25
        det.update(0.0)
        det.update(1.0)
        # variance == threshold → not > threshold → IDLE
        assert det.state == LinkState.IDLE

    def test_threshold_boundary_above(self):
        """Variance just above threshold → MOTION."""
        det = LinkDetector("AB", window_size=2, threshold=0.24)
        # Values [0, 1]: variance = 0.25 > 0.24
        det.update(0.0)
        det.update(1.0)
        assert det.state == LinkState.MOTION


class TestLinkDetectorVarianceComputation:
    """Tests for the two-pass variance algorithm."""

    def test_known_variance(self):
        """Verify variance matches manual computation.

        Values: [1, 2, 3, 4, 5]
        Mean = 3.0
        Variance = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5 = 2.0
        """
        det = LinkDetector("AB", window_size=5)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            det.update(v)
        assert math.isclose(det.variance, 2.0, rel_tol=1e-10)

    def test_uniform_values_zero_variance(self):
        """All identical values → variance = 0."""
        det = LinkDetector("AB", window_size=5)
        for _ in range(5):
            det.update(0.042)
        assert det.variance == 0.0

    def test_variance_is_population(self):
        """Variance uses ddof=0 (population variance, not sample)."""
        det = LinkDetector("AB", window_size=2)
        # Values [0, 2]: mean=1, pop_var = ((0-1)^2 + (2-1)^2) / 2 = 1.0
        # sample_var would be = 2.0
        det.update(0.0)
        det.update(2.0)
        assert math.isclose(det.variance, 1.0, rel_tol=1e-10)

    def test_sliding_window_eviction(self):
        """New values evict old ones from the circular buffer."""
        det = LinkDetector("AB", window_size=3)
        # Fill: [10, 10, 10] → var=0
        for _ in range(3):
            det.update(10.0)
        assert det.variance == 0.0

        # Push one different value: [10, 10, 100] → var > 0
        det.update(100.0)
        assert det.variance > 0.0

        # Push two more: [100, 100, 100] → var=0 again
        det.update(100.0)
        det.update(100.0)
        assert det.variance == 0.0


class TestLinkDetectorGetStatus:
    """Tests for the diagnostic get_status() method."""

    def test_status_contains_required_keys(self):
        """Status dict must have state, variance, and window_full."""
        det = LinkDetector("AB")
        status = det.get_status()
        assert "state" in status
        assert "variance" in status
        assert "window_full" in status

    def test_status_state_is_string(self):
        """State value should be the enum's string value."""
        det = LinkDetector("AB")
        assert det.get_status()["state"] == "IDLE"

    def test_status_reflects_updates(self):
        """Status should reflect the current state after updates."""
        det = LinkDetector("AB", window_size=2, threshold=0.001)
        det.update(0.0)
        det.update(1.0)
        status = det.get_status()
        assert status["window_full"] is True
        assert status["variance"] > 0
        assert status["state"] == "MOTION"


class TestLinkDetectorReset:
    """Tests for the reset() method."""

    def test_reset_clears_state(self):
        """After reset, detector is back to initial state."""
        det = LinkDetector("AB", window_size=3, threshold=0.001)
        for val in [0.0, 1.0, 0.0]:
            det.update(val)
        assert det.state == LinkState.MOTION

        det.reset()
        assert det.state == LinkState.IDLE
        assert det.variance == 0.0
        assert det.window_full is False


# ---------------------------------------------------------------------------
# PresenceEngine tests
# ---------------------------------------------------------------------------


class TestPresenceEngineInitialization:
    """Tests for PresenceEngine initial state."""

    def test_initial_room_state_empty(self):
        """Engine starts in EMPTY state."""
        engine = PresenceEngine()
        assert engine.room_state == RoomState.EMPTY

    def test_default_link_ids(self):
        """Default engine has all 6 canonical links."""
        engine = PresenceEngine()
        states = engine.get_link_states()
        assert set(states.keys()) == set(LINK_IDS)

    def test_custom_link_ids(self):
        """Engine can be configured with custom link IDs."""
        engine = PresenceEngine(link_ids=["XY", "XZ"])
        states = engine.get_link_states()
        assert set(states.keys()) == {"XY", "XZ"}

    def test_all_links_start_idle(self):
        """Every link detector starts in IDLE state."""
        engine = PresenceEngine()
        for lid, status in engine.get_link_states().items():
            assert status["state"] == "IDLE", f"Link {lid} should start IDLE"


class TestPresenceEngineUpdate:
    """Tests for PresenceEngine update and aggregation."""

    def _fill_link_constant(
        self,
        engine: PresenceEngine,
        link_id: str,
        value: float,
        count: int | None = None,
    ) -> None:
        """Feed a constant turbulence value to one link."""
        if count is None:
            count = engine.get_detector(link_id).window_size
        for _ in range(count):
            engine.update(link_id, value)

    def test_all_idle_is_empty(self):
        """When all links are IDLE, room is EMPTY."""
        engine = PresenceEngine(window_size=5, threshold=0.005)
        for lid in LINK_IDS:
            self._fill_link_constant(engine, lid, 0.01)
        assert engine.room_state == RoomState.EMPTY

    def test_one_link_motion_is_occupied(self):
        """If any single link transitions to MOTION, room becomes OCCUPIED."""
        engine = PresenceEngine(window_size=4, threshold=0.001)
        # Fill all links with stable values
        for lid in LINK_IDS:
            self._fill_link_constant(engine, lid, 0.01)
        assert engine.room_state == RoomState.EMPTY

        # Disturb just AB
        for val in [0.0, 0.2, 0.0, 0.2]:
            engine.update("AB", val)
        assert engine.room_state == RoomState.OCCUPIED

    def test_multiple_links_motion_still_occupied(self):
        """Multiple links in MOTION → still OCCUPIED (OR logic)."""
        engine = PresenceEngine(window_size=4, threshold=0.001)
        # Disturb AB and CD
        for val in [0.0, 0.2, 0.0, 0.2]:
            engine.update("AB", val)
            engine.update("CD", val)
        # Fill remaining links with stable
        for lid in ["AC", "AD", "BC", "BD"]:
            self._fill_link_constant(engine, lid, 0.01)
        assert engine.room_state == RoomState.OCCUPIED

    def test_occupied_to_empty_when_all_calm(self):
        """OCCUPIED → EMPTY when all disturbed links return to IDLE."""
        engine = PresenceEngine(window_size=4, threshold=0.001)
        # Fill all stable first
        for lid in LINK_IDS:
            self._fill_link_constant(engine, lid, 0.01)

        # Disturb AB
        for val in [0.0, 0.2, 0.0, 0.2]:
            engine.update("AB", val)
        assert engine.room_state == RoomState.OCCUPIED

        # Calm AB down
        self._fill_link_constant(engine, "AB", 0.01)
        assert engine.room_state == RoomState.EMPTY

    def test_unknown_link_raises_keyerror(self):
        """Updating a non-existent link ID raises KeyError."""
        engine = PresenceEngine()
        try:
            engine.update("ZZ", 0.01)
            assert False, "Should have raised KeyError"
        except KeyError:
            pass

    def test_update_returns_room_state(self):
        """update() return value is the current RoomState."""
        engine = PresenceEngine(window_size=3)
        result = engine.update("AB", 0.01)
        assert isinstance(result, RoomState)

    def test_window_not_full_stays_empty(self):
        """Engine stays EMPTY while link windows are filling,
        even with high-variance values."""
        engine = PresenceEngine(window_size=5, threshold=0.001)
        # Feed 4 samples (window not full) with alternating values
        for val in [0.0, 0.5, 0.0, 0.5]:
            engine.update("AB", val)
        assert engine.room_state == RoomState.EMPTY


class TestPresenceEngineGetLinkStates:
    """Tests for diagnostic get_link_states() method."""

    def test_returns_all_links(self):
        """get_link_states() returns status for every configured link."""
        engine = PresenceEngine()
        states = engine.get_link_states()
        assert len(states) == 6
        assert set(states.keys()) == set(LINK_IDS)

    def test_each_entry_has_required_keys(self):
        """Each link status entry has state, variance, window_full."""
        engine = PresenceEngine()
        for lid, status in engine.get_link_states().items():
            assert "state" in status, f"Missing 'state' for {lid}"
            assert "variance" in status, f"Missing 'variance' for {lid}"
            assert "window_full" in status, f"Missing 'window_full' for {lid}"

    def test_reflects_individual_link_updates(self):
        """Link states reflect updates to specific links."""
        engine = PresenceEngine(window_size=2, threshold=0.001)
        # Only update AB with high-variance values
        engine.update("AB", 0.0)
        engine.update("AB", 1.0)
        states = engine.get_link_states()
        assert states["AB"]["state"] == "MOTION"
        assert states["AB"]["window_full"] is True
        # CD should still be initial
        assert states["CD"]["state"] == "IDLE"
        assert states["CD"]["window_full"] is False


class TestPresenceEngineReset:
    """Tests for engine-wide reset."""

    def test_reset_all_links(self):
        """reset() returns all links to IDLE and room to EMPTY."""
        engine = PresenceEngine(window_size=2, threshold=0.001)
        engine.update("AB", 0.0)
        engine.update("AB", 1.0)
        assert engine.room_state == RoomState.OCCUPIED

        engine.reset()
        assert engine.room_state == RoomState.EMPTY
        for lid, status in engine.get_link_states().items():
            assert status["state"] == "IDLE", f"{lid} should be IDLE after reset"
            assert status["window_full"] is False


class TestPresenceEngineGetDetector:
    """Tests for direct detector access."""

    def test_get_known_detector(self):
        """get_detector returns the LinkDetector for a valid link."""
        engine = PresenceEngine()
        det = engine.get_detector("AB")
        assert isinstance(det, LinkDetector)
        assert det.link_id == "AB"

    def test_get_unknown_detector_raises(self):
        """get_detector raises KeyError for unknown link IDs."""
        engine = PresenceEngine()
        try:
            engine.get_detector("ZZ")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass


# ---------------------------------------------------------------------------
# Full scenario: EMPTY → OCCUPIED → EMPTY lifecycle
# ---------------------------------------------------------------------------


class TestOccupancyLifecycle:
    """Integration-style tests covering the complete detection lifecycle."""

    def test_empty_occupied_empty_cycle(self):
        """Simulate a person entering and leaving a room.

        Phase 1: 20 stable frames on all links → EMPTY
        Phase 2: Disturbed frames on AB and CD → OCCUPIED
        Phase 3: Stable frames again → EMPTY
        """
        engine = PresenceEngine(window_size=5, threshold=0.001)

        # Phase 1: all links stable → EMPTY
        for lid in LINK_IDS:
            for _ in range(5):
                engine.update(lid, 0.01)
        assert engine.room_state == RoomState.EMPTY

        # Phase 2: disturb AB and CD with alternating values
        for _ in range(5):
            engine.update("AB", 0.0)
            engine.update("CD", 0.0)
        for _ in range(5):
            engine.update("AB", 0.2)
            engine.update("CD", 0.2)
        # At this point AB and CD have high variance windows
        # (the alternating 0.0/0.2 values evicted the stable ones)
        # Need to ensure we push enough alternating to fill window
        # Let's do a full window of alternation
        for i in range(5):
            val = 0.0 if i % 2 == 0 else 0.2
            engine.update("AB", val)
            engine.update("CD", val)

        assert engine.room_state == RoomState.OCCUPIED

        # Phase 3: all links calm down → EMPTY
        for lid in LINK_IDS:
            for _ in range(5):
                engine.update(lid, 0.01)
        assert engine.room_state == RoomState.EMPTY

    def test_single_link_drives_occupancy(self):
        """Even one link in MOTION is enough for OCCUPIED."""
        engine = PresenceEngine(window_size=3, threshold=0.001)

        # Fill all links stable
        for lid in LINK_IDS:
            for _ in range(3):
                engine.update(lid, 0.01)
        assert engine.room_state == RoomState.EMPTY

        # Disturb only BD
        for val in [0.0, 0.3, 0.0]:
            engine.update("BD", val)
        assert engine.room_state == RoomState.OCCUPIED

        # Calm BD
        for _ in range(3):
            engine.update("BD", 0.01)
        assert engine.room_state == RoomState.EMPTY
