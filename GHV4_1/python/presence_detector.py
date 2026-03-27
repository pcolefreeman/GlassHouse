"""
Presence detection — per-link state machine and multi-link aggregation.

LinkDetector maintains a moving variance of turbulence values for a single
Wi-Fi link and classifies it as IDLE or MOTION.  PresenceEngine aggregates
six LinkDetector instances into a binary OCCUPIED / EMPTY determination.

Architecture:
    CSI frame → select_subcarriers → compute_turbulence → LinkDetector.update()
                                                        → PresenceEngine.update()

Variance computation uses two-pass algorithm (mean first, then sum of squared
deviations) to avoid catastrophic cancellation with small CV values.
"""

from __future__ import annotations

from collections import deque
from enum import Enum

# ---------------------------------------------------------------------------
# Constants & enums
# ---------------------------------------------------------------------------

#: Default number of turbulence samples for the moving variance window.
#: At ~5 Hz per link, 20 samples ≈ 4 seconds of history.
DEFAULT_WINDOW_SIZE: int = 20

#: Default variance threshold for IDLE ↔ MOTION transitions.
#: Tuned for CV-normalized turbulence values (σ/μ).
DEFAULT_THRESHOLD: float = 0.005

#: The six canonical link IDs in an ESP32 mesh (4 nodes: A, B, C, D).
LINK_IDS: list[str] = ["AB", "AC", "AD", "BC", "BD", "CD"]


class LinkState(Enum):
    """Per-link motion state."""
    IDLE = "IDLE"
    MOTION = "MOTION"


class RoomState(Enum):
    """Aggregated room occupancy."""
    EMPTY = "EMPTY"
    OCCUPIED = "OCCUPIED"


# ---------------------------------------------------------------------------
# LinkDetector — per-link state machine
# ---------------------------------------------------------------------------


class LinkDetector:
    """Moving-variance state machine for a single Wi-Fi link.

    Maintains a circular buffer of turbulence values and computes the
    variance over the most recent ``window_size`` samples.  State
    transitions only occur after the window is full — early frames
    produce unreliable variance estimates.

    Args:
        link_id: Canonical link identifier (e.g. "AB").  Stored for
            diagnostics; no re-canonicalization is performed.
        window_size: Number of turbulence samples in the moving window.
        threshold: Variance above which the link is classified as MOTION.
    """

    def __init__(
        self,
        link_id: str,
        window_size: int = DEFAULT_WINDOW_SIZE,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.link_id = link_id
        self.window_size = window_size
        self.threshold = threshold

        self._buffer: deque[float] = deque(maxlen=window_size)
        self._state: LinkState = LinkState.IDLE
        self._variance: float = 0.0

    # -- public interface ---------------------------------------------------

    @property
    def state(self) -> LinkState:
        """Current link state (IDLE or MOTION)."""
        return self._state

    @property
    def variance(self) -> float:
        """Most recently computed moving variance."""
        return self._variance

    @property
    def window_full(self) -> bool:
        """True once the buffer has accumulated ``window_size`` samples."""
        return len(self._buffer) == self.window_size

    def update(self, turbulence: float) -> LinkState:
        """Ingest one turbulence sample and return the updated state.

        The variance is recomputed after every sample, but state transitions
        are suppressed until the window is full.

        Args:
            turbulence: CV-normalized turbulence from
                :func:`csi_features.compute_turbulence`.

        Returns:
            Current :class:`LinkState` after ingesting this sample.
        """
        self._buffer.append(turbulence)
        self._variance = self._compute_variance()

        if self.window_full:
            if self._variance > self.threshold:
                self._state = LinkState.MOTION
            else:
                self._state = LinkState.IDLE

        return self._state

    def get_status(self) -> dict:
        """Diagnostic snapshot of this link's detection state.

        Returns:
            Dict with keys: state (str), variance (float), window_full (bool).
        """
        return {
            "state": self._state.value,
            "variance": self._variance,
            "window_full": self.window_full,
        }

    def reset(self) -> None:
        """Clear the buffer and reset state to IDLE."""
        self._buffer.clear()
        self._state = LinkState.IDLE
        self._variance = 0.0

    # -- internal -----------------------------------------------------------

    def _compute_variance(self) -> float:
        """Two-pass variance of the current buffer contents.

        Pass 1: compute the mean.
        Pass 2: compute the sum of squared deviations from the mean.

        Returns population variance (ddof=0) — the buffer contents are
        the complete window, not a sample from a larger distribution.

        Returns 0.0 if the buffer has fewer than 2 elements.
        """
        n = len(self._buffer)
        if n < 2:
            return 0.0

        # Pass 1: mean
        total = 0.0
        for v in self._buffer:
            total += v
        mean = total / n

        # Pass 2: sum of squared deviations
        sq_sum = 0.0
        for v in self._buffer:
            diff = v - mean
            sq_sum += diff * diff

        return sq_sum / n


# ---------------------------------------------------------------------------
# PresenceEngine — multi-link aggregation
# ---------------------------------------------------------------------------


class PresenceEngine:
    """Aggregates six LinkDetector instances into OCCUPIED / EMPTY.

    Rule: OCCUPIED if **any** link is MOTION; EMPTY if **all** links are IDLE.
    State changes are only meaningful after at least one link's window is full.

    Args:
        window_size: Window size forwarded to each LinkDetector.
        threshold: Threshold forwarded to each LinkDetector.
        link_ids: List of link IDs.  Defaults to :data:`LINK_IDS`.
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        threshold: float = DEFAULT_THRESHOLD,
        link_ids: list[str] | None = None,
    ) -> None:
        if link_ids is None:
            link_ids = list(LINK_IDS)

        self._detectors: dict[str, LinkDetector] = {
            lid: LinkDetector(lid, window_size=window_size, threshold=threshold)
            for lid in link_ids
        }
        self._room_state: RoomState = RoomState.EMPTY

    # -- public interface ---------------------------------------------------

    @property
    def room_state(self) -> RoomState:
        """Current aggregated room state."""
        return self._room_state

    def update(self, link_id: str, turbulence: float) -> RoomState:
        """Ingest one turbulence sample for a specific link.

        Updates the appropriate LinkDetector, recomputes the aggregated
        room state, and returns it.

        Args:
            link_id: Canonical link identifier (e.g. "AB").
            turbulence: CV-normalized turbulence value.

        Returns:
            Current :class:`RoomState` after this update.

        Raises:
            KeyError: If ``link_id`` is not a recognized link.
        """
        detector = self._detectors[link_id]
        detector.update(turbulence)
        self._room_state = self._aggregate()
        return self._room_state

    def get_link_states(self) -> dict[str, dict]:
        """Diagnostic snapshot of all links.

        Returns:
            Dict mapping link_id → {state, variance, window_full}.
        """
        return {lid: det.get_status() for lid, det in self._detectors.items()}

    def get_detector(self, link_id: str) -> LinkDetector:
        """Access a specific LinkDetector (for testing / diagnostics).

        Raises:
            KeyError: If ``link_id`` is not recognized.
        """
        return self._detectors[link_id]

    def reset(self) -> None:
        """Reset all detectors and room state."""
        for det in self._detectors.values():
            det.reset()
        self._room_state = RoomState.EMPTY

    # -- internal -----------------------------------------------------------

    def _aggregate(self) -> RoomState:
        """Any-link-OR: OCCUPIED if any link is MOTION, else EMPTY."""
        for det in self._detectors.values():
            if det.state == LinkState.MOTION:
                return RoomState.OCCUPIED
        return RoomState.EMPTY
