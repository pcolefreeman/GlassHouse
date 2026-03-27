"""
Zone detection — maps per-link CSI disturbance patterns to room quadrants.

ZoneDetector consumes get_link_states() from PresenceEngine and scores
each of 4 zones (quadrants) using a static geometric weight matrix.
Zone scores are computed from continuous variance values, not binary
MOTION/IDLE, giving better spatial discrimination.

Room layout assumption (looking down):
    A (top-left)  ----  B (top-right)
    |        \\ /        |
    |         X         |
    |        / \\        |
    C (bot-left)  ----  D (bot-right)

Quadrants:
    Q1 = top-left  (near A)
    Q2 = top-right (near B)
    Q3 = bot-left  (near C)
    Q4 = bot-right (near D)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from presence_detector import PresenceEngine


# ---------------------------------------------------------------------------
# Enums & data structures
# ---------------------------------------------------------------------------

class Zone(Enum):
    """Room quadrant identifiers, ordered for tiebreaking (lowest wins)."""
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"


@dataclass
class ZoneResult:
    """Result of a zone estimation.

    Attributes:
        zone: Winning zone, or None if room is EMPTY / no windows full.
        scores: Per-zone weighted variance score.
        confidence: Ratio of top score to second-best score.
            0.0 when there are no meaningful scores.
    """
    zone: Zone | None
    scores: dict[Zone, float] = field(default_factory=dict)
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Weight matrix
# ---------------------------------------------------------------------------

#: Geometric weights mapping each link to its influence on each zone.
#:
#: Edge links get 1.0 for their 2 adjacent zones and 0.0 for the others.
#: Diagonal links get 0.5 for their primary 2 zones (endpoints) and 0.3
#: for the secondary 2 zones (partial coverage from crossing the room).
LINK_ZONE_WEIGHTS: dict[str, dict[Zone, float]] = {
    # AB — top edge, adjacent to Q1 (top-left) and Q2 (top-right)
    "AB": {Zone.Q1: 1.0, Zone.Q2: 1.0, Zone.Q3: 0.0, Zone.Q4: 0.0},
    # CD — bottom edge, adjacent to Q3 (bot-left) and Q4 (bot-right)
    "CD": {Zone.Q1: 0.0, Zone.Q2: 0.0, Zone.Q3: 1.0, Zone.Q4: 1.0},
    # AC — left edge, adjacent to Q1 (top-left) and Q3 (bot-left)
    "AC": {Zone.Q1: 1.0, Zone.Q2: 0.0, Zone.Q3: 1.0, Zone.Q4: 0.0},
    # BD — right edge, adjacent to Q2 (top-right) and Q4 (bot-right)
    "BD": {Zone.Q1: 0.0, Zone.Q2: 1.0, Zone.Q3: 0.0, Zone.Q4: 1.0},
    # AD — diagonal from A (top-left) to D (bot-right)
    "AD": {Zone.Q1: 0.5, Zone.Q2: 0.3, Zone.Q3: 0.3, Zone.Q4: 0.5},
    # BC — diagonal from B (top-right) to C (bot-left)
    "BC": {Zone.Q1: 0.3, Zone.Q2: 0.5, Zone.Q3: 0.5, Zone.Q4: 0.3},
}


# ---------------------------------------------------------------------------
# ZoneDetector
# ---------------------------------------------------------------------------

class ZoneDetector:
    """Estimates which room quadrant a person is in.

    Uses per-link variance from PresenceEngine.get_link_states() and the
    static LINK_ZONE_WEIGHTS matrix to compute a weighted variance score
    for each zone.  The zone with the highest score wins; ties are broken
    by lowest Zone enum ordinal (Q1 < Q2 < Q3 < Q4).

    Args:
        engine: PresenceEngine instance providing get_link_states().
    """

    def __init__(self, engine: PresenceEngine) -> None:
        self._engine = engine

    def estimate(self) -> ZoneResult:
        """Compute the most likely zone from current link states.

        Only links with ``window_full=True`` contribute to scoring.
        If no link has a full window, or all contributing links are
        IDLE (variance at or below threshold), returns zone=None.

        Returns:
            ZoneResult with the winning zone, all scores, and confidence.
        """
        link_states = self._engine.get_link_states()

        # Accumulate zone scores from links with full windows
        scores: dict[Zone, float] = {z: 0.0 for z in Zone}
        any_contributing = False

        for link_id, state_info in link_states.items():
            if not state_info["window_full"]:
                continue
            if link_id not in LINK_ZONE_WEIGHTS:
                continue

            variance = state_info["variance"]
            weights = LINK_ZONE_WEIGHTS[link_id]

            for zone, weight in weights.items():
                scores[zone] += weight * variance

            any_contributing = True

        # Room is effectively EMPTY if nothing contributed or all scores zero
        if not any_contributing:
            return ZoneResult(zone=None, scores=scores, confidence=0.0)

        max_score = max(scores.values())
        if max_score == 0.0:
            return ZoneResult(zone=None, scores=scores, confidence=0.0)

        # Pick the winning zone — ties broken by enum order (Q1 first).
        # Round scores to 12 decimal places to absorb floating-point
        # accumulation noise before comparing, so near-identical scores
        # reliably fall through to the enum-order tiebreaker.
        zones_by_priority = list(Zone)  # Q1, Q2, Q3, Q4
        winning_zone = max(
            zones_by_priority,
            key=lambda z: (round(scores[z], 12), -zones_by_priority.index(z)),
        )

        # Confidence = ratio of best to second-best
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[1] > 0.0:
            confidence = sorted_scores[0] / sorted_scores[1]
        else:
            # Only one zone has a nonzero score (or only one zone exists)
            confidence = float("inf") if max_score > 0.0 else 0.0

        return ZoneResult(
            zone=winning_zone,
            scores=scores,
            confidence=confidence,
        )

    def get_zone_scores(self) -> dict[str, float]:
        """Diagnostic: return raw zone scores keyed by zone name.

        Returns:
            Dict like {"Q1": 0.042, "Q2": 0.018, ...}.
        """
        result = self.estimate()
        return {z.value: score for z, score in result.scores.items()}
