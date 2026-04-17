"""
Zone detection — maps motion CSI evidence to room quadrants.

ZoneDetector consumes motion diagnostics from a callable that returns
a link states dict (decoupled from PresenceEngine).

Room layout assumption (looking down):
    2 (top-left)  ----  3 (top-right)
    |        \\ /        |
    |         X         |
    |        / \\        |
    1 (bot-left)  ----  4 (bot-right)

Quadrants:
    Q1 = top-left  (near 2)
    Q2 = top-right (near 3)
    Q3 = bot-left  (near 1)
    Q4 = bot-right (near 4)
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


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

#: Spike weights — derived from capture data via `debug/capture.py --weights`.
#: Q4 vs Q3 discrimination is weak — link 14 (the only bottom-edge spike link)
#: gives Q3 weight 1.0 vs Q4 0.77, and absorption on link 23 adds only 0.23
#: to Q4.  This is a hardware layout limitation, not a tuning bug.
LINK_ZONE_WEIGHTS: dict[str, dict[Zone, float]] = {
    "13": {Zone.Q1: 1.0, Zone.Q2: 0.0, Zone.Q3: 0.19, Zone.Q4: 0.0},
    "14": {Zone.Q1: 0.0, Zone.Q2: 0.0, Zone.Q3: 1.0,  Zone.Q4: 0.77},
    "24": {Zone.Q1: 1.0, Zone.Q2: 0.0, Zone.Q3: 0.0,  Zone.Q4: 0.0},
}

#: Absorption weights — for high-baseline links that DROP when person blocks LOS.
LINK_ABSORPTION_WEIGHTS: dict[str, dict[Zone, float]] = {
    "23": {Zone.Q1: 0.0, Zone.Q2: 1.0, Zone.Q3: 0.0, Zone.Q4: 0.23},
    "34": {Zone.Q1: 0.0, Zone.Q2: 1.0, Zone.Q3: 0.0, Zone.Q4: 0.0},
}


# ---------------------------------------------------------------------------
# ZoneDetector
# ---------------------------------------------------------------------------

class ZoneDetector:
    """Estimates which room quadrant a person is in.

    Uses rolling-baseline change detection: each link maintains an EMA
    baseline that adapts continuously.  When a link's variance jumps
    above its baseline by more than a deviation threshold, the link is
    considered active.  No empty-room calibration needed — works from
    the first frame, even with people already present.

    Args:
        link_states_fn: Callable returning a link states dict.
    """

    _EMA_ALPHA = 0.03            # Baseline adapts with ~33-frame half-life
    _DEVIATION_MULT = 3.0        # Active when variance > baseline * mult
    _MIN_CONTRIBUTING_LINKS = 2  # Require 2+ unique links in recent window
    _RECENT_WINDOW = 8           # Frames to look back for contributing links
    _BASELINE_FLOOR = 0.1        # Don't let baselines drop below this
    _WARMUP_FRAMES = 10          # Suppress detection during baseline settling
    _ABSORPTION_MULT = 0.3       # Active when variance < baseline * 0.3
    _ABSORPTION_FLOOR = 10.0     # Only check absorption on high-baseline links

    def __init__(self, link_states_fn: Callable[[], dict[str, dict]]) -> None:
        self._link_states_fn = link_states_fn
        self._baselines: dict[str, float] = {}
        self._last_result: ZoneResult | None = None
        self._recent_active: deque[dict[str, float]] = deque(maxlen=self._RECENT_WINDOW)
        self._frame_count: int = 0

    @staticmethod
    def _empty_scores() -> dict[Zone, float]:
        """Return a zero-initialized score map for all zones."""
        return {z: 0.0 for z in Zone}

    def _get_link_states_snapshot(self) -> dict[str, dict]:
        """Read motion-link diagnostics from the callable."""
        return self._link_states_fn()

    def _update_baselines(self, link_states: dict[str, dict]) -> None:
        """Update rolling EMA baselines from link variances.

        Skips zero/near-zero variance frames — these indicate missing
        data (link not observed), not actual zero activity.  Prevents
        baselines from being dragged down by intermittent delivery.
        """
        for link_id, state_info in link_states.items():
            if not state_info.get("window_full", False):
                continue
            variance = state_info.get("variance", 0.0)
            if variance < self._BASELINE_FLOOR:
                continue  # skip: no real data this frame
            if link_id in self._baselines:
                self._baselines[link_id] += self._EMA_ALPHA * (
                    variance - self._baselines[link_id]
                )
            else:
                self._baselines[link_id] = variance

    @staticmethod
    def _finalize_result(scores: dict[Zone, float], any_contributing: bool) -> ZoneResult:
        """Convert accumulated scores into a ZoneResult."""
        if not any_contributing:
            return ZoneResult(zone=None, scores=scores, confidence=0.0)

        max_score = max(scores.values())
        if max_score == 0.0:
            return ZoneResult(zone=None, scores=scores, confidence=0.0)

        zones_by_priority = list(Zone)  # Q1, Q2, Q3, Q4
        winning_zone = max(
            zones_by_priority,
            key=lambda z: (round(scores[z], 12), -zones_by_priority.index(z)),
        )

        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[1] > 0.0:
            confidence = sorted_scores[0] / sorted_scores[1]
        else:
            confidence = float("inf") if max_score > 0.0 else 0.0

        return ZoneResult(
            zone=winning_zone,
            scores=scores,
            confidence=confidence,
        )

    def _score_motion_links(self) -> tuple[dict[Zone, float], bool]:
        """Score zones using bidirectional change detection.

        Spike detection: link active when variance exceeds baseline * mult.
        Absorption detection: link active when variance drops well below
        baseline on high-baseline links (person blocking LOS).
        A sliding window merges asynchronous events across frames.
        """
        link_states = self._get_link_states_snapshot()
        self._update_baselines(link_states)
        self._frame_count += 1
        scores = self._empty_scores()
        any_contributing = False
        frame_active: dict[str, float] = {}

        if self._frame_count <= self._WARMUP_FRAMES:
            self._recent_active.append(frame_active)
            return scores, any_contributing

        # Spike detection (variance increase)
        for link_id, state_info in link_states.items():
            if not state_info["window_full"]:
                continue
            if link_id not in LINK_ZONE_WEIGHTS:
                continue
            if link_id not in self._baselines:
                continue

            variance = state_info["variance"]
            if variance < self._BASELINE_FLOOR:
                continue
            baseline = self._baselines[link_id]
            if variance > baseline * self._DEVIATION_MULT:
                frame_active[link_id] = (variance - baseline) / max(baseline, 1.0)

        # Absorption detection (variance decrease on high-baseline links)
        # Only fires when window_full=True AND variance is well below baseline.
        # The "not window_full" (stale/dark link) path is intentionally omitted:
        # it cannot distinguish person-blocking from sparse packet delivery,
        # causing false positives in empty rooms with intermittent link data.
        for link_id in LINK_ABSORPTION_WEIGHTS:
            if link_id not in self._baselines:
                continue
            baseline = self._baselines[link_id]
            if baseline < self._ABSORPTION_FLOOR:
                continue
            state_info = link_states.get(link_id)
            if state_info is None:
                continue
            if not state_info.get("window_full", False):
                continue
            variance = state_info.get("variance", 0.0)
            if variance < baseline * self._ABSORPTION_MULT:
                frame_active[f"abs:{link_id}"] = min(
                    (baseline - variance) / max(baseline, 1.0), 1.0
                )

        self._recent_active.append(frame_active)

        # Merge active links from recent window — keep best excess per link
        merged: dict[str, float] = {}
        for past in self._recent_active:
            for lid, excess in past.items():
                if lid not in merged or excess > merged[lid]:
                    merged[lid] = excess

        # Count unique contributing links (spike or absorption)
        contributing_links: set[str] = set()
        for lid in merged:
            raw_id = lid.removeprefix("abs:")
            contributing_links.add(raw_id)

        if len(contributing_links) >= self._MIN_CONTRIBUTING_LINKS:
            for lid, excess in merged.items():
                if lid.startswith("abs:"):
                    raw_id = lid[4:]
                    weights = LINK_ABSORPTION_WEIGHTS[raw_id]
                else:
                    weights = LINK_ZONE_WEIGHTS[lid]
                for zone, weight in weights.items():
                    scores[zone] += weight * excess
            any_contributing = True

        return scores, any_contributing

    def estimate(self) -> ZoneResult:
        """Compute the most likely zone from motion link evidence."""
        motion_scores, motion_contributing = self._score_motion_links()
        if motion_contributing:
            result = self._finalize_result(motion_scores, any_contributing=True)
        else:
            result = self._finalize_result(motion_scores, any_contributing=False)
        self._last_result = result
        return result

    def get_zone_scores(self) -> dict[str, float]:
        """Diagnostic: return raw zone scores keyed by zone name."""
        if self._last_result is None:
            self.estimate()
        return {z.value: score for z, score in self._last_result.scores.items()}


class StableZoneTracker:
    """Hold a zone until a challenger wins consistently or decisively."""

    def __init__(
        self,
        switch_streak: int = 2,
        confidence_override: float = 1.35,
        switch_ratio: float = 1.15,
    ) -> None:
        self._switch_streak = switch_streak
        self._confidence_override = confidence_override
        self._switch_ratio = switch_ratio
        self._held_zone: Zone | None = None
        self._candidate_zone: Zone | None = None
        self._candidate_count: int = 0

    def reset(self) -> None:
        """Forget the currently held and pending zones."""
        self._held_zone = None
        self._candidate_zone = None
        self._candidate_count = 0

    def update(self, result: ZoneResult, occupied: bool) -> ZoneResult:
        """Return a stabilized zone result suitable for UI/console display."""
        if not occupied:
            self.reset()
            return ZoneResult(zone=None, scores=result.scores, confidence=result.confidence)

        candidate = result.zone
        if candidate is None:
            return ZoneResult(
                zone=self._held_zone,
                scores=result.scores,
                confidence=result.confidence,
            )

        if self._held_zone is None:
            self._held_zone = candidate
            self._candidate_zone = None
            self._candidate_count = 0
            return result

        if candidate == self._held_zone:
            self._candidate_zone = None
            self._candidate_count = 0
            return result

        held_score = result.scores.get(self._held_zone, 0.0)
        candidate_score = result.scores.get(candidate, 0.0)
        decisive_switch = (
            candidate_score > 0.0
            and (held_score <= 0.0 or (candidate_score / held_score) >= self._switch_ratio)
            and result.confidence >= self._confidence_override
        )
        if decisive_switch:
            self._held_zone = candidate
            self._candidate_zone = None
            self._candidate_count = 0
            return result

        if candidate != self._candidate_zone:
            self._candidate_zone = candidate
            self._candidate_count = 1
        else:
            self._candidate_count += 1

        if self._candidate_count >= self._switch_streak:
            self._held_zone = candidate
            self._candidate_zone = None
            self._candidate_count = 0
            return result

        return ZoneResult(
            zone=self._held_zone,
            scores=result.scores,
            confidence=result.confidence,
        )


def build_live_zone_tracker() -> StableZoneTracker:
    """Return a tracker tuned for variance-ratio scoring."""
    return StableZoneTracker(
        switch_streak=2,
        confidence_override=1.0,
        switch_ratio=1.15,
    )
