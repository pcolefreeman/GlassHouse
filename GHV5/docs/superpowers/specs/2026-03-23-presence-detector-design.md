# Presence Detector Design — Reworked Amplitude Approach

**Date:** 2026-03-23
**Status:** Draft
**Project:** GHV5 — SAR Breathing Detection

## Problem

The current `_amplitude_score()` method in `BreathingDetector` uses linear detrend + FFT
to detect breathing-band SNR. The detrend step removes the static attenuation signal —
the strongest indicator of human presence (body absorbs/reflects WiFi, reducing mean CSI
amplitude on intersected paths). Hardware testing showed inconsistent results: some paths
gained 7-35x with a person present, while others decreased.

The root cause is that `_amplitude_score()` was designed to detect periodic micro-motion
(breathing), not static presence. It actively discards the signal we need for presence
detection.

## Solution

Replace `_amplitude_score()` with a new `_presence_score()` that detects static
attenuation using two zero-calibration signals. Keep `_pca_score()` unchanged for
breathing detection. This creates a two-stage pipeline:

- **Presence score** (new): "Is someone on this path?" — via amplitude attenuation
- **Breathing score** (existing PCA): "Are they breathing?" — via periodic micro-motion

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Calibration | Zero-calibration (no empty-room baseline) | Operational simplicity |
| Output format | Two separate scores per cell (presence + breathing) | Maps to existing A:/P: display slots |
| Presence signals | Cross-path ranking + per-path variance | Ranking is primary (large effect), variance is fallback |
| Breathing method | Keep `_pca_score()` as-is | Detrend is correct for periodic motion isolation |
| Dead code | Remove `_amplitude_score()`, `CSIRatioExtractor`, `BreathingAnalyzer` | Replaced or unused; recoverable from git |

## Architecture

```
CSI snap frames --> Ring Buffers (unchanged)
                        |
                 +------+------+
                 |             |
          Presence Score   Breathing Score
          (NEW: static     (EXISTING: PCA
           attenuation)     _pca_score())
                 |             |
                 +------+------+
                        |
                GridProjector (unchanged)
                --> {cell: {presence: 0-100, breathing: 0-100}}
```

### Unchanged components
- `CSIRingBuffer` — same buffering
- `_pca_score()` — kept as-is for breathing detection
- `GridProjector` — same path-to-cell projection
- `BreathingThread`, `SARDemoThread` — minimal key renames
- `run_sar.py` — label renames only

## Presence Score Algorithm

### Signal 1: Cross-path amplitude ranking

For each path's window of CSI data:

1. Compute mean amplitude per subcarrier across all time steps.
2. Take the median across valid subcarriers to get a single scalar `path_mean`.
3. Collect `path_mean` values across all ready paths.
4. Rank score per path: `(group_median - path_mean) / group_median`, clamped to [0, 1].
5. Paths with lower-than-group amplitude score higher (body absorbing signal).
6. Paths at or above group median score 0.
7. Requires 3+ active paths. If fewer, returns 0 for all paths.

**Why it works:** A person blocks specific paths. Those paths lose amplitude relative to
unblocked paths. The unblocked paths serve as the baseline — no calibration needed.

### Signal 2: Per-path amplitude variance

For each path's window:

1. Compute amplitude per subcarrier per time step: `(n_time, n_valid_subs)` matrix.
2. Compute variance over time for each subcarrier.
3. Take 75th percentile of variances across subcarriers.
4. Map through log-sigmoid: low variance (static room) maps to ~5%, high variance
   (involuntary motion) maps to ~95%.

**Why 75th percentile:** Only some subcarriers respond to motion. Median reflects the
non-responsive majority. 75th percentile captures responsive subcarriers without being
as sensitive to outliers as 95th.

### Combining signals

```
presence = max(rank_score, variance_score)
```

Simple max fusion. If either signal fires, presence is detected. Rank score is the
primary signal (large effect size from static attenuation). Variance is the fallback for
multi-person scenarios where most paths are attenuated (breaking the cross-path
comparison assumption).

## Interface Changes

### `_presence_score` signature

```python
@staticmethod
def _presence_score(window: np.ndarray,
                    all_path_means: dict[tuple, float] | None = None) -> float:
```

Needs cross-path context via `all_path_means`. When `None` or fewer than 3 paths,
falls back to variance-only.

### `get_all_scores()` two-pass approach

1. **Pass 1:** Compute `path_mean` for every ready buffer (mean amplitude, cheap).
2. **Pass 2:** Call `_presence_score(window, all_path_means)` per path with group context.

### Return value change

```python
def get_all_scores(self) -> dict:
    return {
        "presence":  self._projector.project(presence_confidences),
        "pca":       self._projector.project(pca_confidences),
        "path_conf": presence_confidences,
    }
```

Key `"amp"` becomes `"presence"`.

### `get_grid_scores()` change

Calls `_presence_score()` instead of `_amplitude_score()`.

### Display changes

- Dict keys: `"amp_grid"` becomes `"presence_grid"` in thread result dicts.
- `BreathingDisplay`: field/param `amp_grid` becomes `presence_grid`.
- Cell labels: `"A:"` becomes `"Pr:"` (presence), `"P:"` stays (PCA/breathing).
- Console: `"Path confidence (amp)"` becomes `"Path confidence (presence)"`.

### New config constants

```python
PRESENCE_VARIANCE_MIDPOINT = 0.01    # sigmoid center (tune with hardware)
PRESENCE_VARIANCE_STEEPNESS = 3.0    # sigmoid steepness
```

Existing `BREATHING_CONFIDENCE_THRESHOLD` still applies to the presence score.

### Removals

- `_amplitude_score()` static method
- `CSIRatioExtractor` class
- `BreathingAnalyzer` class
- Related imports

### Debug logging

`_presence_score` logs at DEBUG level: `path_mean`, `group_median`, `rank_score`,
`path_var`, and final `presence`. Same pattern as the old `snr_p95` logging, enabling
hardware tuning via `--log-level DEBUG`.

## Test Plan

### New tests

1. **Cross-path ranking — one attenuated path:** 4 paths, one at 50% lower mean
   amplitude. Verify high presence score on attenuated path, ~0 on others.
2. **Cross-path ranking — all paths equal:** Same mean amplitude on all paths.
   All presence scores ~0.
3. **Cross-path ranking — fewer than 3 paths:** Falls back to variance-only.
4. **Variance — static room:** Constant amplitude across time. Near-zero score.
5. **Variance — involuntary motion:** Small random amplitude fluctuations.
   Elevated score.
6. **Max fusion:** Verify `presence = max(rank, variance)`.
7. **Integration:** `get_all_scores()` returns `{"presence": ..., "pca": ...}` keys.

### Tests to remove

- Tests exercising `CSIRatioExtractor`, `BreathingAnalyzer`, or `_amplitude_score`.

## Tuning

The variance sigmoid midpoint (`PRESENCE_VARIANCE_MIDPOINT`) needs tuning with real
hardware data. Run with `--log-level DEBUG` to see raw `path_var` values for empty room
vs. person present, then set the midpoint between the two distributions.

The cross-path ranking signal requires no tuning — it's a relative comparison.
