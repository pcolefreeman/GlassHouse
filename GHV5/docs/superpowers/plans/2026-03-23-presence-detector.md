# Presence Detector Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `_amplitude_score()` with a two-signal zero-calibration presence detector (cross-path amplitude ranking + per-path variance), keep `_pca_score()` for breathing, and remove dead code (`CSIRatioExtractor`, `BreathingAnalyzer`).

**Architecture:** New `_presence_score(window, all_path_means)` static method on `BreathingDetector` uses cross-path amplitude ranking as the primary signal and per-path amplitude variance as a fallback. `get_all_scores()` does a two-pass computation: Pass 1 collects per-path mean amplitudes for the group ranking, Pass 2 scores each path with that group context. Output keys rename `"amp"` → `"presence"` and `"amp_grid"` → `"presence_grid"` throughout.

**Tech Stack:** Python 3.12+, NumPy, pytest. No new dependencies.

---

## File Map

| File | Action | What Changes |
|------|--------|-------------|
| `ghv5/config.py` | Modify | Add `PRESENCE_VARIANCE_MIDPOINT`, `PRESENCE_VARIANCE_STEEPNESS` |
| `ghv5/breathing.py` | Modify | Add `_presence_score()`, update `get_all_scores()` + `get_grid_scores()`, remove `_amplitude_score()`, `CSIRatioExtractor`, `BreathingAnalyzer`; rename dict keys in threads + display |
| `tests/test_breathing.py` | Modify | Remove `TestCSIRatioExtractor`, `TestBreathingAnalyzer`; add `TestPresenceScore`; update `TestGetAllScores` key refs; update `TestDemoThread` key refs; update imports |
| `run_sar.py` | Modify | Rename `amp_scores` → `presence_scores`, update console labels |

---

## Task 1: Add Config Constants

**Files:**
- Modify: `ghv5/config.py`

- [ ] **Step 1: Add the two new constants** after the existing `BREATHING_PCA_COMPONENTS` line:

```python
PRESENCE_VARIANCE_MIDPOINT  = 50.0   # sigmoid center (tune with --log-level DEBUG)
PRESENCE_VARIANCE_STEEPNESS = 0.5    # sigmoid steepness
```

> Note: `PRESENCE_VARIANCE_MIDPOINT` is in raw amplitude-variance units (not log scale). The log-sigmoid operates on `log(path_var)`, so the midpoint is `log(50.0) ≈ 3.9`. Initial value chosen to sit between expected empty-room and person-present variance; tune after first hardware run with `--log-level DEBUG`.

- [ ] **Step 2: Verify the constants are importable**

Run: `python -c "from ghv5.config import PRESENCE_VARIANCE_MIDPOINT, PRESENCE_VARIANCE_STEEPNESS; print(PRESENCE_VARIANCE_MIDPOINT, PRESENCE_VARIANCE_STEEPNESS)"`

Expected output: `50.0 0.5`

> **Note:** User handles all git operations — skip commit steps.

---

## Task 2: Write Failing Tests for `_presence_score`

**Files:**
- Modify: `tests/test_breathing.py`

Add the new `TestPresenceScore` class **before** `TestBreathingDetector`. Each test verifies one clearly defined behavior.

- [ ] **Step 1: Add the new test class**

Insert after the `TestGridProjector` class (before `TestBreathingDetector`, around line 283):

```python
class TestPresenceScore:
    """Tests for BreathingDetector._presence_score()."""

    @staticmethod
    def _make_complex_window(n_time=600, n_subs=128, mean_amp=1000.0, noise_std=0.0, rng_seed=0):
        """Build a proper complex64 window."""
        rng = np.random.default_rng(rng_seed)
        amp = np.full((n_time, n_subs), mean_amp, dtype=np.float64)
        if noise_std > 0:
            amp += rng.normal(0, noise_std, (n_time, n_subs))
        amp = np.clip(amp, 0, None).astype(np.float32)
        return (amp + 0j).astype(np.complex64)

    def test_returns_float_in_range(self):
        window = self._make_complex_window()
        score = BreathingDetector._presence_score(window)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_one_attenuated_path_gets_high_rank_score(self):
        """Path with 50% lower mean amplitude than group should score high."""
        # 4 paths: 3 at 1000, one (this path) at 500
        low_window  = self._make_complex_window(mean_amp=500.0)
        high_mean = 1000.0
        all_path_means = {
            (1, 2): high_mean,
            (1, 3): high_mean,
            (2, 3): high_mean,
            (1, 4): 500.0,  # this path
        }
        score = BreathingDetector._presence_score(low_window, all_path_means)
        assert score > 0.3, f"Attenuated path should have presence > 0.3, got {score:.3f}"

    def test_all_paths_equal_rank_score_near_zero(self):
        """If all paths have the same mean amplitude, rank score = 0."""
        window = self._make_complex_window(mean_amp=1000.0)
        all_path_means = {
            (1, 2): 1000.0,
            (1, 3): 1000.0,
            (2, 3): 1000.0,
        }
        score = BreathingDetector._presence_score(window, all_path_means)
        # No attenuation on any path — rank score = 0
        # Variance = 0 (no noise) — variance score near 0
        assert score < 0.2, f"Equal paths should have presence < 0.2, got {score:.3f}"

    def test_fewer_than_3_paths_falls_back_to_variance(self):
        """With < 3 paths, rank score is 0 and only variance signal fires."""
        # Static window (no noise) → variance ≈ 0 → presence ≈ 0
        static_window = self._make_complex_window(mean_amp=1000.0, noise_std=0.0)
        all_path_means = {(1, 2): 1000.0, (1, 3): 1000.0}  # only 2 paths
        score_static = BreathingDetector._presence_score(static_window, all_path_means)
        assert score_static < 0.2, f"Static 2-path should be near 0, got {score_static:.3f}"

        # Noisy window (high variance) → variance score fires even without rank
        noisy_window = self._make_complex_window(mean_amp=1000.0, noise_std=500.0, rng_seed=1)
        score_noisy = BreathingDetector._presence_score(noisy_window, all_path_means)
        assert score_noisy > score_static, "Noisy window should score higher than static"

    def test_static_room_low_variance_score(self):
        """Constant amplitude (no motion) should yield near-zero presence score."""
        window = self._make_complex_window(mean_amp=1000.0, noise_std=0.0)
        score = BreathingDetector._presence_score(window)  # no path means → variance only
        assert score < 0.2, f"Static room should score < 0.2, got {score:.3f}"

    def test_high_variance_yields_nonzero_score(self):
        """Strong amplitude fluctuations should yield elevated presence score."""
        # Amplitude oscillates significantly (std = 500 on mean of 1000 = 50% variation)
        window = self._make_complex_window(mean_amp=1000.0, noise_std=500.0, rng_seed=42)
        score = BreathingDetector._presence_score(window)
        assert score > 0.1, f"High-variance window should score > 0.1, got {score:.3f}"

    def test_max_fusion_rank_fires_variance_does_not(self):
        """Presence = max(rank, variance): if only rank fires, result equals rank."""
        # Low-noise window → variance_score ≈ 0
        low_noise_window = self._make_complex_window(mean_amp=500.0, noise_std=1.0, rng_seed=7)
        all_path_means = {
            (1, 2): 1000.0,
            (1, 3): 1000.0,
            (2, 3): 1000.0,
            (1, 4): 500.0,
        }
        score = BreathingDetector._presence_score(low_noise_window, all_path_means)
        # Rank score dominates: group_median ≈ 1000, this_mean ≈ 500 → rank ≈ 0.5
        assert score > 0.3, f"Rank signal should dominate, got {score:.3f}"

    def test_get_all_scores_returns_presence_key(self):
        """get_all_scores() must return 'presence' key, not 'amp'."""
        from ghv5.config import BREATHING_WINDOW_N, BREATHING_PATH_MAP
        import struct as _struct
        det = BreathingDetector()
        csi_bytes = b''.join(_struct.pack('<hh', 1000, 0) for _ in range(128))
        for _ in range(BREATHING_WINDOW_N):
            for key in BREATHING_PATH_MAP:
                det.feed_frame('csi_snap', {
                    'reporter_id': key[0], 'peer_id': key[1], 'csi': csi_bytes
                })
        result = det.get_all_scores()
        assert "presence" in result, f"Expected 'presence' key, got keys: {list(result.keys())}"
        assert "amp" not in result, "'amp' key must be removed"
        assert "pca" in result
        assert "path_conf" in result
```

- [ ] **Step 2: Run the new tests to confirm they fail**

Run: `cd C:\GlassHouse\GHV5 && python -m pytest tests/test_breathing.py::TestPresenceScore -v 2>&1 | head -40`

Expected: Multiple failures — `_presence_score` does not exist yet, `AttributeError`.

---

## Task 3: Implement `_presence_score`

**Files:**
- Modify: `ghv5/breathing.py`

- [ ] **Step 1: Add the import for new config constants** at the top of `breathing.py` (add to the existing `from ghv5.config import (` block, around line 11):

```python
    PRESENCE_VARIANCE_MIDPOINT,
    PRESENCE_VARIANCE_STEEPNESS,
```

The full import block should now include these two new names alongside the existing ones.

- [ ] **Step 2: Add `_presence_score` as a static method on `BreathingDetector`**

Insert the method **after** `_pca_score` (around line 373, before `get_all_scores`):

```python
    @staticmethod
    def _presence_score(window: np.ndarray,
                        all_path_means: dict | None = None) -> float:
        """Zero-calibration presence score: cross-path ranking + amplitude variance.

        Signal 1 — cross-path ranking (requires 3+ paths):
          Compares this path's mean amplitude against the group median.
          A body attenuates specific paths; those paths score higher.
          rank_score = (group_median - this_mean) / group_median, clamped [0, 1].

        Signal 2 — per-path amplitude variance (always available):
          A person (even stationary) causes involuntary motion that increases
          amplitude variance vs an empty, static environment.
          Mapped through a log-sigmoid; midpoint tuned via PRESENCE_VARIANCE_MIDPOINT.

        Final score: max(rank_score, variance_score).

        Args:
            window: (n_time, n_subcarriers) complex64 CSI array.
            all_path_means: {(s1, s2): mean_amp} for all ready paths (for ranking).
                            If None or fewer than 3 entries, rank signal is skipped.

        Returns:
            Presence confidence 0.0–1.0.
        """
        n_time, n_subs = window.shape
        valid = sorted(set(range(n_subs)) - set(NULL_SUBCARRIER_INDICES))
        amp = np.abs(window[:, valid]).astype(np.float64)  # (n_time, n_valid)

        # ── Signal 1: cross-path amplitude ranking ──────────────────────────
        rank_score = 0.0
        if all_path_means is not None and len(all_path_means) >= 3:
            this_mean = float(np.median(np.mean(amp, axis=0)))
            group_median = float(np.median(list(all_path_means.values())))
            if group_median > 1e-9:
                rank_score = float(np.clip(
                    (group_median - this_mean) / group_median, 0.0, 1.0
                ))
            _log.debug("  path_mean=%.3f group_median=%.3f rank_score=%.3f",
                       this_mean, group_median, rank_score)

        # ── Signal 2: per-path amplitude variance ───────────────────────────
        var_per_sub = np.var(amp, axis=0)                   # variance over time
        path_var = float(np.percentile(var_per_sub, 75))    # 75th pct across subcarriers
        _log.debug("  path_var=%.6f", path_var)

        log_var = np.log(max(path_var, 1e-12))
        log_mid = np.log(max(PRESENCE_VARIANCE_MIDPOINT, 1e-12))
        variance_score = float(1.0 / (1.0 + np.exp(
            -PRESENCE_VARIANCE_STEEPNESS * (log_var - log_mid)
        )))
        _log.debug("  variance_score=%.3f", variance_score)

        presence = float(max(rank_score, variance_score))
        _log.debug("  presence=%.3f", presence)
        return presence
```

- [ ] **Step 3: Run the new tests**

Run: `cd C:\GlassHouse\GHV5 && python -m pytest tests/test_breathing.py::TestPresenceScore -v`

Expected: All 8 tests pass. If any fail, check the sigmoid midpoint constant — you may need to adjust `PRESENCE_VARIANCE_MIDPOINT` so that `test_static_room_low_variance_score` and `test_high_variance_yields_nonzero_score` both pass.

> **Tuning note:** If `test_static_room_low_variance_score` fails (score too high), increase `PRESENCE_VARIANCE_MIDPOINT`. If `test_high_variance_yields_nonzero_score` fails (score too low), decrease it.

- [ ] **Step 4: Run the full test suite to confirm nothing broke**

Run: `cd C:\GlassHouse\GHV5 && python -m pytest tests/ -v 2>&1 | tail -20`

Expected: All previously passing tests still pass (new tests added, none broken). Some tests will still reference `_amplitude_score` via `TestGetAllScores` — those will pass for now because we haven't changed `get_all_scores()` yet.

---

## Task 4: Update `get_all_scores()` and `get_grid_scores()`

**Files:**
- Modify: `ghv5/breathing.py`

Replace both methods with two-pass implementations that use `_presence_score`.

- [ ] **Step 1: Replace `get_grid_scores()`**

Find the existing `get_grid_scores` method (around line 241) and replace it with:

```python
    def get_grid_scores(self) -> dict[str, float | None]:
        """Run presence analysis on all ready paths and project onto grid.

        Uses a two-pass approach: Pass 1 collects per-path mean amplitudes
        for cross-path ranking context; Pass 2 scores each path.
        """
        # Pass 1: collect mean amplitudes for cross-path ranking
        all_path_means: dict[tuple, float] = {}
        ready_windows: dict[tuple, np.ndarray] = {}
        for key, buf in self._buffers.items():
            if not buf.is_full():
                continue
            window = buf.get_window()
            ready_windows[key] = window
            valid = sorted(set(range(window.shape[1])) - set(NULL_SUBCARRIER_INDICES))
            amp = np.abs(window[:, valid]).astype(np.float64)
            all_path_means[key] = float(np.median(np.mean(amp, axis=0)))

        # Pass 2: score each path
        presence_confidences: dict[tuple, float] = {}
        for key, window in ready_windows.items():
            confidence = self._presence_score(window, all_path_means)
            _log.info("Path S%d↔S%d presence=%.3f", key[0], key[1], confidence)
            presence_confidences[key] = confidence

        return self._projector.project(presence_confidences)
```

- [ ] **Step 2: Replace `get_all_scores()`**

Find the existing `get_all_scores` method (around line 375) and replace it with:

```python
    def get_all_scores(self, k: int = BREATHING_PCA_COMPONENTS) -> dict:
        """Run presence and PCA scoring on all ready path buffers.

        Returns:
            {
              "presence":  {cell: score_0_to_100_or_None},
              "pca":       {cell: score_0_to_100_or_None},
              "path_conf": {(s1, s2): presence_confidence_0_to_1},
            }
        """
        # Pass 1: collect mean amplitudes for cross-path ranking
        all_path_means: dict[tuple, float] = {}
        ready_windows: dict[tuple, np.ndarray] = {}
        for key, buf in self._buffers.items():
            if not buf.is_full():
                continue
            window = buf.get_window()
            ready_windows[key] = window
            valid = sorted(set(range(window.shape[1])) - set(NULL_SUBCARRIER_INDICES))
            amp = np.abs(window[:, valid]).astype(np.float64)
            all_path_means[key] = float(np.median(np.mean(amp, axis=0)))

        # Pass 2: score each path
        presence_confidences: dict[tuple, float] = {}
        pca_confidences: dict[tuple, float] = {}
        for key, window in ready_windows.items():
            p_conf = self._presence_score(window, all_path_means)
            pca_conf = self._pca_score(window, k=k)
            presence_confidences[key] = p_conf
            pca_confidences[key] = pca_conf
            _log.info("Path S%d↔S%d presence=%.3f pca=%.3f",
                      key[0], key[1], p_conf, pca_conf)

        return {
            "presence":  self._projector.project(presence_confidences),
            "pca":       self._projector.project(pca_confidences),
            "path_conf": presence_confidences,
        }
```

- [ ] **Step 3: Run the test suite**

Run: `cd C:\GlassHouse\GHV5 && python -m pytest tests/ -v 2>&1 | tail -30`

Expected failures (tests that reference the old `"amp"` key — will be fixed in Task 5):
- `TestGetAllScores::test_returns_correct_keys` — still checks `{"amp", "pca", "path_conf"}`
- `TestGetAllScores::test_amp_and_pca_grids_contain_all_cells` — references `result["amp"]`
- `TestGetAllScores::test_amp_matches_get_grid_scores` — references `result["amp"]`
- `TestGetAllScores::test_empty_detector_returns_empty_grids` — references `result["amp"]`

All other tests should still pass.

---

## Task 5: Update `TestGetAllScores` Tests

**Files:**
- Modify: `tests/test_breathing.py`

Update the four failing tests in `TestGetAllScores` to use the `"presence"` key.

- [ ] **Step 1: Update `test_returns_correct_keys`**

Find (around line 377):
```python
    def test_returns_correct_keys(self):
        det = self._fill_detector()
        result = det.get_all_scores()
        assert set(result.keys()) == {"amp", "pca", "path_conf"}
```

Replace with:
```python
    def test_returns_correct_keys(self):
        det = self._fill_detector()
        result = det.get_all_scores()
        assert set(result.keys()) == {"presence", "pca", "path_conf"}
```

- [ ] **Step 2: Update `test_amp_and_pca_grids_contain_all_cells`**

Find (around line 382):
```python
    def test_amp_and_pca_grids_contain_all_cells(self):
        det = self._fill_detector()
        result = det.get_all_scores()
        for cell in CELL_LABELS:
            assert cell in result["amp"]
            assert cell in result["pca"]
```

Replace with:
```python
    def test_presence_and_pca_grids_contain_all_cells(self):
        det = self._fill_detector()
        result = det.get_all_scores()
        for cell in CELL_LABELS:
            assert cell in result["presence"]
            assert cell in result["pca"]
```

- [ ] **Step 3: Update `test_amp_matches_get_grid_scores`**

Find (around line 389):
```python
    def test_amp_matches_get_grid_scores(self):
        det = self._fill_detector()
        all_scores = det.get_all_scores()
        grid_scores = det.get_grid_scores()
        for cell, val in grid_scores.items():
            if val is None:
                assert all_scores["amp"][cell] is None
            else:
                assert all_scores["amp"][cell] == pytest.approx(val, abs=0.01)
```

Replace with:
```python
    def test_presence_matches_get_grid_scores(self):
        det = self._fill_detector()
        all_scores = det.get_all_scores()
        grid_scores = det.get_grid_scores()
        for cell, val in grid_scores.items():
            if val is None:
                assert all_scores["presence"][cell] is None
            else:
                assert all_scores["presence"][cell] == pytest.approx(val, abs=0.01)
```

- [ ] **Step 4: Update `test_empty_detector_returns_empty_grids`**

Find (around line 413):
```python
    def test_empty_detector_returns_empty_grids(self):
        det = BreathingDetector()
        result = det.get_all_scores()
        for cell in CELL_LABELS:
            assert result["amp"][cell] is None
            assert result["pca"][cell] is None
        assert result["path_conf"] == {}
```

Replace with:
```python
    def test_empty_detector_returns_empty_grids(self):
        det = BreathingDetector()
        result = det.get_all_scores()
        for cell in CELL_LABELS:
            assert result["presence"][cell] is None
            assert result["pca"][cell] is None
        assert result["path_conf"] == {}
```

- [ ] **Step 5: Run the test suite**

Run: `cd C:\GlassHouse\GHV5 && python -m pytest tests/test_breathing.py::TestGetAllScores -v`

Expected: All tests in `TestGetAllScores` pass.

---

## Task 6: Remove Dead Code from `breathing.py`

**Files:**
- Modify: `ghv5/breathing.py`

Remove `_amplitude_score()`, `CSIRatioExtractor`, and `BreathingAnalyzer`.

- [ ] **Step 1: Remove `_amplitude_score` static method**

Find the `@staticmethod` block for `_amplitude_score` (around line 258) — it starts with:
```python
    @staticmethod
    def _amplitude_score(window: np.ndarray) -> float:
```
Delete the entire method through its closing `return float(score)` line (approximately 57 lines).

- [ ] **Step 2: Remove `CSIRatioExtractor` class**

Find the `class CSIRatioExtractor:` definition (around line 64) and delete the entire class through its last method's closing line (approximately 37 lines).

- [ ] **Step 3: Remove `BreathingAnalyzer` class**

Find the `class BreathingAnalyzer:` definition (around line 104) and delete the entire class through its last method's closing line (approximately 49 lines).

- [ ] **Step 4: Verify breathing.py still imports cleanly**

Run: `python -c "from ghv5.breathing import BreathingDetector, GridProjector, CSIRingBuffer, reconstruct_csi_from_csv_row; print('OK')"`

Expected: `OK`

- [ ] **Step 5: Run the full test suite**

Run: `cd C:\GlassHouse\GHV5 && python -m pytest tests/ -v 2>&1 | tail -30`

Expected failures (tests for the now-deleted classes):
- `TestCSIRatioExtractor` — all 3 tests fail with `ImportError`
- `TestBreathingAnalyzer` — all 7 tests fail with `ImportError`

All other tests should pass. Confirm before continuing.

---

## Task 7: Remove Dead Tests and Update Imports

**Files:**
- Modify: `tests/test_breathing.py`

- [ ] **Step 1: Update the import block at the top of `test_breathing.py`**

Find (around line 10):
```python
from ghv5.breathing import (
    CSIRingBuffer,
    CSIRatioExtractor,
    BreathingAnalyzer,
    GridProjector,
    BreathingDetector,
    reconstruct_csi_from_csv_row,
)
```

Replace with:
```python
from ghv5.breathing import (
    CSIRingBuffer,
    GridProjector,
    BreathingDetector,
    reconstruct_csi_from_csv_row,
)
```

- [ ] **Step 2: Delete `TestCSIRatioExtractor` class**

Find `class TestCSIRatioExtractor:` (around line 82) and delete the entire class including all its test methods (approximately 34 lines, through `assert np.allclose(result, 0.0, atol=1e-5)`).

- [ ] **Step 3: Delete `TestBreathingAnalyzer` class**

Find `class TestBreathingAnalyzer:` (around line 118) and delete the entire class including all its test methods (approximately 51 lines, through `assert analyzer._fs == BREATHING_SNAP_HZ`).

- [ ] **Step 4: Run the full test suite**

Run: `cd C:\GlassHouse\GHV5 && python -m pytest tests/ -v 2>&1 | tail -20`

Expected: All tests pass. Count should be lower than before (removed ~10 dead tests). One test is skipped (pygame import guard) — that is expected.

---

## Task 8: Update Thread Dict Keys and Display in `breathing.py`

**Files:**
- Modify: `ghv5/breathing.py`

Update `BreathingThread`, `SARDemoThread`, and `BreathingDisplay` to use `"presence_grid"` instead of `"amp_grid"`, and `"Pr:"` instead of `"A:"`.

- [ ] **Step 1: Update `BreathingThread.run()` dict key**

Find in `BreathingThread.run()` (around line 498):
```python
                    self._q.put({
                        "type":      "scores",
                        "amp_grid":  all_scores["amp"],
                        "pca_grid":  all_scores["pca"],
                        "path_conf": all_scores["path_conf"],
                    })
```

Replace with:
```python
                    self._q.put({
                        "type":          "scores",
                        "presence_grid": all_scores["presence"],
                        "pca_grid":      all_scores["pca"],
                        "path_conf":     all_scores["path_conf"],
                    })
```

- [ ] **Step 2: Update `SARDemoThread.run()` dict key**

Find in `SARDemoThread.run()` (around line 535):
```python
            self._q.put({
                "type":      "scores",
                "amp_grid":  grid,
                "pca_grid":  grid,   # demo mode: duplicate amp projection
                "path_conf": path_conf,
            })
```

Replace with:
```python
            self._q.put({
                "type":          "scores",
                "presence_grid": grid,
                "pca_grid":      grid,   # demo mode: duplicate presence projection
                "path_conf":     path_conf,
            })
```

- [ ] **Step 3: Update `BreathingDisplay.__init__` field**

Find (around line 567):
```python
            self._amp_grid = {cell: None for cell in CELL_LABELS}
```

Replace with:
```python
            self._presence_grid = {cell: None for cell in CELL_LABELS}
```

- [ ] **Step 4: Update `BreathingDisplay.update()` parameter and field**

Find (around line 631):
```python
        def update(self, amp_grid, pca_grid, path_conf):
            self._amp_grid = amp_grid
            self._pca_grid = pca_grid
            self._path_conf = path_conf
```

Replace with:
```python
        def update(self, presence_grid, pca_grid, path_conf):
            self._presence_grid = presence_grid
            self._pca_grid = pca_grid
            self._path_conf = path_conf
```

- [ ] **Step 5: Update `BreathingDisplay.render()` — field reference and cell label**

Find (around line 672):
```python
                amp_score = self._amp_grid.get(label)
```

Replace with:
```python
                amp_score = self._presence_grid.get(label)
```

Find (around line 677):
```python
                    amp_text = f"A:{amp_score:.0f}%"
```

Replace with:
```python
                    amp_text = f"Pr:{amp_score:.0f}%"
```

Find (around line 681):
```python
                    amp_text = f"fill {cfill*100:.0f}%"
```
(Leave this unchanged — it's the fill progress display, not a label)

Find (around line 684):
```python
                    amp_text = "A:--"
```

Replace with:
```python
                    amp_text = "Pr:--"
```

- [ ] **Step 6: Verify breathing.py imports cleanly**

Run: `python -c "from ghv5.breathing import BreathingDetector, BreathingThread, SARDemoThread; print('OK')"`

Expected: `OK`

---

## Task 9: Update `TestDemoThread` Test

**Files:**
- Modify: `tests/test_breathing.py`

- [ ] **Step 1: Update `TestDemoThread.test_produces_valid_grid_scores`**

Find (around line 576):
```python
        assert item["type"] == "scores"
        assert "amp_grid" in item,  "'grid' key renamed to 'amp_grid'"
        assert "pca_grid" in item,  "new 'pca_grid' key expected"
        assert "grid" not in item,  "old 'grid' key must be removed"
        assert "r0c0" in item["amp_grid"]
        assert "r0c0" in item["pca_grid"]
        assert "path_conf" in item
```

Replace with:
```python
        assert item["type"] == "scores"
        assert "presence_grid" in item, "'amp_grid' key renamed to 'presence_grid'"
        assert "pca_grid" in item,      "pca_grid key expected"
        assert "amp_grid" not in item,  "old 'amp_grid' key must be removed"
        assert "r0c0" in item["presence_grid"]
        assert "r0c0" in item["pca_grid"]
        assert "path_conf" in item
```

- [ ] **Step 2: Run the demo thread test**

Run: `cd C:\GlassHouse\GHV5 && python -m pytest tests/test_breathing.py::TestDemoThread -v`

Expected: PASS.

---

## Task 10: Update `run_sar.py` Console Labels

**Files:**
- Modify: `run_sar.py`

- [ ] **Step 1: Update `_print_console` function signature and labels**

Find (around line 23):
```python
def _print_console(amp_scores: dict, pca_scores: dict, path_conf: dict):
    """Print a 3x3 grid with both amp and PCA scores to the console."""
```

Replace with:
```python
def _print_console(presence_scores: dict, pca_scores: dict, path_conf: dict):
    """Print a 3x3 grid with both presence and PCA scores to the console."""
```

- [ ] **Step 2: Update variable reference inside `_print_console`**

Find (around line 36):
```python
            a = amp_scores.get(cell)
```

Replace with:
```python
            a = presence_scores.get(cell)
```

- [ ] **Step 3: Update the call sites of `_print_console`**

Find (around line 109):
```python
                _print_console(all_scores["amp"], all_scores["pca"], all_scores["path_conf"])
```

Replace with:
```python
                _print_console(all_scores["presence"], all_scores["pca"], all_scores["path_conf"])
```

- [ ] **Step 4: Update `_run_pygame_loop` display.update() call**

Find (around line 156):
```python
                display.update(latest_scores["amp_grid"], latest_scores["pca_grid"], latest_scores["path_conf"])
```

Replace with:
```python
                display.update(latest_scores["presence_grid"], latest_scores["pca_grid"], latest_scores["path_conf"])
```

- [ ] **Step 5: Update console path confidence label**

Find (around line 46):
```python
    lines.append(f"Path confidence (amp): {' '.join(parts)}")
```

Replace with:
```python
    lines.append(f"Path confidence (presence): {' '.join(parts)}")
```

- [ ] **Step 6: Verify run_sar.py --help still works**

Run: `cd C:\GlassHouse\GHV5 && python run_sar.py --help`

Expected: exits 0, shows `--port`, `--demo`, `--display`, `--fullscreen`.

---

## Task 11: Final Verification

**Files:** None (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `cd C:\GlassHouse\GHV5 && python -m pytest tests/ -v`

Expected:
- All tests pass
- 1 test skipped (pygame import guard — `TestBreathingDisplay::test_import_guarded` if pygame not installed)
- No references to `CSIRatioExtractor`, `BreathingAnalyzer`, or `_amplitude_score` in passing tests

- [ ] **Step 2: Verify no stale references remain in source files**

Run: `grep -r "amp_grid\|_amplitude_score\|CSIRatioExtractor\|BreathingAnalyzer\|\"amp\"" ghv5/ run_sar.py tests/`

Expected: No output (all stale references cleaned up).

> **Exception:** The string `"amp"` may appear in comments or unrelated contexts — check any hits manually.

- [ ] **Step 3: Verify demo mode launches cleanly (if pygame available)**

Run: `cd C:\GlassHouse\GHV5 && timeout 3 python run_sar.py --demo --display pygame 2>&1 | head -5`

Expected: pygame initializes, "Demo mode — synthetic breathing" message appears, no `KeyError` or `AttributeError`.

---

## Post-Implementation: Sigmoid Tuning

After hardware testing with the new detector:

1. Run `python run_sar.py --port COMX --log-level DEBUG` with the room empty.
2. Note `path_var` values logged for each path.
3. Run again with a person stationary in the room.
4. Compare the empty vs. occupied `path_var` distributions.
5. Set `PRESENCE_VARIANCE_MIDPOINT` in `config.py` to the midpoint between the two distributions.
6. Re-run tests to confirm thresholds still hold.
