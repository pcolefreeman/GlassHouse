"""SAR feasibility test — single-capture, calibration-free breathing detection.

Operating constraint: in a Search-And-Rescue deployment the responder CANNOT
pre-calibrate an empty baseline for the specific room. The system must decide,
from one fresh capture alone, whether a human is present — and ideally where.

This script tests ONE falsifiable claim:

    CLAIM: The link_reporter variance time-series contains a detectable
           breathing-band (0.1–0.5 Hz) spectral component when a human is
           present in the room, and that component is absent (or much weaker)
           in the empty-room capture — WITHOUT needing the empty capture as
           a reference.

If the claim holds: a per-capture breathing-band power statistic separates
empty from occupied, which is the minimum SAR signature.

If it fails: the data (as currently captured) does NOT support calibration-
free presence detection, independent of any classifier design.

Method
------
1.  Read captures_v2/capture_*.jsonl.
2.  For each capture and each of the 6 link pairs, extract the variance
    time-series: (t, variance).
3.  Lomb–Scargle periodogram (handles irregular sampling).
4.  Compute power in the breathing band (0.1–0.5 Hz) vs out-of-band reference
    (0.6–1.2 Hz, above breathing and below any possible HR component).
5.  Report the ratio breathing_power / out_of_band_power per pair per capture.
    A value well above 1.0, with the breathing-band also exceeding a flat-
    spectrum expectation, is evidence of a periodic perturbation in the
    breathing band.

SAR decision metric
-------------------
    max_over_links( breathing_power / baseline_power )

    If empty_capture.max < 2.0 and every occupied_capture.max > 3.0,
    a calibration-free threshold near 2.5 could discriminate presence without
    the empty baseline. That would prove SAR feasibility in principle.

Run from glasshouse-capture/:
    python -m tools.sar_breathing_test
"""

from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.signal import lombscargle

BREATHING_LO = 0.1   # Hz
BREATHING_HI = 0.5
BASELINE_LO  = 0.6
BASELINE_HI  = 1.2

CAPTURES_DIR = Path("captures_v2")


def load_link_series(path: Path) -> dict[str, list[tuple[float, float]]]:
    """Return {pair: [(t, variance), ...]} for one capture."""
    series: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") != "link":
                continue
            series[r["link"]].append((r["t"], r["variance"]))
    for k in series:
        series[k].sort(key=lambda p: p[0])
    return dict(series)


def lomb_power(t: np.ndarray, y: np.ndarray, freq_lo: float, freq_hi: float,
               n_freqs: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Lomb–Scargle periodogram over a bounded frequency range.

    Returns (freqs_hz, power) — power is the raw LS periodogram.
    """
    freqs = np.linspace(freq_lo, freq_hi, n_freqs)
    omega = 2.0 * np.pi * freqs
    pwr = lombscargle(t, y - y.mean(), omega, normalize=False)
    return freqs, pwr


def breathing_stat(t: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Breathing-band power + baseline power + ratio for one series."""
    if len(t) < 20 or (t[-1] - t[0]) < 60.0:
        return {"n": len(t), "dur_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
                "nyquist_hz": 0.0, "breath_pwr": 0.0, "base_pwr": 0.0,
                "ratio": 0.0, "reason": "too short"}

    avg_dt = float(np.mean(np.diff(t)))
    nyquist = 0.5 / avg_dt if avg_dt > 0 else 0.0

    _, pwr_b = lomb_power(t, y, BREATHING_LO, BREATHING_HI)
    _, pwr_0 = lomb_power(t, y, BASELINE_LO, min(BASELINE_HI, nyquist * 0.95))
    breath = float(pwr_b.mean())
    base = float(pwr_0.mean()) if pwr_0.size else 1e-9
    ratio = breath / base if base > 1e-9 else 0.0

    return {"n": len(t), "dur_s": float(t[-1] - t[0]),
            "nyquist_hz": round(nyquist, 3),
            "breath_pwr": round(breath, 4),
            "base_pwr":   round(base, 4),
            "ratio":      round(ratio, 3),
            "reason": ""}


def analyse_capture(path: Path) -> dict:
    series = load_link_series(path)
    per_pair = {}
    for pair, samples in series.items():
        t = np.array([s[0] for s in samples], dtype=float)
        y = np.array([s[1] for s in samples], dtype=float)
        per_pair[pair] = breathing_stat(t, y)
    ratios = [v["ratio"] for v in per_pair.values() if v["ratio"] > 0]
    return {
        "file": path.name,
        "per_pair": per_pair,
        "max_ratio":    round(max(ratios), 3) if ratios else 0.0,
        "median_ratio": round(statistics.median(ratios), 3) if ratios else 0.0,
        "nyquist_min":  round(min(v["nyquist_hz"] for v in per_pair.values() if v["nyquist_hz"] > 0), 3) if per_pair else 0.0,
        "nyquist_max":  round(max(v["nyquist_hz"] for v in per_pair.values() if v["nyquist_hz"] > 0), 3) if per_pair else 0.0,
    }


def main() -> int:
    if not CAPTURES_DIR.exists():
        print(f"error: {CAPTURES_DIR} not found (run from glasshouse-capture/)", file=sys.stderr)
        return 2

    results = []
    for path in sorted(CAPTURES_DIR.glob("capture_*.jsonl")):
        if path.stat().st_size == 0:
            continue
        results.append(analyse_capture(path))

    print("=" * 100)
    print("SAR feasibility test — breathing-band power in link-variance time series")
    print(f"breathing band {BREATHING_LO}–{BREATHING_HI} Hz | baseline {BASELINE_LO}–{BASELINE_HI} Hz")
    print("=" * 100)
    print()
    print(f"{'capture':<20} {'max_ratio':>10} {'median_ratio':>14} {'nyquist min':>12} {'nyquist max':>12}")
    for r in results:
        print(f"{r['file']:<20} {r['max_ratio']:>10} {r['median_ratio']:>14} "
              f"{r['nyquist_min']:>12} {r['nyquist_max']:>12}")

    print("\nPer-capture per-pair detail:")
    for r in results:
        print(f"\n--- {r['file']} ---")
        print(f"  {'pair':<8} {'n':>5} {'dur_s':>8} {'nyq':>6} {'breath':>10} {'base':>10} {'ratio':>8} {'note':<12}")
        for pair, s in r["per_pair"].items():
            print(f"  link{pair:<4} {s['n']:>5} {s['dur_s']:>8.1f} {s['nyquist_hz']:>6.2f} "
                  f"{s['breath_pwr']:>10.3f} {s['base_pwr']:>10.3f} {s['ratio']:>8.2f}  {s.get('reason',''):<12}")

    # SAR-mode decision
    print("\n" + "=" * 100)
    print("SAR decision")
    print("=" * 100)

    empty = next((r for r in results if r["file"].startswith("capture_empty")), None)
    occupied = [r for r in results if r["file"].startswith("capture_occupied_")]
    if not empty or not occupied:
        print("Insufficient captures for SAR decision.")
        return 0

    emp_max = empty["max_ratio"]
    occ_maxes = [(r["file"], r["max_ratio"]) for r in occupied]
    print(f"empty max ratio        : {emp_max}")
    for name, m in occ_maxes:
        delta = m - emp_max
        tag = " <- exceeds empty" if m > emp_max * 1.3 else \
              (" (marginal)"      if m > emp_max else " <- BELOW empty")
        print(f"{name:<30} max ratio: {m:>6.2f}   (vs empty: {delta:+.2f}){tag}")

    separable = all(m > emp_max * 1.3 for _, m in occ_maxes)
    reverse_or_weak = any(m <= emp_max for _, m in occ_maxes)

    print()
    if separable:
        print("RESULT: Calibration-free breathing-band signal is PLAUSIBLE.")
        print("   All occupied captures exceed empty max_ratio by >30%.")
        print("   A per-capture threshold near {:.2f} would separate empty from occupied".format(
            (emp_max + min(m for _, m in occ_maxes)) / 2))
        print("   WITHOUT needing the empty capture as a reference.")
    elif reverse_or_weak:
        print("RESULT: Calibration-free breathing signal is NOT demonstrable on this data.")
        print("   At least one occupied capture has ratio <= empty capture.")
        print("   The link-variance stream alone is INSUFFICIENT for SAR-mode presence")
        print("   detection. Deeper features (CSI per-subcarrier phase) are required.")
    else:
        print("RESULT: Marginal. Occupied captures exceed empty but by <30%.")
        print("   Not a robust SAR-mode signature. Would need more sessions to confirm")
        print("   whether the effect is reproducible.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
