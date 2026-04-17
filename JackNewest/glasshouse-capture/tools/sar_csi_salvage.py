"""SAR Option B — salvage attempt on the raw CSI I/Q stream.

Stream rate reality
-------------------
CSI frame rate is 0.38–0.77 Hz per capture (see captures_v2/). Nyquist ≈0.2–0.4
Hz. Breathing's lower edge (0.1–0.2 Hz) is still within Nyquist; upper edge
(0.3–0.5 Hz) is aliased. Heart rate (0.8–2.0 Hz) is unreachable.

What we test
------------
For each capture, for each subcarrier k, we build a time-series of amplitude
√(I² + Q²) and ask two questions:

  T1 (periodicity, calibration-aware):
        Does any subcarrier show Lomb–Scargle power in the slow-breathing band
        (0.05–0.20 Hz) that exceeds its own off-band baseline (0.20–Nyquist)?
        Report the fraction of subcarriers with ratio > 2.0.

  T2 (temporal stationarity, calibration-free-ish):
        For each subcarrier, compute stdev of amplitude over the capture.
        A static empty room should have lower per-subcarrier stdev than an
        occupied one IF the person creates ongoing multipath flicker.
        Report median-across-subcarriers of stdev/mean (coefficient of
        variation, dimensionless and self-normalising).

T2 is the nearest proxy to a calibration-free SAR signature using only what
this stream carries. A clear empty-vs-occupied gap on T2 would be salvageable.

Run from glasshouse-capture/:
    python -m tools.sar_csi_salvage
"""

from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.signal import lombscargle

CAPTURES_DIR = Path("captures_v2")

SLOW_BREATH_LO = 0.05
SLOW_BREATH_HI = 0.20


def parse_csi_amplitudes(raw_hex: str) -> np.ndarray | None:
    """Return per-subcarrier amplitude √(I²+Q²) for one CSI frame.

    Skips the 20-byte header. I/Q payload is interleaved signed int8 pairs.
    """
    try:
        b = bytes.fromhex(raw_hex)
    except ValueError:
        return None
    if len(b) < 24 or b[:4] != b'\x01\x00\x11\xc5':
        return None
    iq = b[20:]
    n_pairs = len(iq) // 2
    if n_pairs < 32:
        return None
    iq_arr = np.frombuffer(iq[: 2 * n_pairs], dtype=np.int8).reshape(-1, 2)
    amps = np.sqrt(iq_arr[:, 0].astype(float) ** 2 + iq_arr[:, 1].astype(float) ** 2)
    return amps


def analyse_capture(path: Path) -> dict:
    per_node_t: dict[int, list[float]] = collections.defaultdict(list)
    per_node_amps: dict[int, list[np.ndarray]] = collections.defaultdict(list)

    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") != "csi":
                continue
            amps = parse_csi_amplitudes(r.get("raw", ""))
            if amps is None:
                continue
            nid = r["node_id"]
            per_node_t[nid].append(r["t"])
            per_node_amps[nid].append(amps)

    # Per-node analysis
    node_reports = {}
    for nid, t_list in per_node_t.items():
        amps_list = per_node_amps[nid]
        if len(t_list) < 30:
            node_reports[nid] = {"n_frames": len(t_list), "reason": "too few frames"}
            continue

        # Trim every frame to the MINIMUM subcarrier count so indexes align
        min_sc = min(a.shape[0] for a in amps_list)
        # Guard against pathological 1-subcarrier outliers
        if min_sc < 16:
            # Drop the smallest 10% and retry
            lengths = sorted(a.shape[0] for a in amps_list)
            cutoff = lengths[max(1, len(lengths) // 10)]
            kept = [(t, a) for t, a in zip(t_list, amps_list) if a.shape[0] >= cutoff]
            if len(kept) < 30:
                node_reports[nid] = {"n_frames": len(t_list), "reason": f"min_sc={min_sc} too few long frames"}
                continue
            t_list = [t for t, _ in kept]
            amps_list = [a for _, a in kept]
            min_sc = min(a.shape[0] for a in amps_list)

        mat = np.array([a[:min_sc] for a in amps_list])  # (n_frames, min_sc)
        t_arr = np.array(t_list, dtype=float)

        # T1 — periodicity check per subcarrier
        dt = float(np.median(np.diff(t_arr)))
        nyquist = 0.5 / dt if dt > 0 else 0.0
        slow_hi = min(SLOW_BREATH_HI, nyquist * 0.95)
        slow_lo = SLOW_BREATH_LO
        base_lo = slow_hi
        base_hi = max(nyquist * 0.95, base_lo + 0.01)

        if slow_hi <= slow_lo or base_hi <= base_lo or nyquist <= slow_lo:
            ratios = np.zeros(min_sc)
        else:
            omega_b = 2.0 * np.pi * np.linspace(slow_lo, slow_hi, 40)
            omega_0 = 2.0 * np.pi * np.linspace(base_lo, base_hi, 40)
            ratios = np.zeros(min_sc)
            for k in range(min_sc):
                y = mat[:, k]
                y = y - y.mean()
                if y.std() < 1e-6:
                    continue
                p_b = lombscargle(t_arr, y, omega_b, normalize=False).mean()
                p_0 = lombscargle(t_arr, y, omega_0, normalize=False).mean()
                if p_0 > 1e-9:
                    ratios[k] = p_b / p_0

        frac_sc_above2 = float(np.mean(ratios > 2.0))
        frac_sc_above3 = float(np.mean(ratios > 3.0))
        mean_ratio = float(ratios.mean())
        max_ratio = float(ratios.max()) if ratios.size else 0.0

        # T2 — temporal CoV (stationarity)
        means = mat.mean(axis=0)
        stds = mat.std(axis=0)
        nz = means > 0.5
        cov_per_sc = np.zeros(min_sc)
        cov_per_sc[nz] = stds[nz] / means[nz]
        median_cov = float(np.median(cov_per_sc[nz])) if nz.any() else 0.0
        p90_cov    = float(np.percentile(cov_per_sc[nz], 90)) if nz.any() else 0.0

        node_reports[nid] = {
            "n_frames": int(len(t_list)),
            "n_subcar": int(min_sc),
            "nyquist_hz": round(nyquist, 3),
            "frac_sc_ratio_gt2": round(frac_sc_above2, 3),
            "frac_sc_ratio_gt3": round(frac_sc_above3, 3),
            "mean_ratio_across_sc": round(mean_ratio, 3),
            "max_ratio_across_sc": round(max_ratio, 3),
            "median_cov": round(median_cov, 3),
            "p90_cov": round(p90_cov, 3),
        }

    return {"file": path.name, "per_node": node_reports}


def main() -> int:
    if not CAPTURES_DIR.exists():
        print(f"error: {CAPTURES_DIR} not found", file=sys.stderr)
        return 2

    results = [analyse_capture(p) for p in sorted(CAPTURES_DIR.glob("capture_*.jsonl"))
               if p.stat().st_size > 0]

    print("=" * 105)
    print("SAR Option B — CSI raw I/Q salvage test")
    print(f"  T1 slow-breathing band: {SLOW_BREATH_LO}–{SLOW_BREATH_HI} Hz (or up to Nyquist)")
    print(f"  T2 coefficient of variation (stdev/mean per subcarrier, median across subcarriers)")
    print("=" * 105)

    print(f"\n{'capture':<30}{'node':>5}{'frames':>8}{'subcar':>8}{'nyq':>7}"
          f"{'frac_r>2':>10}{'frac_r>3':>10}{'max_r':>8}{'med_cov':>9}{'p90_cov':>9}")
    for r in results:
        for nid, s in sorted(r["per_node"].items()):
            if "reason" in s:
                print(f"{r['file']:<30}{nid:>5}{s['n_frames']:>8}  (skipped: {s['reason']})")
                continue
            print(f"{r['file']:<30}{nid:>5}{s['n_frames']:>8}{s['n_subcar']:>8}"
                  f"{s['nyquist_hz']:>7.2f}{s['frac_sc_ratio_gt2']:>10.2f}"
                  f"{s['frac_sc_ratio_gt3']:>10.2f}{s['max_ratio_across_sc']:>8.2f}"
                  f"{s['median_cov']:>9.3f}{s['p90_cov']:>9.3f}")

    # SAR aggregate: per-capture max over nodes of each statistic
    print("\n" + "=" * 105)
    print("Per-capture aggregate (max over nodes)")
    print("=" * 105)
    print(f"\n{'capture':<30}{'max_frac_r>2':>15}{'max_frac_r>3':>15}{'max_r':>8}"
          f"{'max_med_cov':>14}{'max_p90_cov':>14}")

    per_cap = {}
    for r in results:
        valid = [s for s in r["per_node"].values() if "reason" not in s]
        if not valid:
            continue
        agg = {
            "max_frac_r_gt2": max(s['frac_sc_ratio_gt2'] for s in valid),
            "max_frac_r_gt3": max(s['frac_sc_ratio_gt3'] for s in valid),
            "max_max_r":     max(s['max_ratio_across_sc'] for s in valid),
            "max_med_cov":   max(s['median_cov'] for s in valid),
            "max_p90_cov":   max(s['p90_cov'] for s in valid),
        }
        per_cap[r['file']] = agg
        print(f"{r['file']:<30}{agg['max_frac_r_gt2']:>15.2f}{agg['max_frac_r_gt3']:>15.2f}"
              f"{agg['max_max_r']:>8.2f}{agg['max_med_cov']:>14.3f}{agg['max_p90_cov']:>14.3f}")

    # SAR decision
    print("\n" + "=" * 105)
    print("SAR decision — can any of T1 / T2 separate empty from occupied")
    print("           WITHOUT using the empty capture as a training reference?")
    print("=" * 105)
    emp_key = next((k for k in per_cap if k.startswith("capture_empty")), None)
    occ_keys = [k for k in per_cap if k.startswith("capture_occupied_")]
    if not emp_key or not occ_keys:
        print("Insufficient captures.")
        return 0
    emp = per_cap[emp_key]
    print(f"\nEmpty baseline (single-session, 1 room only):")
    for k, v in emp.items():
        print(f"   {k:<20} = {v:.3f}")

    for stat in ["max_frac_r_gt2", "max_med_cov", "max_p90_cov"]:
        occ_vals = [per_cap[k][stat] for k in occ_keys]
        emp_val = emp[stat]
        gap = min(occ_vals) - emp_val
        all_above = all(v > emp_val for v in occ_vals)
        print(f"\n  Statistic: {stat}")
        print(f"    empty:    {emp_val:.3f}")
        for k in occ_keys:
            tag = " (>empty)" if per_cap[k][stat] > emp_val else " (<=empty)"
            print(f"    {k:<30}: {per_cap[k][stat]:.3f}{tag}")
        print(f"    min(occupied) - empty = {gap:+.3f}    all-above-empty: {all_above}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
