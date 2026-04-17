"""Complete audit of all 5 SAR-mode captures before firmware redesign.

Sections:
  1. Rate overview
  2. Vitals regression status
  3. SAR_AMP pair coverage (which pairs appear in which captures)
  4. Link variance per pair per capture
  5. SAR_AMP breathing-band ratio per (rx,tx) per capture
  6. Link-variance feature vector and L2 separability
"""

from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.signal import lombscargle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from python.frame_decoder import parse_packet

CAPS = [
    ("empty", "captures/capture_sar_empty.jsonl"),
    ("q1",    "captures/capture_sar_occupied_q1.jsonl"),
    ("q2",    "captures/capture_sar_occupied_q2.jsonl"),
    ("q3",    "captures/capture_sar_occupied_q3.jsonl"),
    ("q4",    "captures/capture_sar_occupied_q4.jsonl"),
]


def load(path):
    out = []
    for l in open(path):
        l = l.strip()
        if not l:
            continue
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out


def analyze(rows):
    types = collections.Counter()
    link_pairs = collections.Counter()
    link_var = collections.defaultdict(list)
    sar_b = collections.defaultdict(list)
    csi_per = collections.Counter()
    iq_per = collections.Counter()
    v_pres = v_tot = 0
    v_e = []
    for r in rows:
        raw = r.get("raw", "")
        if not raw:
            continue
        try:
            p = parse_packet(bytes.fromhex(raw))
        except Exception:
            continue
        t = p.get("type")
        types[t] += 1
        if t == "link":
            link_pairs[(p["node"], p["partner"])] += 1
            a, b = p["node"], p["partner"]
            link_var[f"{min(a,b)}{max(a,b)}"].append(p["variance"])
        elif t == "sar_amp":
            sar_b[(p["node_id"], p["peer_id"])].append({
                "host_t": r["t"], "n_samples": p["n_samples"],
                "interval_s": p["interval_us"] / 1e6, "amps": p["amps"],
            })
        elif t == "csi":
            csi_per[p["node_id"]] += 1
        elif t == "iq":
            iq_per[p["node_id"]] += 1
        elif t == "vitals":
            v_tot += 1
            if p.get("presence"):
                v_pres += 1
            v_e.append(p.get("motion_energy", 0))
    ts = [r.get("t", 0) for r in rows if "t" in r]
    dur = max(ts) - min(ts) if ts else 0
    return {
        "total": len(rows), "dur": dur, "types": types,
        "link_pairs": dict(link_pairs),
        "link_var": {k: {"n": len(v), "mean": statistics.mean(v),
                         "max": max(v), "stdev": statistics.stdev(v) if len(v) > 1 else 0}
                     for k, v in link_var.items()},
        "sar_b": sar_b, "csi_per": dict(csi_per), "iq_per": dict(iq_per),
        "v_pres": v_pres, "v_tot": v_tot,
        "v_e_mean": statistics.mean(v_e) if v_e else 0,
        "v_e_max": max(v_e) if v_e else 0,
    }


def breath_test(batches, min_span=30):
    if len(batches) < 2:
        return None
    all_t, all_a = [], []
    for b in batches:
        for i, amp in enumerate(b["amps"]):
            all_t.append(b["host_t"] - (len(b["amps"]) - 1 - i) * b["interval_s"])
            all_a.append(amp)
    order = np.argsort(all_t)
    t = np.array(all_t)[order]
    y = np.array(all_a)[order]
    span = t[-1] - t[0]
    if span < min_span:
        return {"span": span, "n": len(t), "reason": "short", "ratio": None}
    yc = y - y.mean()
    dt_med = np.median(np.diff(t))
    nyq = 0.5 / dt_med if dt_med > 0 else 0
    if nyq < 0.5:
        return {"span": span, "n": len(t), "reason": "nyq_low", "ratio": None}
    fb = np.linspace(0.1, min(0.5, nyq * 0.95), 40)
    f0 = np.linspace(0.6, min(2.0, nyq * 0.95), 40)
    if len(f0) < 2:
        return {"span": span, "n": len(t), "reason": "nyq_low", "ratio": None}
    pb = lombscargle(t, yc, 2 * np.pi * fb, normalize=False).mean()
    p0 = lombscargle(t, yc, 2 * np.pi * f0, normalize=False).mean()
    ff = np.linspace(0.05, min(2.0, nyq * 0.95), 200)
    pf = lombscargle(t, yc, 2 * np.pi * ff, normalize=False)
    return {"span": span, "n": len(t), "nyq": nyq,
            "ratio": pb / p0 if p0 > 0 else 0, "peak_Hz": float(ff[pf.argmax()])}


def main():
    res = {n: analyze(load(p)) for n, p in CAPS}

    # ============================================
    print("=" * 92)
    print("SECTION 1 -- RATE OVERVIEW")
    print("=" * 92)
    print(f"{'cap':<8}{'dur':>7}{'total':>8}{'rate':>8} | {'link':>6}{'csi':>6}{'iq':>5}{'vit':>5}{'hb':>6}{'sar':>5}")
    for n, r in res.items():
        t = r["types"]
        print(f"{n:<8}{r['dur']:>6.0f}s{r['total']:>8}{r['total']/r['dur']:>7.1f}Hz | "
              f"{t.get('link',0):>6}{t.get('csi',0):>6}{t.get('iq',0):>5}{t.get('vitals',0):>5}"
              f"{t.get('heartbeat',0):>6}{t.get('sar_amp',0):>5}")

    # ============================================
    print("\n" + "=" * 92)
    print("SECTION 2 -- VITALS")
    print("=" * 92)
    print(f"{'cap':<8}{'n_vitals':>10}{'presence_true':>17}{'energy_mean':>14}{'energy_max':>14}")
    for n, r in res.items():
        pct = 100 * r['v_pres'] / r['v_tot'] if r['v_tot'] else 0
        print(f"{n:<8}{r['v_tot']:>10}{r['v_pres']:>10} ({pct:>5.0f}%){r['v_e_mean']:>14.3f}{r['v_e_max']:>14.3f}")

    # ============================================
    print("\n" + "=" * 92)
    print("SECTION 3 -- SAR_AMP PAIR COVERAGE (batches per capture)")
    print("=" * 92)
    print(f"  {'pair':<10}{'empty':>7}{'q1':>5}{'q2':>5}{'q3':>5}{'q4':>5}{'persist':>10}")
    persist_all = 0
    for rx in [1, 2, 3, 4]:
        for tx in [1, 2, 3, 4]:
            if rx == tx:
                continue
            counts = [len(res[n]["sar_b"].get((rx, tx), [])) for n in ["empty", "q1", "q2", "q3", "q4"]]
            n_pres = sum(1 for c in counts if c > 0)
            if n_pres == 0:
                continue
            row = f"  rx{rx}<-tx{tx} " + "".join(f"{c:>5}" for c in counts) + f"{n_pres:>8}/5"
            print(row)
            if n_pres == 5:
                persist_all += 1
    print(f"\n  Pairs present in ALL 5 captures: {persist_all}/12")

    # ============================================
    print("\n" + "=" * 92)
    print("SECTION 4 -- LINK VARIANCE PER PAIR PER CAPTURE (mean)")
    print("=" * 92)
    print(f"  {'link':<6}{'empty':>10}{'q1':>10}{'q2':>10}{'q3':>10}{'q4':>10}")
    all_links = set()
    for r in res.values():
        all_links.update(r["link_var"].keys())
    for link in sorted(all_links):
        row = f"  {link:<6}"
        for n in ["empty", "q1", "q2", "q3", "q4"]:
            lv = res[n]["link_var"].get(link)
            if lv:
                row += f" {lv['mean']:>6.2f}(n{lv['n']:>3})"
            else:
                row += f"{'—':>10}"
        print(row)

    # ============================================
    print("\n" + "=" * 92)
    print("SECTION 5 -- SAR_AMP BREATHING-BAND TEST (ratio breath/base, peak Hz)")
    print("=" * 92)
    print(f"  {'pair':<10}{'empty':>12}{'q1':>12}{'q2':>12}{'q3':>12}{'q4':>12}    verdict")
    candidates = []
    for rx in [1, 2, 3, 4]:
        for tx in [1, 2, 3, 4]:
            if rx == tx:
                continue
            row = f"  rx{rx}<-tx{tx} "
            per = {}
            for n in ["empty", "q1", "q2", "q3", "q4"]:
                b = res[n]["sar_b"].get((rx, tx), [])
                rr = breath_test(b)
                if rr is None:
                    row += f"{'—':>12}"
                    per[n] = None
                elif rr.get("ratio") is None:
                    row += f"{'short':>12}"
                    per[n] = None
                else:
                    cell = f"{rr['ratio']:.2f}@{rr['peak_Hz']:.2f}Hz"
                    row += f"{cell:>12}"
                    per[n] = rr
            verdict = ""
            if per.get("empty") and all(per.get(q) for q in ["q1", "q2", "q3", "q4"]):
                er = per["empty"]["ratio"]
                occs = [per[q]["ratio"] for q in ["q1", "q2", "q3", "q4"]]
                peaks = [per[q]["peak_Hz"] for q in ["q1", "q2", "q3", "q4"]]
                if all(r > er * 1.3 for r in occs):
                    verdict = "   ALL OCC > empty * 1.3 **"
                    candidates.append((rx, tx, er, occs, peaks))
                elif all(r > er for r in occs):
                    verdict = "   all > empty (weak)"
            if not any(per.values()):
                continue
            print(row + verdict)

    if candidates:
        print("\n  Breathing-like candidate pairs:")
        for (rx, tx, er, occs, peaks) in candidates:
            print(f"    rx{rx}<-tx{tx}: empty={er:.2f}  q1..q4 ratios={[round(x,2) for x in occs]}  peaks={[round(x,2) for x in peaks]}")

    # ============================================
    print("\n" + "=" * 92)
    print("SECTION 6 -- LINK-VARIANCE FEATURE VECTOR (6-dim) AND L2 SEPARABILITY")
    print("=" * 92)
    fv = {}
    for n in ["empty", "q1", "q2", "q3", "q4"]:
        v = []
        for link in ["12", "13", "14", "23", "24", "34"]:
            lv = res[n]["link_var"].get(link, {"mean": 0})
            v.append(lv["mean"])
        fv[n] = np.array(v)
    print(f"  {'cap':<6}  {'12':>7}{'13':>7}{'14':>7}{'23':>7}{'24':>7}{'34':>7}")
    for n, vec in fv.items():
        print(f"  {n:<6}  " + "".join(f"{x:>7.2f}" for x in vec))

    print("\n  Pairwise L2 distances:")
    caps_ord = ["empty", "q1", "q2", "q3", "q4"]
    print(f"  {'':<8}" + "".join(f"{c:>8}" for c in caps_ord))
    for a in caps_ord:
        row = f"  {a:<8}"
        for b in caps_ord:
            row += f"{np.linalg.norm(fv[a]-fv[b]):>8.2f}"
        print(row)

    print("\n  Empty -> each-quadrant L2 distance (larger = more separable from empty):")
    for q in ["q1", "q2", "q3", "q4"]:
        print(f"    empty -> {q}: {np.linalg.norm(fv['empty']-fv[q]):.2f}")

    print("\n  Between-quadrant L2 (lower = more confusable):")
    qs = ["q1", "q2", "q3", "q4"]
    pairs_q = [(qs[i], qs[j]) for i in range(4) for j in range(i + 1, 4)]
    for a, b in pairs_q:
        print(f"    {a} <-> {b}: {np.linalg.norm(fv[a]-fv[b]):.2f}")

    # ============================================
    print("\n" + "=" * 92)
    print("SUMMARY")
    print("=" * 92)
    rates = {n: res[n]["total"] / res[n]["dur"] for n in res}
    print(f"  Aggregate packet rate across captures: "
          f"min={min(rates.values()):.1f}Hz  max={max(rates.values()):.1f}Hz")
    print(f"  Vitals emission: "
          f"{'WORKING' if all(res[n]['v_tot'] > 0 for n in res) else 'BROKEN ('+str([n for n in res if res[n]['v_tot']==0])+' have 0 vitals)'}")
    print(f"  SAR_AMP pairs present in ALL 5 captures: {persist_all}/12")
    print(f"  SAR_AMP breathing-band candidates (all occ > empty*1.3): {len(candidates)}")
    emp_to_occ = [np.linalg.norm(fv['empty'] - fv[q]) for q in ['q1', 'q2', 'q3', 'q4']]
    within_occ = [np.linalg.norm(fv[a] - fv[b]) for a, b in pairs_q]
    print(f"  Link-variance empty->occupied distance: min={min(emp_to_occ):.2f} max={max(emp_to_occ):.2f}")
    print(f"  Link-variance within-occupied distance: min={min(within_occ):.2f} max={max(within_occ):.2f}")


if __name__ == "__main__":
    main()
