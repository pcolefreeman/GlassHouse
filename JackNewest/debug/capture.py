"""Detailed CSI capture & analysis tool for tuning zone detection.

Logs every packet at full rate to a JSON-lines file, then runs
statistical analysis to find optimal thresholds.

Usage:
    python -m debug.capture --port COM9 --seconds 30 --label empty
    python -m debug.capture --port COM9 --seconds 30 --label occupied_q3
    python -m debug.capture --analyze debug/capture_empty.jsonl debug/capture_occupied_q3.jsonl
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

# ── Capture mode ──────────────────────────────────────────────────────

def capture(port: str, baud: int, seconds: float, label: str,
            delay: float = 0) -> Path:
    from python.serial_receiver import SerialReceiver

    out_path = Path(f"debug/capture_{label}.jsonl")
    receiver = SerialReceiver(port=port, baud=baud)

    if delay > 0:
        print(f"Get into position for '{label}' capture...")
        for remaining in range(int(delay), 0, -1):
            print(f"  Starting in {remaining}s...", end="\r")
            time.sleep(1)
        print(f"  GO — capturing now!          ")

    print(f"Capturing to {out_path} for {seconds}s  (label={label})")
    print("Press Ctrl+C to stop early.\n")

    pkt_count = 0
    link_count = 0
    vitals_count = 0
    iq_count = 0
    start = time.monotonic()

    with open(out_path, "w") as f:
        try:
            for packet in receiver.read_packets():
                elapsed = time.monotonic() - start
                if elapsed > seconds:
                    break

                record = {"t": round(elapsed, 4), "label": label}

                if packet[0] == 0x01 and len(packet) == 10:
                    _, node, partner, variance, state, count = struct.unpack(
                        '<BBBfBH', packet[:10]
                    )
                    lo, hi = min(node, partner), max(node, partner)
                    record.update({
                        "type": "link",
                        "link": f"{lo}{hi}",
                        "node": node,
                        "partner": partner,
                        "variance": round(variance, 6),
                        "state": int(state),
                        "count": count,
                    })
                    link_count += 1

                elif len(packet) >= 4 and packet[:4] == b'\x02\x00\x11\xC5':
                    if len(packet) >= 32:
                        flags = packet[5]
                        energy = struct.unpack_from('<f', packet, 16)[0]
                        record.update({
                            "type": "vitals",
                            "flags": flags,
                            "presence": bool(flags & 0x01),
                            "motion_bit": bool(flags & 0x04),
                            "motion_energy": round(energy, 6),
                        })
                        vitals_count += 1
                    else:
                        record.update({"type": "vitals_short", "len": len(packet)})
                elif len(packet) >= 8 and packet[:4] == b'\x06\x00\x11\xC5':
                    node_id = packet[4]
                    channel = packet[5]
                    iq_len = struct.unpack_from('<H', packet, 6)[0]
                    record.update({
                        "type": "iq",
                        "node_id": node_id,
                        "channel": channel,
                        "iq_len": iq_len,
                        "hex": packet[8:40].hex(),
                    })
                    iq_count += 1

                else:
                    record.update({"type": "unknown", "len": len(packet),
                                   "hex": packet[:16].hex()})

                pkt_count += 1
                try:
                    f.write(json.dumps(record) + "\n")
                except (ValueError, TypeError) as exc:
                    print(f"\n  [warn] skipping malformed record: {exc}",
                          file=sys.stderr)

                # Live progress every 2s
                if pkt_count % 50 == 0:
                    print(f"\r  {elapsed:.1f}s  pkts={pkt_count} "
                          f"links={link_count} vitals={vitals_count} "
                          f"iq={iq_count}", end="")

        except KeyboardInterrupt:
            pass
        finally:
            receiver.close()

    print(f"\n\nDone: {pkt_count} packets -> {out_path}")
    return out_path


# ── Analysis mode ─────────────────────────────────────────────────────

def analyze(*paths: str) -> None:
    """Load captures and print detailed statistics."""
    from collections import defaultdict

    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue

        records = [json.loads(line) for line in open(path)]
        label = records[0].get("label", path.stem) if records else "?"
        links = [r for r in records if r.get("type") == "link"]
        vitals = [r for r in records if r.get("type") == "vitals"]

        print(f"\n{'='*70}")
        print(f"  FILE: {path}  label={label}")
        print(f"  Total: {len(records)} packets | {len(links)} link | {len(vitals)} vitals")
        duration = records[-1]["t"] - records[0]["t"] if len(records) > 1 else 0
        print(f"  Duration: {duration:.1f}s")
        print(f"{'='*70}")

        # ── Vitals analysis ──
        if vitals:
            energies = [v["motion_energy"] for v in vitals]
            print(f"\n  VITALS ({len(vitals)} packets, ~{len(vitals)/max(duration,1):.1f}/s):")
            print(f"    motion_energy: min={min(energies):.4f}  max={max(energies):.4f}  "
                  f"mean={sum(energies)/len(energies):.4f}")
            print(f"    presence_bit:  {sum(1 for v in vitals if v['presence'])}/{len(vitals)}")
            print(f"    motion_bit:    {sum(1 for v in vitals if v['motion_bit'])}/{len(vitals)}")

            # Energy distribution buckets
            buckets = [0, 0.5, 1, 2, 3, 5, 8, 15, 30, 100]
            print(f"    energy distribution:")
            for i in range(len(buckets) - 1):
                lo, hi = buckets[i], buckets[i+1]
                n = sum(1 for e in energies if lo <= e < hi)
                bar = "#" * n
                print(f"      [{lo:5.1f}, {hi:5.1f})  {n:3d}  {bar}")
            n = sum(1 for e in energies if e >= buckets[-1])
            if n:
                print(f"      [{buckets[-1]:5.1f},  inf)  {n:3d}  {'#'*n}")

        # ── I/Q analysis ──
        iq_pkts = [r for r in records if r.get("type") == "iq"]
        if iq_pkts:
            print(f"\n  I/Q PACKETS ({len(iq_pkts)}, "
                  f"~{len(iq_pkts)/max(duration,1):.1f}/s):")
            by_node: dict[int, list[dict]] = defaultdict(list)
            for r in iq_pkts:
                by_node[r["node_id"]].append(r)
            for nid in sorted(by_node):
                recs = by_node[nid]
                iq_lens = [r["iq_len"] for r in recs]
                print(f"    Node {nid}: {len(recs)} packets, "
                      f"iq_len min={min(iq_lens)} max={max(iq_lens)} "
                      f"mean={sum(iq_lens)/len(iq_lens):.0f}")

        # ── Per-link analysis ──
        if links:
            by_link: dict[str, list[dict]] = defaultdict(list)
            for r in links:
                by_link[r["link"]].append(r)

            print(f"\n  LINKS ({len(links)} reports across {len(by_link)} links):")
            for lid in sorted(by_link):
                recs = by_link[lid]
                variances = [r["variance"] for r in recs]
                nonzero = [v for v in variances if v >= 0.1]
                rate = len(recs) / max(duration, 1)

                print(f"\n    Link {lid}  ({len(recs)} reports, {rate:.1f}/s):")
                print(f"      variance: min={min(variances):.4f}  max={max(variances):.4f}  "
                      f"mean={sum(variances)/len(variances):.4f}")
                if nonzero:
                    print(f"      non-zero (>=0.1): min={min(nonzero):.4f}  max={max(nonzero):.4f}  "
                          f"mean={sum(nonzero)/len(nonzero):.4f}  n={len(nonzero)}/{len(variances)}")
                else:
                    print(f"      non-zero (>=0.1): NONE")

                # Percentiles
                sv = sorted(variances)
                for p in [50, 75, 90, 95, 99]:
                    idx = min(int(len(sv) * p / 100), len(sv) - 1)
                    print(f"      p{p}: {sv[idx]:.4f}", end="")
                print()

                # Variance distribution
                vbuckets = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
                for i in range(len(vbuckets) - 1):
                    lo, hi = vbuckets[i], vbuckets[i+1]
                    n = sum(1 for v in variances if lo <= v < hi)
                    if n > 0:
                        bar = "#" * min(n, 50)
                        print(f"        [{lo:5.1f}, {hi:5.1f})  {n:3d}  {bar}")

                # Direction analysis (node->partner vs partner->node)
                fwd = [r for r in recs if r["node"] < r["partner"]]
                rev = [r for r in recs if r["node"] > r["partner"]]
                if fwd and rev:
                    fwd_mean = sum(r["variance"] for r in fwd) / len(fwd)
                    rev_mean = sum(r["variance"] for r in rev) / len(rev)
                    print(f"      direction: {lid[0]}->{lid[1]} mean={fwd_mean:.4f} (n={len(fwd)})  "
                          f"{lid[1]}->{lid[0]} mean={rev_mean:.4f} (n={len(rev)})")

                # Motion state breakdown
                motion_n = sum(1 for r in recs if r["state"] == 1)
                print(f"      motion_state=1: {motion_n}/{len(recs)} "
                      f"({100*motion_n/len(recs):.0f}%)")

            # ── Inter-link spike correlation ──
            print(f"\n  SPIKE ANALYSIS (variance > baseline estimate):")
            # Use median as baseline proxy, flag spikes > 3x median
            for lid in sorted(by_link):
                recs = by_link[lid]
                variances = [r["variance"] for r in recs]
                nonzero = [v for v in variances if v >= 0.1]
                if not nonzero:
                    continue
                median = sorted(nonzero)[len(nonzero)//2]
                spikes = [(r["t"], r["variance"]) for r in recs
                          if r["variance"] > median * 3.0 and r["variance"] >= 0.1]
                if spikes:
                    print(f"    Link {lid}: median={median:.4f}, "
                          f"{len(spikes)} spikes >3x median:")
                    for t, v in spikes[:10]:
                        print(f"      t={t:.2f}s  var={v:.4f}  ratio={v/median:.1f}x")
                    if len(spikes) > 10:
                        print(f"      ... and {len(spikes)-10} more")

        # ── Threshold sweep (if single file) ──
        if links:
            print(f"\n  THRESHOLD SWEEP (simulated false-positive rate):")
            print(f"    Shows how many frames would trigger at each deviation multiplier,")
            print(f"    using per-link median as baseline proxy.\n")

            by_link_arr: dict[str, list[float]] = defaultdict(list)
            for r in links:
                by_link_arr[r["link"]].append(r["variance"])

            medians = {}
            for lid in by_link_arr:
                nonzero = sorted(v for v in by_link_arr[lid] if v >= 0.1)
                medians[lid] = nonzero[len(nonzero)//2] if nonzero else 1.0

            # Build time-bucketed frames (~0.2s windows like main loop)
            time_frames: dict[int, dict[str, float]] = defaultdict(dict)
            for r in links:
                frame_idx = int(r["t"] / 0.2)
                lid = r["link"]
                # Keep max variance per link per frame
                if lid not in time_frames[frame_idx] or r["variance"] > time_frames[frame_idx][lid]:
                    time_frames[frame_idx][lid] = r["variance"]

            total_frames = len(time_frames)
            for mult in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
                triggered = 0
                for frame_idx, frame_links in time_frames.items():
                    active_count = sum(
                        1 for lid, v in frame_links.items()
                        if v >= 0.1 and lid in medians and v > medians[lid] * mult
                    )
                    if active_count >= 2:
                        triggered += 1
                pct = 100 * triggered / max(total_frames, 1)
                bar = "#" * triggered
                print(f"    mult={mult:.1f}:  {triggered:3d}/{total_frames} frames ({pct:.1f}%)  {bar}")

    # ── Cross-file comparison ──
    if len(paths) >= 2:
        print(f"\n{'='*70}")
        print(f"  COMPARISON: {' vs '.join(str(Path(p).stem) for p in paths)}")
        print(f"{'='*70}")

        all_data = {}
        for path_str in paths:
            path = Path(path_str)
            records = [json.loads(line) for line in open(path)]
            label = records[0].get("label", path.stem) if records else "?"
            all_data[label] = records

        # Compare energy distributions
        print(f"\n  Energy comparison:")
        for label, records in all_data.items():
            vitals = [r for r in records if r.get("type") == "vitals"]
            if vitals:
                energies = [v["motion_energy"] for v in vitals]
                print(f"    {label:20s}: n={len(energies):3d}  "
                      f"min={min(energies):.2f}  max={max(energies):.2f}  "
                      f"mean={sum(energies)/len(energies):.2f}")

        # Compare per-link variance distributions
        print(f"\n  Per-link variance comparison (non-zero mean ± max):")
        for lid in ["12", "13", "14", "23", "24", "34"]:
            row = f"    Link {lid}: "
            for label, records in all_data.items():
                link_recs = [r for r in records
                             if r.get("type") == "link" and r.get("link") == lid]
                nonzero = [r["variance"] for r in link_recs if r["variance"] >= 0.1]
                if nonzero:
                    mean = sum(nonzero) / len(nonzero)
                    row += f" {label}={mean:.2f}(max {max(nonzero):.1f})"
                else:
                    row += f" {label}=N/A"
            print(row)

        # Suggest threshold
        print(f"\n  SUGGESTED THRESHOLDS:")
        for label, records in all_data.items():
            vitals = [r for r in records if r.get("type") == "vitals"]
            if vitals:
                energies = [v["motion_energy"] for v in vitals]
                print(f"    {label} energy range: [{min(energies):.2f}, {max(energies):.2f}]")
        print(f"    -> Set _ENERGY_THRESHOLD between empty max and occupied min")


# ── Shared JSONL loader ──────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, skipping malformed lines with a stderr warning."""
    records: list[dict] = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [warn] {path}:{lineno}: skipping bad line: {exc}",
                      file=sys.stderr)
    return records


# ── Replay mode ──────────────────────────────────────────────────────

#: Frame window for replay bucketing — matches main loop rate (~5 Hz).
REPLAY_FRAME_WINDOW = 0.2


def replay(*paths: str) -> None:
    """Replay captured JSONL data through ZoneDetector frame by frame."""
    from collections import defaultdict
    from python.zone_detector import ZoneDetector, LINK_ZONE_WEIGHTS, LINK_ABSORPTION_WEIGHTS

    # Collect all weight-table link IDs (for absent-link padding)
    all_weight_links = set(LINK_ZONE_WEIGHTS) | set(LINK_ABSORPTION_WEIGHTS)

    # Load and merge link reports from all files, sorted by time
    all_links: list[dict] = []
    file_label: str | None = None
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        records = _load_jsonl(path)
        file_label = records[0].get("label") if records else None
        all_links.extend(r for r in records if r.get("type") == "link")

    if not all_links:
        print("No link reports found in input files.")
        return

    all_links.sort(key=lambda r: r["t"])
    t_min = all_links[0]["t"]
    t_max = all_links[-1]["t"]

    # Group into time frames
    frames: dict[int, list[dict]] = defaultdict(list)
    for r in all_links:
        frame_idx = int(r["t"] / REPLAY_FRAME_WINDOW)
        frames[frame_idx].append(r)

    # Build detector with a mutable link_states dict
    link_states: dict[str, dict] = {}

    def link_states_fn() -> dict[str, dict]:
        return link_states

    detector = ZoneDetector(link_states_fn)

    # Detect expected zone from file label
    expected_zone: str | None = None
    if file_label and file_label.startswith("occupied_q"):
        expected_zone = "Q" + file_label[-1].upper()

    zone_counts: dict[str, int] = defaultdict(int)
    total_frames = 0
    correct_frames = 0

    print(f"\nReplay: {len(all_links)} link reports, "
          f"t=[{t_min:.2f}, {t_max:.2f}], "
          f"{len(frames)} frames (window={REPLAY_FRAME_WINDOW}s)\n")

    frame_indices = sorted(frames.keys())
    for fidx in frame_indices:
        frame_records = frames[fidx]
        t = fidx * REPLAY_FRAME_WINDOW

        # Build link_states for this frame
        # Links present in this frame: window_full=True, max variance
        frame_links: dict[str, float] = {}
        for r in frame_records:
            lid = r["link"]
            v = r["variance"]
            if lid not in frame_links or v > frame_links[lid]:
                frame_links[lid] = v

        link_states.clear()
        # Links present in this frame get window_full=True
        for lid, v in frame_links.items():
            link_states[lid] = {"variance": v, "window_full": True}
        # Links in weight tables but absent from frame: window_full=False
        for lid in all_weight_links:
            if lid not in link_states:
                link_states[lid] = {"variance": 0.0, "window_full": False}

        result = detector.estimate()
        zone_str = result.zone.value if result.zone else "NONE"
        total_frames += 1
        zone_counts[zone_str] += 1

        if expected_zone and zone_str == expected_zone:
            correct_frames += 1

        # Build active link annotations
        active_parts: list[str] = []
        for lid in sorted(frame_links):
            if lid in LINK_ZONE_WEIGHTS:
                active_parts.append(f"{lid}*")
            elif lid in LINK_ABSORPTION_WEIGHTS:
                active_parts.append(f"{lid}v")
            else:
                active_parts.append(lid)
        # Check absorption on absent links
        for lid in sorted(all_weight_links - set(frame_links)):
            if lid in LINK_ABSORPTION_WEIGHTS:
                active_parts.append(f"{lid}v")

        scores_str = " ".join(
            f"{z.value}:{s:.1f}" for z, s in result.scores.items()
        )

        print(f"t={t:.2f}  zone={zone_str}  conf={result.confidence:.2f}  "
              f"scores={{{scores_str}}}  active=[{','.join(active_parts)}]")

    # Summary
    print(f"\n{'='*60}")
    print(f"  REPLAY SUMMARY")
    print(f"{'='*60}")
    print(f"  Frames: {total_frames}")
    for z in sorted(zone_counts):
        pct = 100 * zone_counts[z] / max(total_frames, 1)
        print(f"    {z}: {zone_counts[z]} ({pct:.0f}%)")
    if expected_zone:
        accuracy = 100 * correct_frames / max(total_frames, 1)
        print(f"\n  Expected zone: {expected_zone}")
        print(f"  Accuracy: {correct_frames}/{total_frames} ({accuracy:.0f}%)")


# ── Weight generation mode ────────────────────────────────────────────

#: Links with empty-room mean variance above this are candidates for absorption.
ABSORPTION_FLOOR = 10.0

#: Minimum spike/absorption ratio to consider a link significant.
SIGNIFICANCE_RATIO = 2.0

#: Expected labels for the 5 weight-generation input files.
EXPECTED_LABELS = ["empty", "occupied_q1", "occupied_q2", "occupied_q3", "occupied_q4"]


def generate_weights(empty_path: str, q1_path: str, q2_path: str,
                     q3_path: str, q4_path: str, force: bool = False) -> None:
    """Compute optimal weight matrices from labeled capture files."""
    from collections import defaultdict

    paths = [empty_path, q1_path, q2_path, q3_path, q4_path]
    captures: dict[str, list[dict]] = {}

    # Load all 5 files with label validation
    for path_str, expected_label in zip(paths, EXPECTED_LABELS):
        path = Path(path_str)
        if not path.exists():
            print(f"  ERROR: {path} not found")
            return
        records = _load_jsonl(path)
        if not records:
            print(f"  ERROR: {path} is empty")
            return

        actual_label = records[0].get("label", "")
        if actual_label != expected_label:
            print(f"  WARNING: file {path} has label '{actual_label}' "
                  f"but expected '{expected_label}'. "
                  f"Pass --force to override.")
            if not force:
                return

        captures[expected_label] = records

    # Compute per-link mean variance (non-zero only, i.e. >= 0.1) for each capture
    def _link_means(records: list[dict]) -> dict[str, float]:
        by_link: dict[str, list[float]] = defaultdict(list)
        for r in records:
            if r.get("type") != "link":
                continue
            v = r["variance"]
            if v >= 0.1:
                by_link[r["link"]].append(v)
        return {
            lid: sum(vs) / len(vs) for lid, vs in by_link.items() if vs
        }

    empty_means = _link_means(captures["empty"])
    quadrant_means: dict[str, dict[str, float]] = {}
    for qi in range(1, 5):
        quadrant_means[f"Q{qi}"] = _link_means(captures[f"occupied_q{qi}"])

    all_links = set(empty_means)
    for qm in quadrant_means.values():
        all_links |= set(qm)

    # Compute spike and absorption ratios
    spike_weights: dict[str, dict[str, float]] = {}
    absorption_weights: dict[str, dict[str, float]] = {}

    for lid in sorted(all_links):
        empty_mean = empty_means.get(lid)

        # Guard: no baseline data
        if empty_mean is None or empty_mean < 0.1:
            print(f"  Link {lid}: no baseline data in empty capture, skipping")
            continue

        # Spike ratios
        spike_ratios: dict[str, float] = {}
        for qname, qm in quadrant_means.items():
            occ_mean = qm.get(lid)
            if occ_mean is not None:
                spike_ratios[qname] = occ_mean / empty_mean
            else:
                spike_ratios[qname] = 0.0

        max_spike = max(spike_ratios.values()) if spike_ratios else 0.0
        if max_spike > SIGNIFICANCE_RATIO:
            spike_weights[lid] = {}
            for qname, ratio in spike_ratios.items():
                if ratio > SIGNIFICANCE_RATIO:
                    spike_weights[lid][qname] = round(
                        min(ratio / max_spike, 1.0), 2)
                else:
                    spike_weights[lid][qname] = 0.0

        # Absorption ratios (only high-baseline links)
        if empty_mean > ABSORPTION_FLOOR:
            abs_ratios: dict[str, float] = {}
            for qname, qm in quadrant_means.items():
                occ_mean = qm.get(lid)
                if occ_mean is not None and occ_mean > 0.0:
                    abs_ratios[qname] = empty_mean / occ_mean
                else:
                    abs_ratios[qname] = 0.0

            max_abs = max(abs_ratios.values()) if abs_ratios else 0.0
            if max_abs > SIGNIFICANCE_RATIO:
                absorption_weights[lid] = {}
                for qname, ratio in abs_ratios.items():
                    if ratio > SIGNIFICANCE_RATIO:
                        absorption_weights[lid][qname] = round(
                            min(ratio / max_abs, 1.0), 2)
                    else:
                        absorption_weights[lid][qname] = 0.0

    # Print results as copy-pasteable Python dicts
    print(f"\n{'='*60}")
    print(f"  COMPUTED WEIGHT MATRICES")
    print(f"{'='*60}")

    print("\nLINK_ZONE_WEIGHTS = {")
    for lid in sorted(spike_weights):
        w = spike_weights[lid]
        parts = ", ".join(f"Zone.{q}: {w.get(q, 0.0)}" for q in ["Q1", "Q2", "Q3", "Q4"])
        print(f'    "{lid}": {{{parts}}},')
    print("}")

    print("\nLINK_ABSORPTION_WEIGHTS = {")
    for lid in sorted(absorption_weights):
        w = absorption_weights[lid]
        parts = ", ".join(f"Zone.{q}: {w.get(q, 0.0)}" for q in ["Q1", "Q2", "Q3", "Q4"])
        print(f'    "{lid}": {{{parts}}},')
    print("}")

    # Comparison table: old vs new
    from python.zone_detector import (
        LINK_ZONE_WEIGHTS as OLD_SPIKE,
        LINK_ABSORPTION_WEIGHTS as OLD_ABS,
        Zone,
    )

    print(f"\n{'='*60}")
    print(f"  COMPARISON: Old vs New")
    print(f"{'='*60}")
    print(f"\n  Spike weights (LINK_ZONE_WEIGHTS):")
    print(f"  {'Link':<6} {'Zone':<4} {'Old':>6} {'New':>6} {'Delta':>7}")
    all_spike_links = sorted(set(OLD_SPIKE) | set(spike_weights))
    for lid in all_spike_links:
        for z in Zone:
            old_val = OLD_SPIKE.get(lid, {}).get(z, 0.0)
            new_val = spike_weights.get(lid, {}).get(z.value, 0.0)
            delta = new_val - old_val
            if old_val != 0.0 or new_val != 0.0:
                print(f"  {lid:<6} {z.value:<4} {old_val:>6.2f} {new_val:>6.2f} {delta:>+7.2f}")

    print(f"\n  Absorption weights (LINK_ABSORPTION_WEIGHTS):")
    print(f"  {'Link':<6} {'Zone':<4} {'Old':>6} {'New':>6} {'Delta':>7}")
    all_abs_links = sorted(set(OLD_ABS) | set(absorption_weights))
    for lid in all_abs_links:
        for z in Zone:
            old_val = OLD_ABS.get(lid, {}).get(z, 0.0)
            new_val = absorption_weights.get(lid, {}).get(z.value, 0.0)
            delta = new_val - old_val
            if old_val != 0.0 or new_val != 0.0:
                print(f"  {lid:<6} {z.value:<4} {old_val:>6.2f} {new_val:>6.2f} {delta:>+7.2f}")


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSI capture & analysis")
    parser.add_argument("--port", default="COM9")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--label", default="test",
                        help="Label for this capture (e.g. 'empty', 'occupied_q3')")
    parser.add_argument("--analyze", nargs="+", metavar="FILE",
                        help="Analyze existing capture files instead of capturing")
    parser.add_argument("--replay", nargs="+", metavar="FILE",
                        help="Replay captured data through ZoneDetector")
    parser.add_argument("--weights", nargs=5,
                        metavar=("EMPTY", "Q1", "Q2", "Q3", "Q4"),
                        help="Generate weight matrices from 5 labeled captures")
    parser.add_argument("--force", action="store_true",
                        help="Override label mismatch warnings in --weights mode")
    parser.add_argument("--delay", type=float, default=0,
                        help="Countdown delay before capture starts (seconds)")
    args = parser.parse_args()

    if args.analyze:
        analyze(*args.analyze)
    elif args.replay:
        replay(*args.replay)
    elif args.weights:
        generate_weights(*args.weights, force=args.force)
    else:
        capture(args.port, args.baud, args.seconds, args.label,
                delay=args.delay)
