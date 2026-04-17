"""Re-parse existing captures/*.jsonl through the new shared frame decoder.

Reads each original record, pulls the untouched `raw` hex bytes (captured
verbatim by capture.py), runs it through python/frame_decoder.parse_packet,
and writes a richer record to captures_v2/<same-filename>.jsonl.

Also prints a before/after type-distribution table so the parser improvement
is mechanically visible.

Run from the glasshouse-capture folder:
    python -m tools.reparse
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

# Make 'python' importable whether run as `python -m tools.reparse` or directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from python.frame_decoder import parse_packet  # noqa: E402


CAPTURES_IN  = Path("captures")
CAPTURES_OUT = Path("captures_v2")


def reparse_file(src: Path, dst: Path) -> tuple[dict[str, int], dict[str, int]]:
    before: dict[str, int] = collections.Counter()
    after:  dict[str, int] = collections.Counter()

    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r") as fin, dst.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                orig = json.loads(line)
            except json.JSONDecodeError:
                continue

            before[orig.get("type", "?")] += 1

            raw_hex = orig.get("raw", "")
            try:
                packet = bytes.fromhex(raw_hex)
            except ValueError:
                # Preserve the original record but flag the corruption.
                new_rec = dict(orig)
                new_rec["type"] = "bad_hex"
                after["bad_hex"] += 1
                fout.write(json.dumps(new_rec) + "\n")
                continue

            new_rec = {"t": orig.get("t"), "label": orig.get("label")}
            new_rec.update(parse_packet(packet))
            after[new_rec["type"]] += 1
            fout.write(json.dumps(new_rec) + "\n")

    return dict(before), dict(after)


def print_diff(name: str, before: dict[str, int], after: dict[str, int]) -> None:
    keys = sorted(set(before) | set(after))
    total = sum(after.values())
    print(f"\n=== {name}  (total={total}) ===")
    print(f"  {'type':<14s} {'before':>8s}  {'after':>8s}  {'delta':>8s}")
    for k in keys:
        b = before.get(k, 0)
        a = after.get(k, 0)
        delta = a - b
        mark = "   "
        if delta > 0:
            mark = " + "
        elif delta < 0:
            mark = " - "
        print(f"  {k:<14s} {b:>8d}  {a:>8d}  {mark}{abs(delta):>6d}")


def main() -> int:
    if not CAPTURES_IN.exists():
        print(f"error: {CAPTURES_IN} not found (run from glasshouse-capture/ dir)", file=sys.stderr)
        return 2

    grand_before: dict[str, int] = collections.Counter()
    grand_after:  dict[str, int] = collections.Counter()

    for src in sorted(CAPTURES_IN.glob("capture_*.jsonl")):
        dst = CAPTURES_OUT / src.name
        before, after = reparse_file(src, dst)
        print_diff(src.name, before, after)
        grand_before.update(before)
        grand_after.update(after)

    print_diff("GRAND TOTAL", dict(grand_before), dict(grand_after))
    print(f"\nRe-parsed outputs written to: {CAPTURES_OUT.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
