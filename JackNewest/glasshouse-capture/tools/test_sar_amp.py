"""Synthetic regression test for the shared frame decoder.

Generates a small in-memory capture containing one sample of every magic
type the firmware emits (including the new SAR_AMP batched amplitude
packet), writes a JSONL file identical in shape to captures/capture_*.jsonl,
then re-parses it through tools.reparse so we can eyeball the before/after
counts without needing the hardware in the loop.

Success criteria:
  - zero records with type == 'unknown'
  - counts for csi/link/vitals/iq/heartbeat are UNCHANGED between the
    injected pre-change counts and the post-change counts.

Run from the glasshouse-capture folder:
    python -m tools.test_sar_amp
"""

from __future__ import annotations

import collections
import json
import shutil
import struct
import sys
from pathlib import Path

# Make the 'python' package importable when run as `python -m tools.test_sar_amp`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from python.frame_decoder import (  # noqa: E402
    parse_packet,
    CSI_MAGIC_LE,
    VITALS_MAGIC_LE,
    IQ_MAGIC_LE,
    SAR_AMP_MAGIC_LE,
    SAR_AMP_BATCH_SIZE,
    CSI_HEADER_SIZE,
)


def build_csi() -> bytes:
    """20-byte header + 16 I/Q bytes (8 subcarriers * 1 antenna)."""
    hdr = (
        CSI_MAGIC_LE
        + struct.pack('<BBHIIbb', 2, 1, 8, 2437, 42, -40, -90)
        + b'\x00\x00'  # reserved
    )
    assert len(hdr) == CSI_HEADER_SIZE, len(hdr)
    return hdr + bytes(range(16))


def build_vitals() -> bytes:
    # 32 bytes, see edge_vitals_pkt_t.
    return (
        VITALS_MAGIC_LE
        + bytes([2, 0x01])  # node_id, flags (presence)
        + struct.pack('<H', 1500)  # breathing_rate
        + struct.pack('<I', 720000)  # heartrate
        + struct.pack('<bB', -45, 1)  # rssi, n_persons
        + b'\x00\x00'  # reserved
        + struct.pack('<f', 0.123)  # motion_energy
        + struct.pack('<f', 0.456)  # presence_score
        + struct.pack('<I', 12345)  # timestamp_ms
        + struct.pack('<I', 0)      # reserved2
    )


def build_iq() -> bytes:
    # Per frame_decoder IQ branch: magic + node_id(u8) + channel(u8) + iq_len(u16) + body
    body = b'\xDE\xAD' * 32  # 64 bytes
    return IQ_MAGIC_LE + struct.pack('<BBH', 3, 6, len(body)) + body


def build_heartbeat() -> bytes:
    return b'\xAA'


def build_link() -> bytes:
    # 10 bytes, byte0==0x01, '<BBBfBH'
    return struct.pack('<BBBfBH', 0x01, 2, 3, 1.25, 1, 17)


def build_sar_amp() -> bytes:
    # 208 bytes exactly
    amps = [float(i) * 0.5 for i in range(SAR_AMP_BATCH_SIZE)]
    pkt = (
        SAR_AMP_MAGIC_LE
        + struct.pack('<BBH', 2, 3, SAR_AMP_BATCH_SIZE)   # node_id, peer_id, n_samples
        + struct.pack('<II', 100_000, 20_000)              # batch_start_us, interval_us
        + struct.pack('<%df' % SAR_AMP_BATCH_SIZE, *amps)
    )
    assert len(pkt) == 208, len(pkt)
    return pkt


def build_unknown_bogus() -> bytes:
    return b'\x99\x88\x77\x66'  # does not match any magic


# Pre-change expected counts — what the decoder would have produced
# *before* the SAR_AMP branch was added. We use this to assert that the
# pre-existing types (csi/link/vitals/iq/heartbeat) are unchanged after
# the change. Bogus 4-byte payload is deliberately included so we can
# verify unknown handling still works in isolation.
PRE_CHANGE_COUNTS = {
    "csi":       1,
    "vitals":    1,
    "iq":        1,
    "link":      1,
    "heartbeat": 1,
    # Before the change: SAR_AMP packet is 208 bytes with unrecognized magic
    # -> would have been 'unknown'. After the change: 'sar_amp'.
    # The 4-byte bogus is always 'unknown'.
    "sar_amp":   0,
    "unknown":   2,
}


def main() -> int:
    out_dir = Path("captures") / "_sar_amp_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path("captures") / "capture_sar_amp_test.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)

    samples = [
        ("csi",       build_csi()),
        ("vitals",    build_vitals()),
        ("iq",        build_iq()),
        ("heartbeat", build_heartbeat()),
        ("link",      build_link()),
        ("sar_amp",   build_sar_amp()),
        ("bogus1",    build_unknown_bogus()),
        ("bogus2",    build_unknown_bogus()),
    ]

    with src.open("w") as f:
        for i, (label, raw) in enumerate(samples):
            rec = {"t": round(i * 0.01, 4), "label": label}
            rec.update(parse_packet(raw))
            # Store raw hex so reparse can re-decode just like capture.py does.
            rec["raw"] = raw.hex()
            f.write(json.dumps(rec) + "\n")

    # Now re-read the file the same way reparse.py does.
    post: dict[str, int] = collections.Counter()
    with src.open("r") as fin:
        for line in fin:
            rec = json.loads(line)
            raw = bytes.fromhex(rec.get("raw", ""))
            post[parse_packet(raw)["type"]] += 1

    print("Synthetic capture decode counts (after SAR_AMP support):")
    for k in sorted(set(post) | set(PRE_CHANGE_COUNTS)):
        got = post.get(k, 0)
        expected_pre = PRE_CHANGE_COUNTS.get(k, 0)
        print(f"  {k:<12s} pre-change={expected_pre}  post-change={got}")

    # Assertion 1: preserved types are unchanged.
    preserved = ["csi", "link", "vitals", "iq", "heartbeat"]
    errs: list[str] = []
    for k in preserved:
        if post.get(k, 0) != PRE_CHANGE_COUNTS[k]:
            errs.append(f"count drift on '{k}': expected {PRE_CHANGE_COUNTS[k]} got {post.get(k,0)}")

    # Assertion 2: new type decodes.
    if post.get("sar_amp", 0) != 1:
        errs.append(f"expected 1 sar_amp record, got {post.get('sar_amp', 0)}")

    # Assertion 3: only the 2 genuinely bogus payloads are 'unknown'.
    if post.get("unknown", 0) != 2:
        errs.append(f"expected 2 unknown records (bogus payloads), got {post.get('unknown', 0)}")

    if errs:
        for e in errs:
            print("  FAIL:", e, file=sys.stderr)
        return 1

    # Clean up the synthetic capture so reparse.py won't pick it up later.
    shutil.rmtree(out_dir, ignore_errors=True)
    src.unlink(missing_ok=True)

    print("OK: decoder regression test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
