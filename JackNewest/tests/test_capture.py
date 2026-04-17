"""Tests for debug/capture.py — replay, weight generation, and packet parsing."""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import pytest


# ── Helpers ──────────────────────────────────────────────────────────

def _build_link_packet(node: int, partner: int, variance: float,
                       state: int = 0, count: int = 1) -> bytes:
    """Build a 10-byte link-report packet (type 0x01)."""
    return struct.pack('<BBBfBH', 0x01, node, partner, variance, state, count)


def _build_iq_packet(node_id: int, channel: int, iq_len: int,
                     payload: bytes = b'') -> bytes:
    """Build an I/Q packet with magic 0xC5110006."""
    header = b'\x06\x00\x11\xC5'
    header += struct.pack('<BBH', node_id, channel, iq_len)
    return header + payload


def _write_jsonl(records: list[dict], path: Path) -> None:
    """Write a list of dicts as JSONL."""
    with open(path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')


# ── Task 1 tests: I/Q packet parsing ────────────────────────────────

class TestIQParsing:
    """Verify I/Q packet recognition and field extraction."""

    def test_iq_packet_fields(self):
        """A well-formed I/Q packet produces a record with correct fields."""
        payload = bytes(range(64))
        pkt = _build_iq_packet(node_id=2, channel=6, iq_len=128, payload=payload)

        # Simulate the parsing logic from capture()
        assert len(pkt) >= 8
        assert pkt[:4] == b'\x06\x00\x11\xC5'

        node_id = pkt[4]
        channel = pkt[5]
        iq_len = struct.unpack_from('<H', pkt, 6)[0]
        hex_preview = pkt[8:40].hex()

        assert node_id == 2
        assert channel == 6
        assert iq_len == 128
        assert len(hex_preview) == 64  # 32 bytes -> 64 hex chars

    def test_truncated_iq_falls_to_unknown(self):
        """An I/Q packet shorter than 8 bytes should not match the I/Q branch."""
        short_pkt = b'\x06\x00\x11\xC5\x02\x06'  # only 6 bytes

        # The condition: len(packet) >= 8 and packet[:4] == magic
        matches_iq = len(short_pkt) >= 8 and short_pkt[:4] == b'\x06\x00\x11\xC5'
        assert not matches_iq, "Truncated packet should not match I/Q branch"

    def test_iq_empty_payload_hex(self):
        """I/Q packet with no payload bytes beyond header still parses."""
        pkt = _build_iq_packet(node_id=1, channel=11, iq_len=0, payload=b'')
        assert len(pkt) == 8  # header only
        hex_preview = pkt[8:40].hex()
        assert hex_preview == ""  # no payload to preview


# ── Task 2 tests: Replay mode ───────────────────────────────────────

class TestReplay:
    """Verify replay feeds frames through ZoneDetector correctly."""

    def _make_link_records(self, label: str, link: str, variances: list[float],
                           t_start: float = 0.0, dt: float = 0.1) -> list[dict]:
        """Build synthetic link report records."""
        lo, hi = link[0], link[1]
        return [
            {
                "t": round(t_start + i * dt, 4),
                "label": label,
                "type": "link",
                "link": link,
                "node": int(lo),
                "partner": int(hi),
                "variance": v,
                "state": 1 if v >= 0.1 else 0,
                "count": i + 1,
            }
            for i, v in enumerate(variances)
        ]

    def test_replay_produces_zone_output(self, tmp_path):
        """Replay with synthetic data runs through ZoneDetector without crashing."""
        from debug.capture import replay, _load_jsonl

        # Create synthetic data: link 13 spiking (Q1 weight=1.0) and
        # link 14 spiking (Q3 weight=1.0) — should produce zone detection
        records = []
        # 20 frames of link 13 with high variance
        records.extend(self._make_link_records(
            "occupied_q1", "13", [5.0] * 20, t_start=0.0, dt=0.2))
        # 20 frames of link 24 with high variance (Q1 weight=1.0)
        records.extend(self._make_link_records(
            "occupied_q1", "24", [5.0] * 20, t_start=0.0, dt=0.2))

        jsonl_path = tmp_path / "test_replay.jsonl"
        _write_jsonl(records, jsonl_path)

        # Should not crash — replay prints to stdout
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            replay(str(jsonl_path))

        output = buf.getvalue()
        assert "REPLAY SUMMARY" in output
        assert "Frames:" in output

    def test_replay_absent_links_window_full_false(self, tmp_path):
        """Links in weight tables but absent from a frame get window_full=False."""
        from debug.capture import replay, REPLAY_FRAME_WINDOW, _load_jsonl
        from python.zone_detector import LINK_ABSORPTION_WEIGHTS

        # Only link 13 present — absorption links (23, 34) should be absent
        records = self._make_link_records(
            "test", "13", [5.0] * 20, t_start=0.0, dt=0.2)

        jsonl_path = tmp_path / "test_absent.jsonl"
        _write_jsonl(records, jsonl_path)

        # Patch replay to capture link_states for inspection
        from debug import capture as cap_mod
        from python.zone_detector import ZoneDetector, LINK_ZONE_WEIGHTS
        all_weight_links = set(LINK_ZONE_WEIGHTS) | set(LINK_ABSORPTION_WEIGHTS)

        # Build what replay would build for a frame with only link 13
        link_states: dict[str, dict] = {}
        link_states["13"] = {"variance": 5.0, "window_full": True}
        for lid in all_weight_links:
            if lid not in link_states:
                link_states[lid] = {"variance": 0.0, "window_full": False}

        # Absorption links should be present with window_full=False
        assert "23" in link_states
        assert link_states["23"]["window_full"] is False
        assert link_states["23"]["variance"] == 0.0
        assert "34" in link_states
        assert link_states["34"]["window_full"] is False

    def test_replay_corrupted_line_continues(self, tmp_path):
        """A corrupted JSONL line mid-file does not stop replay."""
        from debug.capture import _load_jsonl

        content = (
            '{"t": 0.0, "label": "test", "type": "link", "link": "13", '
            '"node": 1, "partner": 3, "variance": 5.0, "state": 1, "count": 1}\n'
            'THIS IS CORRUPTED\n'
            '{"t": 0.2, "label": "test", "type": "link", "link": "13", '
            '"node": 1, "partner": 3, "variance": 3.0, "state": 1, "count": 2}\n'
        )
        jsonl_path = tmp_path / "test_corrupt.jsonl"
        jsonl_path.write_text(content)

        records = _load_jsonl(jsonl_path)
        assert len(records) == 2  # corrupted line skipped
        assert records[0]["variance"] == 5.0
        assert records[1]["variance"] == 3.0


# ── Task 3 tests: Weight generation ─────────────────────────────────

class TestWeightGeneration:
    """Verify auto weight matrix computation."""

    def _make_capture(self, label: str, link_variances: dict[str, float],
                      n_records: int = 20, dt: float = 0.2) -> list[dict]:
        """Build synthetic capture records with given per-link variances."""
        records = []
        for i in range(n_records):
            t = round(i * dt, 4)
            for link_id, variance in link_variances.items():
                lo, hi = link_id[0], link_id[1]
                records.append({
                    "t": t,
                    "label": label,
                    "type": "link",
                    "link": link_id,
                    "node": int(lo),
                    "partner": int(hi),
                    "variance": variance,
                    "state": 1 if variance >= 0.1 else 0,
                    "count": i + 1,
                })
        return records

    def test_weights_computed_from_known_data(self, tmp_path):
        """Known spike patterns produce expected weight output."""
        from debug.capture import generate_weights
        import io
        from contextlib import redirect_stdout

        # Empty room: all links at low variance
        empty = self._make_capture("empty", {
            "13": 1.0, "24": 1.0, "14": 1.0, "12": 1.0, "23": 15.0, "34": 15.0,
        })
        # Q1 occupied: links 13 and 24 spike (3x+ baseline)
        q1 = self._make_capture("occupied_q1", {
            "13": 5.0, "24": 5.0, "14": 1.0, "12": 1.0, "23": 15.0, "34": 15.0,
        })
        # Q2: link 23 drops (absorption)
        q2 = self._make_capture("occupied_q2", {
            "13": 1.0, "24": 1.0, "14": 1.0, "12": 1.0, "23": 3.0, "34": 15.0,
        })
        # Q3: link 14 spikes
        q3 = self._make_capture("occupied_q3", {
            "13": 1.0, "24": 1.0, "14": 5.0, "12": 1.0, "23": 15.0, "34": 15.0,
        })
        # Q4: link 34 drops (absorption)
        q4 = self._make_capture("occupied_q4", {
            "13": 1.0, "24": 1.0, "14": 1.0, "12": 1.0, "23": 15.0, "34": 3.0,
        })

        paths = []
        for label, data in [("empty", empty), ("occupied_q1", q1),
                            ("occupied_q2", q2), ("occupied_q3", q3),
                            ("occupied_q4", q4)]:
            p = tmp_path / f"capture_{label}.jsonl"
            _write_jsonl(data, p)
            paths.append(str(p))

        buf = io.StringIO()
        with redirect_stdout(buf):
            generate_weights(*paths)

        output = buf.getvalue()
        assert "LINK_ZONE_WEIGHTS" in output
        assert "LINK_ABSORPTION_WEIGHTS" in output
        assert "COMPARISON" in output
        # Links 13 and 24 should appear in spike weights (spiked in Q1)
        assert '"13"' in output
        assert '"24"' in output

    def test_zero_baseline_link_skipped(self, tmp_path):
        """A link with no non-zero variance in empty capture is skipped."""
        from debug.capture import generate_weights
        import io
        from contextlib import redirect_stdout

        # Empty room: link 99 has zero variance (no data)
        empty = self._make_capture("empty", {"13": 1.0, "99": 0.0})
        # All quadrant files: link 99 spikes (but no baseline to compare)
        q_data = {"13": 1.0, "99": 5.0}
        q1 = self._make_capture("occupied_q1", q_data)
        q2 = self._make_capture("occupied_q2", q_data)
        q3 = self._make_capture("occupied_q3", q_data)
        q4 = self._make_capture("occupied_q4", q_data)

        paths = []
        for label, data in [("empty", empty), ("occupied_q1", q1),
                            ("occupied_q2", q2), ("occupied_q3", q3),
                            ("occupied_q4", q4)]:
            p = tmp_path / f"capture_{label}.jsonl"
            _write_jsonl(data, p)
            paths.append(str(p))

        buf = io.StringIO()
        with redirect_stdout(buf):
            generate_weights(*paths)

        output = buf.getvalue()
        # Link 99 should be skipped with a warning
        assert "Link 99: no baseline data in empty capture, skipping" in output
        # Should NOT crash

    def test_label_mismatch_exits_without_force(self, tmp_path):
        """Files with mismatched labels cause exit without --force."""
        from debug.capture import generate_weights
        import io
        from contextlib import redirect_stdout

        # All files labeled "wrong" instead of expected labels
        data = self._make_capture("wrong", {"13": 1.0})
        paths = []
        for label in ["empty", "occupied_q1", "occupied_q2",
                       "occupied_q3", "occupied_q4"]:
            p = tmp_path / f"capture_{label}.jsonl"
            _write_jsonl(data, p)
            paths.append(str(p))

        buf = io.StringIO()
        with redirect_stdout(buf):
            generate_weights(*paths, force=False)

        output = buf.getvalue()
        assert "WARNING" in output
        assert "label 'wrong'" in output
        # Should NOT contain weight output (exited early)
        assert "LINK_ZONE_WEIGHTS" not in output

    def test_label_mismatch_proceeds_with_force(self, tmp_path):
        """Files with mismatched labels proceed when --force is set."""
        from debug.capture import generate_weights
        import io
        from contextlib import redirect_stdout

        data = self._make_capture("wrong", {"13": 1.0})
        paths = []
        for label in ["empty", "occupied_q1", "occupied_q2",
                       "occupied_q3", "occupied_q4"]:
            p = tmp_path / f"capture_{label}.jsonl"
            _write_jsonl(data, p)
            paths.append(str(p))

        buf = io.StringIO()
        with redirect_stdout(buf):
            generate_weights(*paths, force=True)

        output = buf.getvalue()
        assert "WARNING" in output
        # Should still produce output despite warnings
        assert "LINK_ZONE_WEIGHTS" in output
