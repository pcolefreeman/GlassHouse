"""Aggregates link summary reports from perimeter nodes into a state dict."""

from __future__ import annotations

import struct
import time


def _normalize_link_id(a: int, b: int) -> str:
    """Link ID with smaller node first: nodes 3,1 -> '13'."""
    lo, hi = min(a, b), max(a, b)
    return f"{lo}{hi}"


class LinkAggregator:
    """Collects 10-byte link reports and 32-byte vitals packets.

    Produces a ``get_link_states()`` dict compatible with ZoneDetector.
    """

    _REPORT_TYPE = 0x01
    _VITALS_MAGIC = b'\x02\x00\x11\xC5'  # 0xC5110002 little-endian on ESP32
    _IQ_MAGIC = b'\x06\x00\x11\xC5'      # 0xC5110006 little-endian
    _STALENESS_SEC = 2.0
    _ENERGY_THRESHOLD = 5.0    # Fixed: empty-room ~0.3-3.3, occupied ~7+ (no calibration)

    def __init__(self) -> None:
        # Per-link: list of (variance, state, timestamp)
        self._link_data: dict[str, list[tuple[float, str, float]]] = {}
        self._occupied: bool = False
        self._vitals_updated: bool = False
        self._links_dirty: bool = False
        self._motion_energy: float = 0.0
        self._latest_iq: list[tuple[int, int, bytes]] = []  # (node_id, channel, iq_bytes)

    def feed(self, packet: bytes) -> None:
        """Ingest a raw decoded packet (link report or vitals)."""
        if len(packet) < 1:
            return
        if packet[0] == self._REPORT_TYPE and len(packet) == 10:
            self._parse_link_report(packet)
        elif len(packet) >= 4 and packet[:4] == self._VITALS_MAGIC:
            self._parse_vitals(packet)
        elif len(packet) >= 8 and packet[:4] == self._IQ_MAGIC:
            self._parse_iq(packet)

    def _parse_link_report(self, data: bytes) -> None:
        _, node_id, partner_id, variance, state_byte, sample_count = struct.unpack(
            '<BBBfBH', data[:10]
        )
        if not (1 <= node_id <= 4 and 1 <= partner_id <= 4 and node_id != partner_id):
            return  # Malformed report
        link_id = _normalize_link_id(node_id, partner_id)
        state_str = "MOTION" if state_byte == 1 else "IDLE"
        now = time.monotonic()

        if link_id not in self._link_data:
            self._link_data[link_id] = []
        self._link_data[link_id].append((variance, state_str, now))

        # Keep reports within a 1-second averaging window.
        # Link reporter fires every 200ms, so this holds ~5 reports per
        # link — enough for stable variance even with sporadic delivery.
        cutoff = now - 1.0
        self._link_data[link_id] = [
            (v, s, t) for v, s, t in self._link_data[link_id] if t >= cutoff
        ]
        self._links_dirty = True

    def _parse_iq(self, packet: bytes) -> None:
        """Parse an I/Q CSI packet (magic 0xC5110006)."""
        node_id = packet[4]
        if not (1 <= node_id <= 4):
            return
        channel = packet[5]
        iq_len = int.from_bytes(packet[6:8], 'little')
        if iq_len == 0:
            return
        if iq_len > 256 or len(packet) < 8 + iq_len:
            return
        if len(self._latest_iq) >= 200:
            return  # backpressure cap
        self._latest_iq.append((node_id, channel, packet[8:8 + iq_len]))

    def drain_iq(self) -> list[tuple[int, int, bytes]]:
        """Return and clear queued I/Q frames."""
        iq = self._latest_iq
        self._latest_iq = []
        return iq

    def _parse_vitals(self, data: bytes) -> None:
        if len(data) >= 32:  # spec: vitals packet is 32 bytes
            flags = data[5]  # Byte 5: Bit0=presence, Bit1=fall, Bit2=motion
            # Parse motion_energy float at offset 16 (RuView DSP output)
            if len(data) >= 20:
                self._motion_energy = struct.unpack_from('<f', data, 16)[0]

            # Use fixed energy threshold if vitals carry energy;
            # fall back to motion bit (0x04) otherwise.
            if self._motion_energy > 0.0:
                self._occupied = self._motion_energy > self._ENERGY_THRESHOLD
            else:
                self._occupied = bool(flags & 0x04)
            self._vitals_updated = True

    def get_link_states(self) -> dict[str, dict]:
        """Return the current link state dict for ZoneDetector."""
        now = time.monotonic()
        result = {}
        for link_id, entries in self._link_data.items():
            # Filter stale entries
            fresh = [(v, s, t) for v, s, t in entries if (now - t) < self._STALENESS_SEC]
            if not fresh:
                result[link_id] = {
                    "variance": 0.0,
                    "state": "IDLE",
                    "window_full": False,
                }
                continue

            # Use max variance, not mean: each link has two directions
            # (node A→B and B→A) with potentially huge asymmetry (e.g.
            # link 14: direction 1→4 = 71.0, direction 4→1 = 0.05).
            # Averaging dilutes the informative direction with noise floor.
            # Max preserves the direction that carries real signal.
            avg_var = max(v for v, _, _ in fresh)
            # State is MOTION if any report in window says MOTION
            any_motion = any(s == "MOTION" for _, s, _ in fresh)
            result[link_id] = {
                "variance": avg_var,
                "state": "MOTION" if any_motion else "IDLE",
                "window_full": True,
            }
        return result

    def is_occupied(self) -> bool:
        # If vitals packets are available, use them.
        if self._occupied:
            return True
        # Fallback: infer occupancy from link motion states.
        # If any active link reports MOTION, the room is occupied.
        now = time.monotonic()
        for entries in self._link_data.values():
            fresh = [(v, s, t) for v, s, t in entries if (now - t) < self._STALENESS_SEC]
            if any(s == "MOTION" for _, s, _ in fresh):
                return True
        return False

    def has_vitals_update(self) -> bool:
        if self._vitals_updated:
            self._vitals_updated = False
            return True
        return False

    def links_updated(self) -> bool:
        if self._links_dirty:
            self._links_dirty = False
            return True
        return False

    @property
    def motion_energy(self) -> float:
        """Latest motion_energy from firmware vitals DSP."""
        return self._motion_energy
