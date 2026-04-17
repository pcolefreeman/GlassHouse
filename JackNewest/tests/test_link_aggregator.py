import struct
import time

from python.link_aggregator import LinkAggregator


def _make_link_report(node_id: int, partner_id: int, variance: float,
                       state: int, sample_count: int) -> bytes:
    """Build a 10-byte link report packet."""
    return struct.pack('<BBBfBH', 0x01, node_id, partner_id, variance, state, sample_count)


def test_parse_single_link_report():
    agg = LinkAggregator()
    pkt = _make_link_report(1, 3, 0.012, 1, 20)
    agg.feed(pkt)
    states = agg.get_link_states()
    assert "13" in states
    assert abs(states["13"]["variance"] - 0.012) < 1e-6
    assert states["13"]["state"] == "MOTION"
    assert states["13"]["window_full"] is True


def test_normalize_link_id_order():
    """Link from node 3->1 normalizes to '13'."""
    agg = LinkAggregator()
    pkt = _make_link_report(3, 1, 0.005, 0, 15)
    agg.feed(pkt)
    states = agg.get_link_states()
    assert "13" in states
    assert "31" not in states


def test_bidirectional_averaging():
    """Both directions of a link use max variance (not mean).

    Directional asymmetry can be extreme (e.g. link 14: 71.0 vs 0.05).
    Max preserves the informative direction; mean dilutes it with noise.
    """
    agg = LinkAggregator()
    agg.feed(_make_link_report(1, 3, 0.010, 1, 20))
    agg.feed(_make_link_report(3, 1, 0.014, 1, 20))
    states = agg.get_link_states()
    assert abs(states["13"]["variance"] - 0.014) < 1e-6


def _make_vitals(flags: int = 0x04, motion_energy: float = 0.0) -> bytes:
    """Build a 32-byte vitals packet with given flags and motion_energy at offset 16."""
    header = b'\x02\x00\x11\xC5' + b'\x00' + bytes([flags])
    pad_before_energy = b'\x00' * 10  # bytes 6..15
    energy_bytes = struct.pack('<f', motion_energy)
    pad_after = b'\x00' * (32 - 6 - 10 - 4)  # fill to 32
    return header + pad_before_energy + energy_bytes + pad_after


def test_vitals_packet_sets_occupancy():
    """Vitals with high motion energy sets occupied."""
    agg = LinkAggregator()
    vitals = _make_vitals(flags=0x04, motion_energy=8.0)
    assert len(vitals) == 32
    agg.feed(vitals)
    assert agg.is_occupied() is True
    assert agg.has_vitals_update() is True
    assert agg.has_vitals_update() is False  # consumed


def test_vitals_fallback_to_motion_bit():
    """When motion_energy is zero, fall back to motion bit."""
    agg = LinkAggregator()
    vitals = _make_vitals(flags=0x04, motion_energy=0.0)
    agg.feed(vitals)
    # motion_energy=0 → fallback to motion bit (0x04) → occupied
    assert agg.is_occupied() is True


def test_vitals_motion_energy_parsed():
    """Motion energy float is extracted from vitals packet."""
    agg = LinkAggregator()
    vitals = _make_vitals(flags=0x04, motion_energy=9.27)
    agg.feed(vitals)
    assert abs(agg.motion_energy - 9.27) < 0.01


def test_energy_threshold_fixed():
    """Motion energy above fixed threshold sets occupied."""
    agg = LinkAggregator()
    # Low energy → not occupied
    agg.feed(_make_vitals(flags=0x04, motion_energy=0.3))
    assert agg.is_occupied() is False

    # High energy → occupied
    agg.feed(_make_vitals(flags=0x04, motion_energy=8.0))
    assert agg.is_occupied() is True


def test_stale_link_marked_not_full(monkeypatch):
    """Link with no reports in 2s gets window_full=False."""
    agg = LinkAggregator()
    agg.feed(_make_link_report(1, 3, 0.010, 1, 20))
    # Simulate time passing
    original_monotonic = time.monotonic
    monkeypatch.setattr(time, 'monotonic', lambda: original_monotonic() + 3.0)
    states = agg.get_link_states()
    assert states["13"]["window_full"] is False


def test_demux_ignores_unknown_packets():
    """Unknown packet types are silently dropped."""
    agg = LinkAggregator()
    agg.feed(b'\xFF\x00\x00\x00')
    assert agg.get_link_states() == {}


def test_truncated_vitals_rejected():
    """Short packet with vitals magic prefix is NOT parsed as vitals."""
    agg = LinkAggregator()
    # 12 bytes: has magic prefix but too short (spec requires 32)
    truncated = b'\x02\x00\x11\xC5' + b'\x00' + b'\x01' + b'\x00' * 6
    assert len(truncated) == 12
    agg.feed(truncated)
    assert agg.is_occupied() is False
    assert agg.has_vitals_update() is False


def test_multiple_links():
    """Reports from different links appear as separate entries."""
    agg = LinkAggregator()
    agg.feed(_make_link_report(1, 2, 0.005, 0, 10))
    agg.feed(_make_link_report(1, 3, 0.012, 1, 20))
    agg.feed(_make_link_report(2, 3, 0.008, 1, 15))
    states = agg.get_link_states()
    assert set(states.keys()) == {"12", "13", "23"}
