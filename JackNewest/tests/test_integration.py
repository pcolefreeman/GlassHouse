"""End-to-end test: mock serial -> aggregator -> zone detector."""

import struct

from python.link_aggregator import LinkAggregator
from python.zone_detector import ZoneDetector, Zone, build_live_zone_tracker


def _make_link_report(node_id, partner_id, variance, state, count):
    return struct.pack('<BBBfBH', 0x01, node_id, partner_id, variance, state, count)


def _make_vitals(motion_flag, motion_energy=0.0):
    """Build a 32-byte vitals packet with motion flag and energy at offset 16."""
    flags = 0x04 if motion_flag else 0x00
    header = b'\x02\x00\x11\xC5' + b'\x00' + bytes([flags])
    pad_before = b'\x00' * 10  # bytes 6..15
    energy = struct.pack('<f', motion_energy)
    pad_after = b'\x00' * (32 - 6 - 10 - 4)
    return header + pad_before + energy + pad_after


def test_end_to_end_zone_detection():
    """Full pipeline: link reports -> aggregator -> zone detector -> stable zone."""
    aggregator = LinkAggregator()
    detector = ZoneDetector(link_states_fn=aggregator.get_link_states)
    tracker = build_live_zone_tracker()

    # Vitals with high motion energy → occupied
    aggregator.feed(_make_vitals(1, motion_energy=8.0))
    occupied = aggregator.is_occupied()
    assert occupied is True

    # Phase 1: build rolling baseline with realistic quiet reports
    for _ in range(10):  # let baselines settle
        for n, p, v in [(1, 4, 3.0), (1, 3, 0.5), (2, 3, 50.0),
                         (1, 2, 1.0), (2, 4, 0.5), (3, 4, 20.0)]:
            aggregator.feed(_make_link_report(n, p, v, 0, 20))
        detector.estimate()

    # Phase 2: spike links 14 and 13 (person in Q3)
    # Feed enough spikes to dominate the 1-second averaging window
    for _ in range(10):
        aggregator.feed(_make_link_report(1, 4, 30.0, 1, 20))
        aggregator.feed(_make_link_report(1, 3, 5.0, 1, 20))
    aggregator.feed(_make_link_report(2, 3, 50.0, 0, 20))
    aggregator.feed(_make_link_report(1, 2, 1.0, 0, 20))
    aggregator.feed(_make_link_report(2, 4, 0.5, 0, 20))
    aggregator.feed(_make_link_report(3, 4, 20.0, 0, 20))

    raw = detector.estimate()
    stable = tracker.update(raw, occupied=occupied)

    # Should detect a zone after warmup with 2+ spiked links
    assert stable.zone is not None


def test_empty_room_no_zone():
    """No motion -> zone=None."""
    aggregator = LinkAggregator()
    detector = ZoneDetector(link_states_fn=aggregator.get_link_states)
    tracker = build_live_zone_tracker()

    # Not occupied (low energy, no motion bit)
    aggregator.feed(_make_vitals(0, motion_energy=0.2))

    # Build rolling baseline with realistic quiet links
    for _ in range(10):
        for n1, n2, v in [(1, 2, 1.0), (1, 3, 0.5), (1, 4, 3.0),
                           (2, 3, 50.0), (2, 4, 0.5), (3, 4, 20.0)]:
            aggregator.feed(_make_link_report(n1, n2, v, 0, 20))
        detector.estimate()

    # Feed same quiet links again (no spike → no detection)
    for n1, n2, v in [(1, 2, 1.0), (1, 3, 0.5), (1, 4, 3.0),
                       (2, 3, 50.0), (2, 4, 0.5), (3, 4, 20.0)]:
        aggregator.feed(_make_link_report(n1, n2, v, 0, 20))

    raw = detector.estimate()
    stable = tracker.update(raw, occupied=False)
    assert stable.zone is None
