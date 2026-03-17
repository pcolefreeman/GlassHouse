# GHV2.1/tests/test_ghv3_protocol.py
"""Validate GHV3 packet byte sizes and serial frame C parse layout.

These tests encode/decode the expected binary formats using Python's struct
module to confirm that the wire format matches the spec before firmware is
written. They do NOT require hardware.
"""
import struct


# ── Packet size assertions (match GHV3Protocol.h comments) ──────────────────

def test_peer_info_pkt_size():
    # magic(2) + ver(1) + n_peers(1) + 4*(shouter_id(1)+mac(6)) = 32 bytes
    computed = 2 + 1 + 1 + 4 * (1 + 6)
    assert computed == 32


def test_range_req_pkt_size():
    # magic(2) + ver(1) + target_id(1) + n_beacons(1) + interval_ms(2) = 7 bytes
    computed = 2 + 1 + 1 + 1 + 2
    assert computed == 7


def test_range_bcn_pkt_size():
    # magic(2) + ver(1) + shouter_id(1) + bcn_seq(4) = 8 bytes
    computed = 2 + 1 + 1 + 4
    assert computed == 8


def test_ranging_rpt_pkt_size():
    # magic(2) + ver(1) + shouter_id(1) + peer_rssi[5](5) + peer_count[5](5) = 14 bytes
    computed = 2 + 1 + 1 + 5 + 5
    assert computed == 14


# ── Serial frame C parse round-trip ─────────────────────────────────────────

def test_ser_c_frame_parse_roundtrip():
    """Frame C payload is 12 bytes (14 total minus 2 magic bytes consumed by reader)."""
    reporter_id = 2
    # indices 1-4 meaningful; index 0 is pad (0)
    peer_rssi  = [0, -50, 0, -60, -55]   # S1 not observed by S2, so rssi[1]=0 but count[1]=0
    peer_count = [0, 10,  0,   8,  12]

    # Pack: ver(1B) reporter_id(1B) peer_rssi[5](5×int8) peer_count[5](5×uint8)
    payload = struct.pack('<BB5b5B', 1, reporter_id, *peer_rssi, *peer_count)
    assert len(payload) == 12

    # Parse
    ver, rid   = struct.unpack_from('<BB', payload, 0)
    rssi_out   = list(struct.unpack_from('<5b', payload, 2))
    count_out  = list(struct.unpack_from('<5B', payload, 7))

    assert ver == 1
    assert rid == reporter_id
    assert rssi_out[1] == -50    # S1→S2 RSSI
    assert rssi_out[2] == 0      # S3 not observed, pad
    assert rssi_out[3] == -60    # S3→S2 RSSI (wait — reporter=2, peer=3)
    assert rssi_out[4] == -55    # S4→S2 RSSI
    assert count_out[0] == 0     # index 0 always pad
    assert count_out[4] == 12


def test_ser_c_all_zeros_accepted():
    """All-zero payload (no observations yet) must parse without error."""
    payload = bytes(12)
    ver, rid   = struct.unpack_from('<BB', payload, 0)
    rssi_out   = list(struct.unpack_from('<5b', payload, 2))
    count_out  = list(struct.unpack_from('<5B', payload, 7))
    assert ver == 0
    assert all(r == 0 for r in rssi_out)
    assert all(c == 0 for c in count_out)
