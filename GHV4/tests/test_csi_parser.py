import struct, math, pytest
from ghv4 import csi_parser
from tests.conftest import make_listener_frame, make_shouter_frame, MAC_DEFAULT

# ── parse_listener_frame ──────────────────────────────────────────────────────

def test_parse_listener_frame_returns_dict():
    raw = make_listener_frame(poll_seq=42)
    r   = csi_parser.parse_listener_frame(raw, 0)
    assert r is not None and isinstance(r, dict)

def test_parse_listener_frame_fields():
    csi = bytes(384)
    raw = make_listener_frame(poll_seq=7, mac=MAC_DEFAULT, rssi=-55,
                              noise_floor=-95, csi_bytes=csi)
    r = csi_parser.parse_listener_frame(raw, 0)
    assert r['poll_seq']    == 7
    assert r['rssi']        == -55
    assert r['noise_floor'] == -95
    assert r['mac']         == MAC_DEFAULT.hex(':')
    assert len(r['csi_bytes']) == 384

def test_parse_listener_frame_bad_magic_returns_none():
    assert csi_parser.parse_listener_frame(b'\x00\x00' + bytes(24), 0) is None

def test_parse_listener_frame_truncated_returns_none():
    raw = make_listener_frame()
    assert csi_parser.parse_listener_frame(raw[:10], 0) is None

# ── parse_shouter_frame ───────────────────────────────────────────────────────

def test_parse_shouter_frame_fields():
    raw = make_shouter_frame(poll_seq=99, mac=MAC_DEFAULT, flags=0x01,
                             tx_seq=3, poll_rssi=-62, poll_nf=-91)
    r = csi_parser.parse_shouter_frame(raw, 0)
    assert r is not None
    assert r['poll_seq']   == 99
    assert r['mac']        == MAC_DEFAULT.hex(':')
    assert r['tx_seq']     == 3
    assert r['poll_rssi']  == -62
    assert r['flags']      == 0x01

def test_parse_shouter_frame_miss_zero_csi():
    raw = make_shouter_frame(flags=0x00, csi_bytes=b'')
    r   = csi_parser.parse_shouter_frame(raw, 0)
    assert r['csi_len'] == 0
    assert r['csi_bytes'] == b''

def test_parse_shouter_frame_bad_magic_returns_none():
    assert csi_parser.parse_shouter_frame(b'\x00\x00' + bytes(31), 0) is None

# ── parse_csi_bytes ───────────────────────────────────────────────────────────

def test_parse_csi_bytes_is_public():
    """parse_csi_bytes should be a public API (no leading underscore)."""
    assert hasattr(csi_parser, 'parse_csi_bytes')
    # 4 bytes = one I/Q pair = one complex number
    result = csi_parser.parse_csi_bytes(struct.pack('<hh', 100, 200))
    assert len(result) == 1
    assert result[0] == complex(100, 200)

def test_parse_csi_bytes_returns_complex():
    # 4 int16 values → 2 complex (I=1,Q=2) and (I=3,Q=4)
    raw    = struct.pack('<hhhh', 1, 2, 3, 4)
    result = csi_parser.parse_csi_bytes(raw)
    assert len(result) == 2
    assert abs(result[0] - complex(1, 2)) < 1e-9
    assert abs(result[1] - complex(3, 4)) < 1e-9

def test_parse_csi_bytes_empty():
    assert csi_parser.parse_csi_bytes(b'') == []

# ── _extract_features ─────────────────────────────────────────────────────────

SUBCARRIERS = csi_parser.SUBCARRIERS  # convenience alias for tests

def test_extract_features_keys():
    csi  = [complex(1, 0)] * SUBCARRIERS
    feat = csi_parser._extract_features(csi, rssi=-55, noise_floor=-95)
    for k in ('amplitude', 'amplitude_norm', 'phase', 'snr', 'phase_diff'):
        assert k in feat, f"Missing key: {k}"

def test_null_subcarriers_are_nan():
    csi  = [complex(5, 0)] * SUBCARRIERS
    feat = csi_parser._extract_features(csi, rssi=-50, noise_floor=-90)
    for idx in csi_parser.NULL_SUBCARRIER_INDICES:
        assert math.isnan(feat['amplitude'][idx]), f"Sub {idx} should be NaN"

def test_phase_diff_length():
    csi  = [complex(1, 1)] * SUBCARRIERS
    feat = csi_parser._extract_features(csi, rssi=-55, noise_floor=-95)
    assert len(feat['phase_diff']) == SUBCARRIERS - 1

def test_amplitude_norm_range():
    csi  = [complex(float(i+1), 0) for i in range(SUBCARRIERS)]
    # zero out null subcarriers so they don't interfere
    for idx in csi_parser.NULL_SUBCARRIER_INDICES:
        csi[idx] = complex(0, 0)
    feat = csi_parser._extract_features(csi, rssi=-55, noise_floor=-95)
    valid = [v for v in feat['amplitude_norm'] if not math.isnan(v)]
    assert min(valid) >= 0.0 and max(valid) <= 1.0

# ── collect_one_exchange ──────────────────────────────────────────────────────

import io
from tests.conftest import make_listener_frame, make_shouter_frame

def test_collect_one_exchange_returns_matched_pair():
    mac    = b'\xAA\xBB\xCC\xDD\xEE\xFF'
    stream = io.BytesIO(make_listener_frame(poll_seq=10, mac=mac)
                      + make_shouter_frame(poll_seq=10, mac=mac))
    mock_ser = type('S', (), {'read': lambda self, n: stream.read(n)})()
    lf, sf = csi_parser.collect_one_exchange(mock_ser)
    assert lf is not None and sf is not None
    assert lf['poll_seq'] == sf['poll_seq'] == 10
    assert lf['mac'] == sf['mac']

def test_collect_one_exchange_shouter_first():
    """Shouter frame arriving before listener frame is still matched."""
    mac    = b'\x11\x22\x33\x44\x55\x66'
    stream = io.BytesIO(make_shouter_frame(poll_seq=5, mac=mac)
                      + make_listener_frame(poll_seq=5, mac=mac))
    mock_ser = type('S', (), {'read': lambda self, n: stream.read(n)})()
    lf, sf = csi_parser.collect_one_exchange(mock_ser)
    assert lf is not None and sf is not None
    assert lf['poll_seq'] == sf['poll_seq'] == 5

# ── build_feature_names + extract_feature_vector ─────────────────────────────

def test_build_feature_names_column_count():
    names = csi_parser.build_feature_names([1])
    # 6 meta + 1 shouter × 2 directions × (128×4 + 127 + 2) = 6 + 2×641 = 1288
    assert len(names) == 1288

def test_build_feature_names_four_shouters():
    names = csi_parser.build_feature_names([1, 2, 3, 4])
    assert len(names) == 5134

def test_extract_feature_vector_length():
    mac  = b'\xAA\xBB\xCC\xDD\xEE\xFF'
    csi  = bytes(range(256)) + bytes(128)  # 384 bytes
    lf   = csi_parser.parse_listener_frame(make_listener_frame(poll_seq=1, mac=mac, csi_bytes=csi), 0)
    sf   = csi_parser.parse_shouter_frame(make_shouter_frame(poll_seq=1, mac=mac, csi_bytes=csi), 0)
    names = csi_parser.build_feature_names([1])
    vec   = csi_parser.extract_feature_vector(lf, sf, names)
    assert isinstance(vec, list) and len(vec) == len(names)

def test_extract_feature_vector_none_shouter_gives_nan():
    mac  = b'\xAA\xBB\xCC\xDD\xEE\xFF'
    lf   = csi_parser.parse_listener_frame(
               make_listener_frame(poll_seq=2, mac=mac), 0)
    names = csi_parser.build_feature_names([1])
    vec   = csi_parser.extract_feature_vector(lf, None, names)
    # tx columns should all be NaN
    tx_idx = next(i for i, n in enumerate(names) if n == 's1_tx_amp_0')
    assert math.isnan(vec[tx_idx])


# ── parse_csi_snap_frame ────────────────────────────────────────────────────

def _make_snap_buf(reporter=1, peer=2, seq=0, csi_len=256, csi_val=42) -> bytes:
    """Build a parse_csi_snap_frame input buffer (after magic bytes)."""
    import struct
    csi = bytes([csi_val]) * csi_len
    return struct.pack('<BBBBH', 1, reporter, peer, seq, csi_len) + csi


def test_parse_csi_snap_frame_returns_dict():
    from ghv4.csi_parser import parse_csi_snap_frame
    buf = _make_snap_buf()
    result = parse_csi_snap_frame(buf)
    assert result is not None
    assert result['type'] == 'csi_snap'
    assert result['reporter_id'] == 1
    assert result['peer_id'] == 2
    assert result['snap_seq'] == 0
    assert len(result['csi']) == 256


def test_parse_csi_snap_frame_returns_none_on_short_header():
    from ghv4.csi_parser import parse_csi_snap_frame
    assert parse_csi_snap_frame(bytes(5)) is None   # < 6 header bytes


def test_parse_csi_snap_frame_returns_none_on_truncated_csi():
    from ghv4.csi_parser import parse_csi_snap_frame
    import struct
    # Header claims 256 csi bytes but only provides 10
    buf = struct.pack('<BBBBH', 1, 1, 2, 0, 256) + bytes(10)
    assert parse_csi_snap_frame(buf) is None


def test_parse_csi_snap_frame_constants_exist():
    from ghv4.csi_parser import SER_D_MAGIC_0, SER_D_MAGIC_1, CSI_SNAP_HEADER_SIZE
    assert SER_D_MAGIC_0 == 0xEE
    assert SER_D_MAGIC_1 == 0xFF
    assert CSI_SNAP_HEADER_SIZE == 6
