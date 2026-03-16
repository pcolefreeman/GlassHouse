import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import struct, math, pytest
import csi_parser
from conftest import make_listener_frame, make_shouter_frame, MAC_DEFAULT

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

# ── _parse_csi_bytes ──────────────────────────────────────────────────────────

def test_parse_csi_bytes_returns_complex():
    # 4 int16 values → 2 complex (I=1,Q=2) and (I=3,Q=4)
    raw    = struct.pack('<hhhh', 1, 2, 3, 4)
    result = csi_parser._parse_csi_bytes(raw)
    assert len(result) == 2
    assert abs(result[0] - complex(1, 2)) < 1e-9
    assert abs(result[1] - complex(3, 4)) < 1e-9

def test_parse_csi_bytes_empty():
    assert csi_parser._parse_csi_bytes(b'') == []

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
from conftest import make_listener_frame, make_shouter_frame

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
    # 5 meta + 1 shouter × 2 directions × (128×4 + 127 + 2) = 5 + 2×641 = 1287
    assert len(names) == 1287

def test_build_feature_names_four_shouters():
    names = csi_parser.build_feature_names([1, 2, 3, 4])
    assert len(names) == 5133

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
