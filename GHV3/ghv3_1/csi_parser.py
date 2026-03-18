"""csi_parser.py — shared frame parsing and feature extraction for GHV2.

Frame wire layouts (spec Section 5.3):
  [0xAA][0x55]: magic(2) ver(1) flags(1) ts_ms(4) rssi(1) nf(1) mac(6) poll_seq(4)
                csi_len(2) csi[N]   — header after magic: 20 bytes
  [0xBB][0xDD]: magic(2) ver(1) flags(1) listener_ms(4) tx_seq(4) tx_ms(4)
                shouter_id(1) poll_seq(4) poll_rssi(1) poll_nf(1) mac(6)
                csi_len(2) csi[N]   — header after magic: 29 bytes
"""
import struct
import math
from typing import Optional

from ghv3_1.config import (
    SUBCARRIERS,
    BUCKET_MS,
    POLL_INTERVAL_MIN_MS,
    ACTIVE_SHOUTER_IDS,
    NULL_SUBCARRIER_INDICES,
    MAGIC_LISTENER,
    MAGIC_SHOUTER,
    MAGIC_CSI_SNAP,
    LISTENER_HDR_SIZE,
    SHOUTER_HDR_SIZE,
    CSI_SNAP_HDR_SIZE,
)

# Module-level aliases for internal use (preserving existing code references)
_MAGIC_LISTENER = MAGIC_LISTENER
_MAGIC_SHOUTER = MAGIC_SHOUTER
_HDR_A = LISTENER_HDR_SIZE
_HDR_B = SHOUTER_HDR_SIZE

# Derived from MAGIC_CSI_SNAP for consistency
SER_D_MAGIC_0 = MAGIC_CSI_SNAP[0]  # 0xEE
SER_D_MAGIC_1 = MAGIC_CSI_SNAP[1]  # 0xFF

CSI_SNAP_HEADER_SIZE = CSI_SNAP_HDR_SIZE  # after the 2 magic bytes: ver(1)+reporter(1)+peer(1)+seq(1)+csi_len(2) = 6
# NOTE: offsetof(csi_snap_pkt_t, csi) = 8 in C (magic included), but parse_csi_snap_frame
# receives a buffer AFTER magic is already consumed, so the pre-CSI header is only 6 bytes.


def parse_listener_frame(raw: bytes, offset: int) -> Optional[dict]:
    """Parse a [0xAA][0x55] frame starting at offset. Returns dict or None."""
    if len(raw) < offset + 2 or raw[offset:offset+2] != _MAGIC_LISTENER:
        return None
    pos = offset + 2
    if len(raw) < pos + _HDR_A:
        return None
    ver, flags         = struct.unpack_from('<BB', raw, pos); pos += 2
    ts_ms,             = struct.unpack_from('<I',  raw, pos); pos += 4
    rssi, noise_floor  = struct.unpack_from('<bb', raw, pos); pos += 2
    mac_bytes          = raw[pos:pos+6];                      pos += 6
    poll_seq,          = struct.unpack_from('<I',  raw, pos); pos += 4
    csi_len,           = struct.unpack_from('<H',  raw, pos); pos += 2
    if len(raw) < pos + csi_len:
        return None
    return {
        'frame_type':   'listener',
        'ver':          ver,
        'flags':        flags,
        'timestamp_ms': ts_ms,
        'rssi':         rssi,
        'noise_floor':  noise_floor,
        'mac':          mac_bytes.hex(':'),
        'poll_seq':     poll_seq,
        'csi_len':      csi_len,
        'csi_bytes':    raw[pos:pos+csi_len],
        'total_size':   pos + csi_len - offset,
    }


def parse_shouter_frame(raw: bytes, offset: int) -> Optional[dict]:
    """Parse a [0xBB][0xDD] frame starting at offset. Returns dict or None."""
    if len(raw) < offset + 2 or raw[offset:offset+2] != _MAGIC_SHOUTER:
        return None
    pos = offset + 2
    if len(raw) < pos + _HDR_B:
        return None
    ver, flags          = struct.unpack_from('<BB', raw, pos); pos += 2
    listener_ms,        = struct.unpack_from('<I',  raw, pos); pos += 4
    tx_seq,             = struct.unpack_from('<I',  raw, pos); pos += 4
    tx_ms,              = struct.unpack_from('<I',  raw, pos); pos += 4
    shouter_id,         = struct.unpack_from('<B',  raw, pos); pos += 1
    poll_seq,           = struct.unpack_from('<I',  raw, pos); pos += 4
    poll_rssi, poll_nf  = struct.unpack_from('<bb', raw, pos); pos += 2
    mac_bytes           = raw[pos:pos+6];                      pos += 6
    csi_len,            = struct.unpack_from('<H',  raw, pos); pos += 2
    if len(raw) < pos + csi_len:
        return None
    return {
        'frame_type':       'shouter',
        'ver':              ver,
        'flags':            flags,
        'listener_ms':      listener_ms,
        'tx_seq':           tx_seq,
        'tx_ms':            tx_ms,
        'shouter_id':       shouter_id,
        'poll_seq':         poll_seq,
        'poll_rssi':        poll_rssi,
        'poll_noise_floor': poll_nf,
        'mac':              mac_bytes.hex(':'),
        'csi_len':          csi_len,
        'csi_bytes':        raw[pos:pos+csi_len],
        'total_size':       pos + csi_len - offset,
    }


def _parse_csi_bytes(csi_bytes: bytes) -> list:
    """Convert raw CSI bytes (int16 I/Q pairs, little-endian) to list[complex]."""
    return [complex(i, q) for i, q in struct.iter_unpack('<hh', csi_bytes)]


def _normalize_amplitude(amplitudes: list) -> list:
    """Normalize amplitudes to [0, 1]; NaN preserved for null subcarriers."""
    valid = [a for a in amplitudes if not math.isnan(a)]
    if not valid:
        return [float('nan')] * len(amplitudes)
    lo, hi = min(valid), max(valid)
    rng = hi - lo
    if rng == 0:
        return [float('nan') if math.isnan(a) else 0.0 for a in amplitudes]
    return [float('nan') if math.isnan(a) else (a - lo) / rng for a in amplitudes]


def _compute_snr(amplitudes: list, noise_floor_dbm: float) -> list:
    """Per-subcarrier SNR in dB relative to noise floor. NaN for null subs."""
    noise = 10 ** (noise_floor_dbm / 20.0)
    result = []
    for amp in amplitudes:
        if math.isnan(amp) or amp <= 0 or noise <= 0:
            result.append(float('nan'))
        else:
            result.append(20.0 * math.log10(amp / noise))
    return result


def _extract_features(csi_complex: list, rssi: float, noise_floor: float) -> dict:
    """Extract per-subcarrier feature arrays from a list of complex CSI samples.

    Null subcarrier indices (NULL_SUBCARRIER_INDICES) are replaced with NaN.
    Returns dict with keys: amplitude, amplitude_norm, phase, snr, phase_diff.
    """
    n         = len(csi_complex)
    amplitude = []
    phase     = []
    for i, c in enumerate(csi_complex):
        if i in NULL_SUBCARRIER_INDICES:
            amplitude.append(float('nan'))
            phase.append(float('nan'))
        else:
            amplitude.append(abs(c))
            phase.append(math.atan2(c.imag, c.real))

    amplitude_norm = _normalize_amplitude(amplitude)
    snr            = _compute_snr(amplitude, noise_floor)

    phase_diff = []
    for i in range(n - 1):
        if math.isnan(phase[i]) or math.isnan(phase[i + 1]):
            phase_diff.append(float('nan'))
        else:
            diff = phase[i + 1] - phase[i]
            # Wrap to (-π, π] via IEEE remainder — O(1), no loop needed
            phase_diff.append(math.remainder(diff, 2 * math.pi))

    return {
        'amplitude':      amplitude,
        'amplitude_norm': amplitude_norm,
        'phase':          phase,
        'snr':            snr,
        'phase_diff':     phase_diff,
    }


def build_feature_names(active_ids=None) -> list:
    """Return the ordered CSV column name list.  Must match CSVWriter output."""
    if active_ids is None:
        active_ids = ACTIVE_SHOUTER_IDS
    header = ["timestamp_ms", "label", "zone_id", "grid_row", "grid_col", "activity"]
    for s in active_ids:
        for px in [f"s{s}", f"s{s}_tx"]:
            for sc in range(SUBCARRIERS):
                header += [f"{px}_amp_{sc}", f"{px}_amp_norm_{sc}",
                           f"{px}_phase_{sc}", f"{px}_snr_{sc}"]
            for sc in range(SUBCARRIERS - 1):
                header.append(f"{px}_pdiff_{sc}")
            header += [f"{px}_rssi", f"{px}_noise_floor"]
    return header


def _frame_to_feature_dict(frame: Optional[dict], prefix: str) -> dict:
    """Flatten one frame dict into {column_name: float} for a given prefix."""
    nan = float('nan')
    out = {}
    if frame is None or frame.get('csi_len', 0) == 0:
        for sc in range(SUBCARRIERS):
            out[f"{prefix}_amp_{sc}"]      = nan
            out[f"{prefix}_amp_norm_{sc}"] = nan
            out[f"{prefix}_phase_{sc}"]    = nan
            out[f"{prefix}_snr_{sc}"]      = nan
        for sc in range(SUBCARRIERS - 1):
            out[f"{prefix}_pdiff_{sc}"] = nan
        out[f"{prefix}_rssi"]        = nan
        out[f"{prefix}_noise_floor"] = nan
        return out

    csi_complex = _parse_csi_bytes(frame['csi_bytes'])
    while len(csi_complex) < SUBCARRIERS:
        csi_complex.append(complex(0, 0))
    csi_complex = csi_complex[:SUBCARRIERS]

    rssi = frame.get('rssi', frame.get('poll_rssi', 0))
    nf   = frame.get('noise_floor', frame.get('poll_noise_floor', -95))
    feat = _extract_features(csi_complex, rssi, nf)

    for sc in range(SUBCARRIERS):
        out[f"{prefix}_amp_{sc}"]      = feat['amplitude'][sc]
        out[f"{prefix}_amp_norm_{sc}"] = feat['amplitude_norm'][sc]
        out[f"{prefix}_phase_{sc}"]    = feat['phase'][sc]
        out[f"{prefix}_snr_{sc}"]      = feat['snr'][sc]
    for sc in range(SUBCARRIERS - 1):
        out[f"{prefix}_pdiff_{sc}"] = feat['phase_diff'][sc]
    out[f"{prefix}_rssi"]        = rssi
    out[f"{prefix}_noise_floor"] = nf
    return out


def extract_feature_vector(lf: Optional[dict], sf: Optional[dict],
                           feature_names: list,
                           shouter_id: Optional[int] = None) -> list:
    """Flatten (lf, sf) into a float list aligned to feature_names.

    Meta columns (timestamp_ms, label, zone_id, grid_row, grid_col) are NaN;
    the caller fills them before writing to CSV.

    shouter_id: explicit override; if None, derived from sf['shouter_id'].
    Callers should pass shouter_id when sf may be None (MISS frames) to avoid
    misattributing features to the wrong shouter column.
    """
    nan = float('nan')
    # Determine shouter ID: explicit arg > sf field > fallback
    if shouter_id is not None:
        sid = shouter_id
    elif sf is not None:
        sid = sf['shouter_id']
    else:
        sid = 1  # sf missing and no override; caller should supply shouter_id
    lookup = {}
    lookup.update(_frame_to_feature_dict(lf, f"s{sid}"))
    lookup.update(_frame_to_feature_dict(sf, f"s{sid}_tx"))
    return [lookup.get(col, nan) for col in feature_names]


_PENDING_MAX = 64  # max unmatched frames before oldest is evicted


def collect_one_exchange(ser) -> tuple:
    """Read Serial until a matched ([0xAA][0x55], [0xBB][0xDD]) pair is found.

    Matches on (mac, poll_seq). Returns (listener_dict, shouter_dict).
    Returns (None, None) on EOF.
    """
    pending_lf = {}   # (mac, poll_seq) → listener frame dict
    pending_sf = {}   # (mac, poll_seq) → shouter frame dict

    while True:
        b = ser.read(1)
        if not b:
            return None, None
        b0 = b[0]

        if b0 == 0xAA:
            b1 = ser.read(1)
            if not b1 or b1[0] != 0x55:
                continue
            hdr = ser.read(_HDR_A)
            if len(hdr) < _HDR_A:
                return None, None
            csi_len = struct.unpack_from('<H', hdr, 18)[0]
            csi = ser.read(csi_len)
            frame = parse_listener_frame(b'\xAA\x55' + hdr + csi, 0)
            if frame:
                key = (frame['mac'], frame['poll_seq'])
                pending_lf[key] = frame
                if len(pending_lf) > _PENDING_MAX:
                    pending_lf.pop(next(iter(pending_lf)))
                if key in pending_sf:
                    return frame, pending_sf.pop(key)

        elif b0 == 0xBB:
            b1 = ser.read(1)
            if not b1 or b1[0] != 0xDD:
                continue
            hdr = ser.read(_HDR_B)
            if len(hdr) < _HDR_B:
                return None, None
            csi_len = struct.unpack_from('<H', hdr, 27)[0]
            csi = ser.read(csi_len)
            frame = parse_shouter_frame(b'\xBB\xDD' + hdr + csi, 0)
            if frame:
                key = (frame['mac'], frame['poll_seq'])
                pending_sf[key] = frame
                if len(pending_sf) > _PENDING_MAX:
                    pending_sf.pop(next(iter(pending_sf)))
                if key in pending_lf:
                    return pending_lf.pop(key), frame


def parse_csi_snap_frame(buf: bytes):
    """
    Parse payload of [0xEE][0xFF] frame (magic bytes already consumed by dispatcher).
    buf: bytes starting immediately after the two magic bytes.
    Returns dict or None on error.
    """
    if len(buf) < CSI_SNAP_HEADER_SIZE:
        return None
    ver         = buf[0]
    reporter_id = buf[1]
    peer_id     = buf[2]
    snap_seq    = buf[3]
    csi_len     = struct.unpack_from('<H', buf, 4)[0]
    if len(buf) < CSI_SNAP_HEADER_SIZE + csi_len:
        return None
    csi = buf[CSI_SNAP_HEADER_SIZE: CSI_SNAP_HEADER_SIZE + csi_len]
    return {
        'type':        'csi_snap',
        'reporter_id': reporter_id,
        'peer_id':     peer_id,
        'snap_seq':    snap_seq,
        'csi':         csi,
    }
