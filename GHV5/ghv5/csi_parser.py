"""csi_parser.py — shared frame parsing for GHV5.

Frame wire layouts (spec Section 5.3):
  [0xAA][0x55]: magic(2) ver(1) flags(1) ts_ms(4) rssi(1) nf(1) mac(6) poll_seq(4)
                csi_len(2) csi[N]   — header after magic: 20 bytes
  [0xBB][0xDD]: magic(2) ver(1) flags(1) listener_ms(4) tx_seq(4) tx_ms(4)
                shouter_id(1) poll_seq(4) poll_rssi(1) poll_nf(1) mac(6)
                csi_len(2) csi[N]   — header after magic: 29 bytes
"""
import struct
from typing import Optional

from ghv5.config import (
    SUBCARRIERS,
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


def parse_csi_bytes(csi_bytes: bytes) -> list:
    """Convert raw CSI bytes (int16 I/Q pairs, little-endian) to list[complex]."""
    return [complex(i, q) for i, q in struct.iter_unpack('<hh', csi_bytes)]


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
