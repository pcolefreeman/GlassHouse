"""pytest fixtures and frame builder helpers shared across test files."""
import struct, pytest

MAC_DEFAULT = b'\x11\x22\x33\x44\x55\x66'
CSI_DEFAULT = bytes(range(256)) + bytes(128)  # 384 bytes, non-zero

def make_listener_frame(poll_seq=42, mac=MAC_DEFAULT, rssi=-55,
                        noise_floor=-95, csi_bytes=None):
    """Build a well-formed [0xAA][0x55] frame as bytes."""
    if csi_bytes is None:
        csi_bytes = CSI_DEFAULT
    csi_len = len(csi_bytes)
    ts = 12345678
    hdr  = struct.pack('<BB', 1, 0x00)            # ver, flags
    hdr += struct.pack('<I', ts)                   # timestamp_ms
    hdr += struct.pack('<bb', rssi, noise_floor)   # signed rssi, nf
    hdr += mac                                     # 6 bytes
    hdr += struct.pack('<I', poll_seq)             # poll_seq
    hdr += struct.pack('<H', csi_len)              # csi_len
    return b'\xAA\x55' + hdr + csi_bytes

def make_shouter_frame(poll_seq=42, mac=MAC_DEFAULT, flags=0x01,
                       tx_seq=5, tx_ms=12340000, shouter_id=1,
                       poll_rssi=-60, poll_nf=-92, csi_bytes=None):
    """Build a well-formed [0xBB][0xDD] frame as bytes."""
    if csi_bytes is None:
        csi_bytes = CSI_DEFAULT
    csi_len = len(csi_bytes)
    listener_ms = 12345000
    hdr  = struct.pack('<BB', 1, flags)            # ver, flags
    hdr += struct.pack('<I', listener_ms)
    hdr += struct.pack('<I', tx_seq)
    hdr += struct.pack('<I', tx_ms)
    hdr += struct.pack('<B', shouter_id)
    hdr += struct.pack('<I', poll_seq)
    hdr += struct.pack('<bb', poll_rssi, poll_nf)
    hdr += mac
    hdr += struct.pack('<H', csi_len)
    return b'\xBB\xDD' + hdr + csi_bytes

@pytest.fixture
def listener_frame(): return make_listener_frame()

@pytest.fixture
def shouter_frame():  return make_shouter_frame()
