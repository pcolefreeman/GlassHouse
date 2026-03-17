# tests/test_serial_io.py
"""Tests for serial_io SerialReader 0xCC branch and ranging dispatch."""
import io
import queue
import struct
import threading
import time

import pytest


def _make_ser_c_bytes(reporter_id=1, peer_rssi=None, peer_count=None):
    """Build a complete [0xCC][0xDD] serial frame as bytes."""
    if peer_rssi is None:
        peer_rssi = [0, 0, -55, -60, -65]
    if peer_count is None:
        peer_count = [0, 0, 5, 4, 7]
    payload = struct.pack('<BB5b5B', 1, reporter_id, *peer_rssi, *peer_count)
    return bytes([0xCC, 0xDD]) + payload  # 14 bytes total


class _FakeSerial:
    """Minimal serial.Serial stub backed by a BytesIO buffer."""
    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)
        self.timeout = 1

    def read(self, n: int) -> bytes:
        return self._buf.read(n)

    def close(self):
        pass

    def is_open(self):
        return True


# ── SerialReader 0xCC branch ─────────────────────────────────────────────────

def test_serial_reader_enqueues_ranging_frame():
    """A [0xCC][0xDD] frame must be queued as ('ranging', frame_dict)."""
    from ghv3_1.serial_io import SerialReader
    data = _make_ser_c_bytes(reporter_id=2,
                              peer_rssi=[0, -50, 0, -60, -55],
                              peer_count=[0,  5, 0,  5,   5])
    ser = _FakeSerial(data)
    q = queue.Queue()
    reader = SerialReader(ser, q)
    reader._read_one_frame()
    assert not q.empty()
    ftype, frame = q.get_nowait()
    assert ftype == 'ranging'
    assert frame['magic'] == (0xCC, 0xDD)
    assert frame['reporter_id'] == 2
    assert len(frame['payload']) == 12


def test_serial_reader_ignores_wrong_second_byte():
    """[0xCC] not followed by [0xDD] must be silently dropped."""
    from ghv3_1.serial_io import SerialReader
    data = bytes([0xCC, 0x00])  # wrong second byte
    ser = _FakeSerial(data)
    q = queue.Queue()
    reader = SerialReader(ser, q)
    reader._read_one_frame()
    assert q.empty()


def test_serial_reader_0xcc_before_0x20_branch():
    """0xCC (204) is >= 0x20 (32). The 0xCC branch MUST precede the 0x20 branch
    in the if/elif chain, otherwise ranging frames are swallowed as text lines.
    This test injects a frame and verifies it arrives as 'ranging', not as text.
    """
    from ghv3_1.serial_io import SerialReader
    data = _make_ser_c_bytes()
    ser = _FakeSerial(data)
    q = queue.Queue()
    reader = SerialReader(ser, q)
    reader._read_one_frame()
    assert not q.empty()
    ftype, _ = q.get_nowait()
    assert ftype == 'ranging', (
        "0xCC frame was not recognised as 'ranging' — "
        "check that the elif b0 == 0xCC branch precedes elif b0 >= 0x20"
    )


def test_serial_reader_short_payload_dropped():
    """A [0xCC][0xDD] frame with < 12 payload bytes must be dropped."""
    from ghv3_1.serial_io import SerialReader
    data = bytes([0xCC, 0xDD]) + bytes(6)  # only 6 payload bytes
    ser = _FakeSerial(data)
    q = queue.Queue()
    reader = SerialReader(ser, q)
    reader._read_one_frame()
    assert q.empty()


# ── Dispatch loop routes 'ranging' to SpacingEstimator ───────────────────────

def test_csv_writer_routes_ranging_to_spacing_estimator():
    """CSVWriter must pass 'ranging' frames to spacing_estimator.feed()."""
    fed = []

    class FakeSpacingEstimator:
        def feed(self, frame):
            fed.append(frame)

    from ghv3_1.serial_io import CSVWriter
    q = queue.Queue()
    writer = CSVWriter(q, io.StringIO(), spacing_estimator=FakeSpacingEstimator())
    frame = {'magic': (0xCC, 0xDD), 'reporter_id': 1, 'payload': bytes(12)}
    q.put(('ranging', frame))
    q.put(None)  # sentinel to stop the writer
    writer.run()
    assert len(fed) == 1
    assert fed[0] is frame


# ── [0xEE][0xFF] dispatch → CSIMUSICEstimator ─────────────────────────────────

def _make_ser_d_frame(reporter=1, peer=2, seq=0, csi_len=256) -> bytes:
    """Build a complete [0xEE][0xFF] serial frame."""
    import struct
    header = struct.pack('<BBBBH', 1, reporter, peer, seq, csi_len)
    csi = bytes(csi_len)
    return bytes([0xEE, 0xFF]) + header + csi


def test_serial_reader_dispatches_ee_ff_to_music_estimator(tmp_path):
    """SerialReader must call music_estimator.collect() for [0xEE][0xFF] frames."""
    import io
    from unittest.mock import MagicMock
    from ghv3_1 import serial_io as ghv3

    frame_data = _make_ser_d_frame(reporter=1, peer=2, csi_len=256)
    ser = _FakeSerial(frame_data)

    music_est = MagicMock()
    fq = queue.Queue()
    reader = ghv3.SerialReader(ser, fq, music_estimator=music_est)

    # Drive frame parsing manually until queue or music_est is populated
    for _ in range(len(frame_data) + 10):
        reader._read_one_frame()

    music_est.collect.assert_called_once()
    call_args = music_est.collect.call_args[0]
    assert call_args[0] == 1   # reporter_id
    assert call_args[1] == 2   # peer_id
    assert len(call_args[2]) == 256   # csi bytes


def test_read_exact_returns_none_on_short_read(tmp_path):
    """_read_exact must return None if fewer than n bytes are available."""
    from unittest.mock import MagicMock
    from ghv3_1 import serial_io as ghv3

    ser = MagicMock()
    ser.read.return_value = bytes(3)   # only 3 bytes when 8 requested
    reader = ghv3.SerialReader(ser, queue.Queue())
    result = reader._read_exact(8)
    assert result is None
