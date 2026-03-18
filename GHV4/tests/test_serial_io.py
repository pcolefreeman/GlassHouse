# tests/test_serial_io.py
"""Tests for serial_io SerialReader and CSVWriter (GHV4 — MUSIC-only)."""
import io
import queue
import struct
import threading
import time

import pytest


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


def _make_cc_frame(reporter_id=1, peer_rssi=None, peer_count=None):
    """Build a complete [0xCC][0xDD] serial frame as bytes."""
    if peer_rssi is None:
        peer_rssi = [0, 0, -55, -60, -65]
    if peer_count is None:
        peer_count = [0, 0, 5, 4, 7]
    payload = struct.pack('<BB5b5B', 1, reporter_id, *peer_rssi, *peer_count)
    return bytes([0xCC, 0xDD]) + payload  # 14 bytes total


# ── SerialReader 0xCC branch (consume-and-discard) ───────────────────────────

def test_serial_reader_discards_cc_frame():
    """A [0xCC][0xDD] frame must be consumed but NOT enqueued (RSSI removed in GHV4)."""
    from ghv4.serial_io import SerialReader
    data = _make_cc_frame(reporter_id=2)
    ser = _FakeSerial(data)
    q = queue.Queue()
    reader = SerialReader(ser, q)
    reader._read_one_frame()
    assert q.empty(), "0xCC frame should be discarded, not enqueued"


def test_serial_reader_cc_ignores_wrong_second_byte():
    """[0xCC] not followed by [0xDD] must be silently dropped."""
    from ghv4.serial_io import SerialReader
    data = bytes([0xCC, 0x00])
    ser = _FakeSerial(data)
    q = queue.Queue()
    reader = SerialReader(ser, q)
    reader._read_one_frame()
    assert q.empty()


def test_serial_reader_cc_short_payload_consumed():
    """A [0xCC][0xDD] frame with < 12 payload bytes must not crash."""
    from ghv4.serial_io import SerialReader
    data = bytes([0xCC, 0xDD]) + bytes(6)  # only 6 payload bytes
    ser = _FakeSerial(data)
    q = queue.Queue()
    reader = SerialReader(ser, q)
    reader._read_one_frame()
    assert q.empty()


def test_serial_reader_cc_keeps_stream_aligned():
    """After consuming a [0xCC][0xDD] frame, the next frame must parse correctly."""
    from ghv4.serial_io import SerialReader
    # 0xCC frame followed by a 0xEE frame header (incomplete, but stream position matters)
    cc_data = _make_cc_frame()
    # After consuming CC frame, reader should be at the right position
    ser = _FakeSerial(cc_data)
    q = queue.Queue()
    reader = SerialReader(ser, q)
    reader._read_one_frame()
    # Stream should be fully consumed (14 bytes: 2 magic + 12 payload)
    assert ser._buf.read(1) == b''


# ── [0xEE][0xFF] dispatch → CSIMUSICEstimator ─────────────────────────────────

def _make_ee_frame(reporter=1, peer=2, seq=0, csi_len=256) -> bytes:
    """Build a complete [0xEE][0xFF] serial frame."""
    header = struct.pack('<BBBBH', 1, reporter, peer, seq, csi_len)
    csi = bytes(csi_len)
    return bytes([0xEE, 0xFF]) + header + csi


def test_serial_reader_dispatches_ee_ff_to_music_estimator():
    """SerialReader must call music_estimator.collect() for [0xEE][0xFF] frames."""
    from unittest.mock import MagicMock
    from ghv4.serial_io import SerialReader

    frame_data = _make_ee_frame(reporter=1, peer=2, csi_len=256)
    ser = _FakeSerial(frame_data)
    music_est = MagicMock()
    q = queue.Queue()
    reader = SerialReader(ser, q, music_estimator=music_est)

    for _ in range(len(frame_data) + 10):
        reader._read_one_frame()

    music_est.collect.assert_called_once()
    call_args = music_est.collect.call_args[0]
    assert call_args[0] == 1   # reporter_id
    assert call_args[1] == 2   # peer_id
    assert len(call_args[2]) == 256   # csi bytes


def test_read_exact_returns_none_on_short_read():
    """_read_exact must return None if fewer than n bytes are available."""
    from unittest.mock import MagicMock
    from ghv4.serial_io import SerialReader

    ser = MagicMock()
    ser.read.return_value = bytes(3)
    reader = SerialReader(ser, queue.Queue())
    result = reader._read_exact(8)
    assert result is None
