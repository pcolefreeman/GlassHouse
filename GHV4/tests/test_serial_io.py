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


def test_snap_callback_receives_parsed_frame():
    """When snap_callback is set, [0xEE][0xFF] frames are forwarded."""
    from ghv4.serial_io import SerialReader

    received = []

    def on_snap(reporter_id, peer_id, snap_seq, csi_bytes):
        received.append((reporter_id, peer_id, snap_seq, csi_bytes))

    # Build a [0xEE][0xFF] frame: header (6 bytes after magic): ver=1, reporter=1, peer=2, seq=10, csi_len=256
    header = struct.pack("<BBBBH", 1, 1, 2, 10, 256)
    csi_payload = bytes(256)
    frame_data = bytes([0xEE, 0xFF]) + header + csi_payload

    ser = _FakeSerial(frame_data)
    fq = queue.Queue()

    reader = SerialReader(ser, fq, snap_callback=on_snap)
    reader._read_one_frame()

    assert len(received) == 1
    assert received[0][0] == 1   # reporter_id
    assert received[0][1] == 2   # peer_id
    assert received[0][2] == 10  # snap_seq
