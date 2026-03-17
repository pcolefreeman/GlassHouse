# tests/test_debug_tab.py
"""Tests for debug_tab.ListenerDebugThread 0xCC branch."""
import io
import queue
import struct
import time
import pytest


def _make_ser_c_bytes(reporter_id=1):
    peer_rssi  = [0, -50, -60, 0, -55]
    peer_count = [0,  5,   6, 0,   7]
    payload = struct.pack('<BB5b5B', 1, reporter_id, *peer_rssi, *peer_count)
    return bytes([0xCC, 0xDD]) + payload


class _FakeSerial:
    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)
        self._stop = False

    def read(self, n: int) -> bytes:
        return self._buf.read(n)

    def close(self):
        pass

    @property
    def is_open(self):
        return True


def _make_thread(data: bytes):
    from ghv3_1.ui.debug_tab import ListenerDebugThread
    q = queue.Queue()
    t = ListenerDebugThread.__new__(ListenerDebugThread)
    import threading
    t._queue = q
    t._stop_event = threading.Event()
    return t, q, _FakeSerial(data)


def test_0xcc_enqueues_ranging_frame():
    t, q, ser = _make_thread(_make_ser_c_bytes(reporter_id=3))
    t._read_one(ser)
    assert not q.empty()
    item = q.get_nowait()
    assert item['type'] == 'ranging_frame'
    assert item['payload'][1] == 3   # reporter_id at byte 1 of payload


def test_0xcc_wrong_second_byte_dropped():
    t, q, ser = _make_thread(bytes([0xCC, 0x00]) + bytes(12))
    t._read_one(ser)
    assert q.empty()


def test_0xcc_short_payload_dropped():
    t, q, ser = _make_thread(bytes([0xCC, 0xDD]) + bytes(6))
    t._read_one(ser)
    assert q.empty()


def test_0xcc_not_consumed_by_0x20_branch():
    """0xCC (204 >= 32) must be handled before the >= 0x20 text branch."""
    t, q, ser = _make_thread(_make_ser_c_bytes())
    t._read_one(ser)
    assert not q.empty()
    item = q.get_nowait()
    # If consumed by 0x20 branch, type would be 'lst_text', not 'ranging_frame'
    assert item['type'] == 'ranging_frame'
