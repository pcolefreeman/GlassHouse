# tests/test_debug_tab.py
"""Tests for debug_tab.ListenerDebugThread frame parsing."""
import io
import queue
import struct
import time
import pytest


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
    from ghv4.ui.debug_tab import ListenerDebugThread
    q = queue.Queue()
    t = ListenerDebugThread.__new__(ListenerDebugThread)
    import threading
    t._queue = q
    t._stop_event = threading.Event()
    return t, q, _FakeSerial(data)
