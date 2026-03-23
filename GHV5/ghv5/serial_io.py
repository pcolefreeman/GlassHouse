"""serial_io.py — serial byte-stream reader for GHV5.

Threads:
  SerialReader — byte stream → frame_queue  (parses all frame types)
"""
import logging
import queue
import serial
import struct
import threading
import time

from ghv5 import csi_parser
from ghv5.config import BAUD_RATE

_log = logging.getLogger("ghv5.serial_io")


# ── SerialReader ───────────────────────────────────────────────────────────────
class SerialReader(threading.Thread):
    """Reads a byte stream, detects magic sequences, parses frames, enqueues them."""

    def __init__(self, ser, frame_queue: queue.Queue, music_estimator=None, snap_callback=None):
        super().__init__(daemon=True, name="SerialReader")
        self._ser              = ser
        self._queue            = frame_queue
        self._running          = False
        self._music_estimator  = music_estimator
        self._snap_callback    = snap_callback
        self._snap_parsed   = 0
        self._snap_failed   = 0
        self._sync_errors   = 0
        self._last_diag_ts  = time.time()

    def run(self):
        self._running = True
        while self._running:
            self._read_one_frame()
            self._maybe_log_diag()

    def stop(self):
        self._running = False

    def _maybe_log_diag(self):
        now = time.time()
        if now - self._last_diag_ts >= 5.0:
            if self._snap_parsed or self._snap_failed or self._sync_errors:
                _log.info("[DIAG] snap_parsed=%d snap_failed=%d sync_errors=%d (last 5s)",
                          self._snap_parsed, self._snap_failed, self._sync_errors)
            self._snap_parsed  = 0
            self._snap_failed  = 0
            self._sync_errors  = 0
            self._last_diag_ts = now

    def _read_exact(self, n: int):
        """Read exactly n bytes from serial. Returns None if fewer bytes arrive."""
        data = self._ser.read(n)
        return data if len(data) == n else None

    def _read_one_frame(self):
        b = self._ser.read(1)
        if not b:
            return
        b0 = b[0]

        if b0 == 0xAA:
            b1 = self._ser.read(1)
            if not b1 or b1[0] != 0x55:
                return
            hdr = self._ser.read(20)
            if len(hdr) < 20:
                return
            csi_len = struct.unpack_from('<H', hdr, 18)[0]
            csi = self._ser.read(csi_len)
            if len(csi) < csi_len:
                return  # short read — drop truncated frame
            frame = csi_parser.parse_listener_frame(b'\xAA\x55' + hdr + csi, 0)
            if frame:
                self._queue.put(('listener', frame))

        elif b0 == 0xEE:
            b1 = self._ser.read(1)
            if not b1 or b1[0] != 0xFF:
                self._sync_errors += 1
                return
            header = self._read_exact(6)
            if header is None:
                self._snap_failed += 1
                return
            csi_len = struct.unpack_from('<H', header, 4)[0]
            if csi_len < 1 or csi_len > 384:
                _log.debug("[DIAG] bad csi_len=%d in snap frame, discarding", csi_len)
                self._snap_failed += 1
                return  # return to top-level loop for natural resync
            old_timeout = self._ser.timeout
            self._ser.timeout = 0.05  # 50ms read timeout for CSI payload
            csi_bytes = self._read_exact(csi_len)
            self._ser.timeout = old_timeout
            if csi_bytes is None:
                self._snap_failed += 1
                return  # partial frame — return to top-level loop
            frame = csi_parser.parse_csi_snap_frame(header + csi_bytes)
            if frame:
                if self._music_estimator is not None:
                    self._music_estimator.collect(
                        frame['reporter_id'], frame['peer_id'], frame['csi']
                    )
                if self._snap_callback is not None:
                    self._snap_callback(
                        frame['reporter_id'],
                        frame['peer_id'],
                        frame['snap_seq'],
                        frame['csi'],
                    )
                self._queue.put(('csi_snap', frame))
                self._snap_parsed += 1
            elif frame is None:
                self._snap_failed += 1

        elif b0 == 0xBB:
            b1 = self._ser.read(1)
            if not b1 or b1[0] != 0xDD:
                return
            hdr = self._ser.read(29)
            if len(hdr) < 29:
                return
            csi_len = struct.unpack_from('<H', hdr, 27)[0]
            csi = self._ser.read(csi_len)
            if len(csi) < csi_len:
                return  # short read — drop truncated frame
            frame = csi_parser.parse_shouter_frame(b'\xBB\xDD' + hdr + csi, 0)
            if frame:
                self._queue.put(('shouter', frame))
