"""COBS-framed serial receiver for GlassHouse v2 coordinator link."""

from __future__ import annotations

from typing import Generator

import serial
from cobs import cobs as cobs_codec


def decode_cobs(data: bytes) -> bytes:
    """Decode a COBS-encoded byte string."""
    return cobs_codec.decode(data)


class SerialReceiver:
    """Reads COBS-framed packets from USB serial.

    Each frame: [COBS-encoded bytes] [0x00 delimiter]
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 921600) -> None:
        self._port = port
        self._baud = baud
        self._ser: serial.Serial | None = None

    def open(self) -> None:
        # Open without toggling DTR/RTS — on ESP32-S3 USB-Serial/JTAG,
        # DTR transitions reset the chip, killing the coordinator mid-run.
        ser = serial.Serial()
        ser.port = self._port
        ser.baudrate = self._baud
        ser.timeout = 0.1
        ser.dtr = False
        ser.rts = False
        ser.open()
        self._ser = ser

    def close(self) -> None:
        if self._ser:
            self._ser.close()
            self._ser = None

    def read_packets(self) -> Generator[bytes, None, None]:
        """Yield decoded packets from the serial stream."""
        if self._ser is None:
            self.open()
        buf = bytearray()
        while True:
            chunk = self._ser.read(256)
            if not chunk:
                continue
            buf.extend(chunk)
            while b'\x00' in buf:
                idx = buf.index(b'\x00')
                frame = bytes(buf[:idx])
                buf = buf[idx + 1:]
                if len(frame) == 0:
                    continue  # empty frame, skip
                try:
                    yield decode_cobs(frame)
                except Exception:
                    pass  # corrupted frame, skip to next delimiter
