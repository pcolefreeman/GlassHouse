import struct
from unittest.mock import MagicMock
from cobs import cobs as cobs_lib

from python.serial_receiver import decode_cobs, SerialReceiver


def test_decode_cobs_simple():
    """COBS decode: [0x01, 0x01] with 0x00 delimiter -> [0x00]."""
    encoded = bytes([0x01, 0x01])
    assert decode_cobs(encoded) == bytes([0x00])


def test_decode_cobs_no_zeros():
    """COBS decode: data with no zero bytes."""
    # "hello" = [0x68, 0x65, 0x6c, 0x6c, 0x6f]
    # COBS encoded: [0x06, 0x68, 0x65, 0x6c, 0x6c, 0x6f]
    encoded = bytes([0x06, 0x68, 0x65, 0x6c, 0x6c, 0x6f])
    assert decode_cobs(encoded) == b"hello"


def test_decode_cobs_link_report():
    """COBS decode a 10-byte link report packet."""
    # Build a link report: type=0x01, node=1, partner=3, var=0.012, state=1, count=20
    raw = struct.pack('<BBBfBH', 0x01, 1, 3, 0.012, 1, 20)
    assert len(raw) == 10
    # COBS encode it (use cobs library as reference)
    encoded = cobs_lib.encode(raw)
    decoded = decode_cobs(encoded)
    assert decoded == raw


def test_serial_receiver_reads_multiple_packets():
    """SerialReceiver yields multiple decoded packets from a byte stream."""
    pkt1 = b'\x01\x01\x03'  # link report bytes
    pkt2 = b'\x01\x02\x04'
    encoded1 = cobs_lib.encode(pkt1)
    encoded2 = cobs_lib.encode(pkt2)
    stream = encoded1 + b'\x00' + encoded2 + b'\x00'

    receiver = SerialReceiver.__new__(SerialReceiver)
    # Mock serial object that returns our stream then blocks
    mock_ser = MagicMock()
    mock_ser.read = MagicMock(side_effect=[stream, b''])
    receiver._ser = mock_ser

    packets = []
    for pkt in receiver.read_packets():
        packets.append(pkt)
        if len(packets) == 2:
            break
    assert packets == [pkt1, pkt2]


def test_corrupted_frame_skipped():
    """Corrupted COBS frame between two valid packets is silently skipped."""
    pkt1 = b'\x01\x01\x03'
    pkt2 = b'\x01\x02\x04'
    encoded1 = cobs_lib.encode(pkt1)
    encoded2 = cobs_lib.encode(pkt2)
    # Corrupted frame: invalid COBS (0xFF repeated — decodes to garbage or raises)
    corrupted = bytes([0xFF, 0xFF, 0xFF])
    stream = encoded1 + b'\x00' + corrupted + b'\x00' + encoded2 + b'\x00'

    receiver = SerialReceiver.__new__(SerialReceiver)
    mock_ser = MagicMock()
    mock_ser.read = MagicMock(side_effect=[stream, b''])
    receiver._ser = mock_ser

    packets = []
    for pkt in receiver.read_packets():
        packets.append(pkt)
        if len(packets) == 2:
            break
    assert packets == [pkt1, pkt2]
