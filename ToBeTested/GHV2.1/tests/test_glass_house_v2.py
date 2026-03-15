import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import io, queue, struct
import csi_parser
import GlassHouseV2
from conftest import make_listener_frame, make_shouter_frame, MAC_DEFAULT

def mock_serial(data: bytes):
    stream = io.BytesIO(data)
    return type('MockSerial', (), {'read': lambda self, n: stream.read(n)})()

# ── SerialReader ──────────────────────────────────────────────────────────────

def test_serial_reader_enqueues_listener_frame():
    raw = make_listener_frame(poll_seq=1, mac=MAC_DEFAULT)
    q   = queue.Queue()
    r   = GlassHouseV2.SerialReader(mock_serial(raw), q)
    r._read_one_frame()
    assert not q.empty()
    ftype, frame = q.get_nowait()
    assert ftype == 'listener'
    assert frame['poll_seq'] == 1

def test_serial_reader_enqueues_shouter_frame():
    raw = make_shouter_frame(poll_seq=2, mac=MAC_DEFAULT)
    q   = queue.Queue()
    r   = GlassHouseV2.SerialReader(mock_serial(raw), q)
    r._read_one_frame()
    assert not q.empty()
    ftype, frame = q.get_nowait()
    assert ftype == 'shouter'
    assert frame['poll_seq'] == 2

def test_serial_reader_skips_noise_bytes():
    noise = b'\x00\xFF\x12\x34'
    raw   = noise + make_listener_frame(poll_seq=5, mac=MAC_DEFAULT)
    q     = queue.Queue()
    r     = GlassHouseV2.SerialReader(mock_serial(raw), q)
    for _ in range(len(raw)):
        r._read_one_frame()
        if not q.empty(): break
    assert not q.empty()
    _, frame = q.get_nowait()
    assert frame['poll_seq'] == 5

# ── CSVWriter ──────────────────────────────────────────────────────────────────

import csv as _csv, io as _io, math as _math

def _parse_csv(writer_output_io) -> list:
    writer_output_io.seek(0)
    return list(_csv.DictReader(writer_output_io))

def _make_parsed_pair(poll_seq, mac=MAC_DEFAULT):
    csi = bytes(range(256)) + bytes(128)
    lf  = csi_parser.parse_listener_frame(
            make_listener_frame(poll_seq=poll_seq, mac=mac, csi_bytes=csi), 0)
    sf  = csi_parser.parse_shouter_frame(
            make_shouter_frame(poll_seq=poll_seq, mac=mac, csi_bytes=csi,
                               shouter_id=1), 0)
    return lf, sf

def test_csv_writer_writes_one_row():
    q   = queue.Queue()
    out = _io.StringIO()
    lf, sf = _make_parsed_pair(1)
    q.put(('listener', lf))
    q.put(('shouter',  sf))
    q.put(('flush', {'label': 'zone_A', 'zone_id': 1, 'grid_row': 0, 'grid_col': 0}))
    q.put(None)
    GlassHouseV2.CSVWriter(q, out, active_shouter_ids=[1]).run()
    rows = _parse_csv(out)
    assert len(rows) == 1
    assert rows[0]['label'] == 'zone_A'
    assert rows[0]['zone_id'] == '1'

def test_csv_writer_miss_shouter_columns_are_nan():
    """No shouter frame → tx columns are NaN."""
    q   = queue.Queue()
    out = _io.StringIO()
    lf, _ = _make_parsed_pair(2)
    q.put(('listener', lf))
    q.put(('flush', {'label': 'test', 'zone_id': 0, 'grid_row': 0, 'grid_col': 0}))
    q.put(None)
    GlassHouseV2.CSVWriter(q, out, active_shouter_ids=[1]).run()
    rows = _parse_csv(out)
    assert len(rows) == 1
    assert rows[0]['s1_tx_amp_0'] == 'nan'

def test_csv_writer_correct_column_count():
    q   = queue.Queue()
    out = _io.StringIO()
    lf, sf = _make_parsed_pair(3)
    q.put(('listener', lf))
    q.put(('shouter',  sf))
    q.put(('flush', {'label': '', 'zone_id': 0, 'grid_row': 0, 'grid_col': 0}))
    q.put(None)
    GlassHouseV2.CSVWriter(q, out, active_shouter_ids=[1]).run()
    out.seek(0)
    header = out.readline().strip().split(',')
    assert len(header) == 1287  # 5 meta + 1 shouter × 2 directions × 641
