"""serial_io.py — bidirectional CSI data collection (MUSIC-only).

Threads:
  SerialReader — byte stream → frame_queue  (parses all frame types)
  CSVWriter    — frame_queue → 200ms bucket → CSV row
"""
import csv
import datetime
import logging
import math
import os
import queue
import serial
import struct
import threading
import time

from ghv5 import csi_parser
from ghv5.config import ACTIVE_SHOUTER_IDS, BUCKET_MS, BAUD_RATE

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

    def _resync(self):
        """Discard bytes until finding a valid magic pair (0xAA55, 0xBBDD, 0xEEFF).

        Returns the first byte of the magic pair found, or None on timeout/stop.
        The second magic byte is consumed; the caller should proceed to header parsing.
        """
        MAGIC_STARTS = {0xAA: 0x55, 0xBB: 0xDD, 0xEE: 0xFF}
        discarded = 0
        while self._running:
            b = self._ser.read(1)
            if not b:
                return None
            b0 = b[0]
            if b0 in MAGIC_STARTS:
                b1 = self._ser.read(1)
                if b1 and b1[0] == MAGIC_STARTS[b0]:
                    if discarded > 0:
                        _log.debug("[DIAG] resync: discarded %d bytes before magic 0x%02X%02X",
                                   discarded, b0, b1[0])
                    return b0
            discarded += 1
            if discarded > 2048:
                _log.warning("[DIAG] resync: gave up after discarding %d bytes", discarded)
                return None
        return None

    def _dispatch_after_magic(self, b0):
        """Parse a frame whose magic pair has already been consumed by _resync."""
        if b0 == 0xAA:
            self._parse_listener_body()
        elif b0 == 0xEE:
            self._parse_snap_body()
        elif b0 == 0xBB:
            self._parse_shouter_body()

    def _parse_listener_body(self):
        hdr = self._ser.read(20)
        if len(hdr) < 20:
            return
        csi_len = struct.unpack_from('<H', hdr, 18)[0]
        if csi_len > 384:
            self._sync_errors += 1
            magic_byte = self._resync()
            if magic_byte is not None:
                self._dispatch_after_magic(magic_byte)
            return
        csi = self._ser.read(csi_len)
        if len(csi) < csi_len:
            return
        frame = csi_parser.parse_listener_frame(b'\xAA\x55' + hdr + csi, 0)
        if frame:
            self._queue.put(('listener', frame))

    def _parse_snap_body(self):
        header = self._read_exact(6)
        if header is None:
            self._snap_failed += 1
            return
        csi_len = struct.unpack_from('<H', header, 4)[0]
        if csi_len < 1 or csi_len > 384:
            _log.debug("[DIAG] bad csi_len=%d in snap frame, discarding", csi_len)
            self._snap_failed += 1
            self._sync_errors += 1
            magic_byte = self._resync()
            if magic_byte is not None:
                self._dispatch_after_magic(magic_byte)
            return
        old_timeout = self._ser.timeout
        self._ser.timeout = 0.05
        csi_bytes = self._read_exact(csi_len)
        self._ser.timeout = old_timeout
        if csi_bytes is None:
            self._snap_failed += 1
            return
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

    def _parse_shouter_body(self):
        hdr = self._ser.read(29)
        if len(hdr) < 29:
            return
        csi_len = struct.unpack_from('<H', hdr, 27)[0]
        if csi_len > 384:
            self._sync_errors += 1
            magic_byte = self._resync()
            if magic_byte is not None:
                self._dispatch_after_magic(magic_byte)
            return
        csi = self._ser.read(csi_len)
        if len(csi) < csi_len:
            return
        frame = csi_parser.parse_shouter_frame(b'\xBB\xDD' + hdr + csi, 0)
        if frame:
            self._queue.put(('shouter', frame))

    def _read_one_frame(self):
        b = self._ser.read(1)
        if not b:
            return
        b0 = b[0]

        if b0 == 0xAA:
            b1 = self._ser.read(1)
            if not b1 or b1[0] != 0x55:
                return
            self._parse_listener_body()

        elif b0 == 0xEE:
            b1 = self._ser.read(1)
            if not b1 or b1[0] != 0xFF:
                self._sync_errors += 1
                return
            self._parse_snap_body()

        elif b0 == 0xBB:
            b1 = self._ser.read(1)
            if not b1 or b1[0] != 0xDD:
                return
            self._parse_shouter_body()


# ── CSVWriter ──────────────────────────────────────────────────────────────────
class CSVWriter(threading.Thread):
    """Reads frame_queue; correlates (mac, poll_seq) pairs; writes CSV rows on flush."""

    def __init__(self, frame_queue: queue.Queue, output,
                 active_shouter_ids=None, spacing_estimator=None):
        super().__init__(daemon=True, name="CSVWriter")
        self._queue   = frame_queue
        self._output  = output
        self._ids     = active_shouter_ids or ACTIVE_SHOUTER_IDS
        self._names   = csi_parser.build_feature_names(self._ids)
        self._writer  = None
        self._lf_pending = {}   # (mac, poll_seq) → listener frame
        self._sf_pending = {}   # (mac, poll_seq) → shouter frame
        self._mac_to_sid = {}   # mac → shouter_id, built from received shouter frames
        self._spacing = spacing_estimator

    def run(self):
        self._writer = csv.writer(self._output)
        self._writer.writerow(self._names)
        while True:
            item = self._queue.get()
            if item is None:
                break
            ftype, data = item
            if ftype == 'listener':
                key = (data['mac'], data['poll_seq'])
                self._lf_pending[key] = data
                # Do NOT emit here — labels are applied at flush time
            elif ftype == 'shouter':
                key = (data['mac'], data['poll_seq'])
                self._sf_pending[key] = data
                self._mac_to_sid[data['mac']] = data['shouter_id']
                # Do NOT emit here — labels are applied at flush time
            elif ftype == 'flush':
                self._flush_all(data)

    def _flush_all(self, meta: dict):
        """Write all pending (matched or unmatched) as rows, then clear."""
        all_keys = set(self._lf_pending) | set(self._sf_pending)
        for key in all_keys:
            lf = self._lf_pending.pop(key, None)
            sf = self._sf_pending.pop(key, None)
            self._write_row(lf, sf, meta)

    def _write_row(self, lf, sf, meta: dict):
        # Resolve shouter_id so MISS frames (sf=None) are attributed correctly
        if sf is not None:
            sid = sf['shouter_id']
        elif lf is not None:
            sid = self._mac_to_sid.get(lf['mac'])
        else:
            sid = None
        vec = csi_parser.extract_feature_vector(lf, sf, self._names, shouter_id=sid)
        ts  = (lf['timestamp_ms'] if lf else
               sf['listener_ms']  if sf else 0)
        # Fill meta positions (first 5 columns)
        meta_vals = [ts, meta.get('label',''), meta.get('zone_id',''),
                     meta.get('grid_row',''), meta.get('grid_col',''),
                     meta.get('activity', '')]
        for i, v in enumerate(meta_vals):
            vec[i] = v
        self._writer.writerow(vec)


# ── Entry point ────────────────────────────────────────────────────────────────
def build_output_filename(out_dir, width, depth, timestamp=None):
    """Build the CSV output filename, embedding area dimensions when provided.

    Args:
        out_dir:   Directory path for output file.
        width:     Area width in metres, or None.
        depth:     Area depth in metres, or None.
        timestamp: Optional datetime string (YYYY-MM-DD_HHMMSS); generated if None.
    Returns:
        Full absolute path string.
    """
    ts = timestamp or datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if width is not None and depth is not None:
        filename = f"capture_{width:.1f}x{depth:.1f}m_{ts}.csv"
    else:
        filename = f"capture_{ts}.csv"
    return os.path.join(out_dir, filename)


def main():
    import argparse
    from ghv5.spacing_estimator import SpacingEstimator, CSIMUSICEstimator

    parser = argparse.ArgumentParser(description="GHV4 data collection")
    parser.add_argument('--port',   default="COM3")
    parser.add_argument('--output', default="data/processed/capture.csv")
    parser.add_argument('--label',  default='unknown')
    parser.add_argument('--zone',   type=int,   default=0)
    parser.add_argument('--row',    type=int,   default=0)
    parser.add_argument('--col',    type=int,   default=0)
    parser.add_argument('--width',  type=float, default=None,
                        help="Search area width in metres (embeds in filename)")
    parser.add_argument('--depth',  type=float, default=None,
                        help="Search area depth in metres (embeds in filename)")
    parser.add_argument('--duration', type=float, default=0,
                        help="Seconds to run before stopping automatically (0 = unlimited)")
    args = parser.parse_args()

    if not (0 <= args.row <= 2):
        parser.error(f"--row must be 0, 1, or 2 (got {args.row})")
    if not (0 <= args.col <= 2):
        parser.error(f"--col must be 0, 1, or 2 (got {args.col})")
    if args.width is not None and args.width <= 0:
        parser.error(f"--width must be positive (got {args.width})")
    if args.depth is not None and args.depth <= 0:
        parser.error(f"--depth must be positive (got {args.depth})")
    if args.duration < 0:
        parser.error(f"--duration must be >= 0 (got {args.duration})")

    out_dir     = os.path.dirname(os.path.abspath(args.output))
    output_path = build_output_filename(out_dir, args.width, args.depth)
    if args.output != "data/processed/capture.csv" and args.width is not None and args.depth is not None:
        print(f"[GHV4] NOTE: --output filename ignored when --width/--depth given")

    os.makedirs(out_dir, exist_ok=True)
    meta = {'label': args.label, 'zone_id': args.zone,
            'grid_row': args.row, 'grid_col': args.col}

    music_estimator = CSIMUSICEstimator()
    spacing_estimator = SpacingEstimator(
        spacing_path=os.path.join(out_dir, "spacing.json"),
        music_estimator=music_estimator,
    )
    spacing_estimator.start()

    frame_queue = queue.Queue()
    ser = serial.Serial(args.port, BAUD_RATE, timeout=1)
    print(f"[GHV4] {args.port}  →  {output_path}")
    print(f"[GHV4] label={args.label}  zone={args.zone}  row={args.row}  col={args.col}")
    if args.width is not None and args.depth is not None:
        print(f"[GHV4] area={args.width:.1f}m × {args.depth:.1f}m")
    if args.duration > 0:
        print(f"[GHV4] auto-stop after {args.duration:.0f}s")
    print("[GHV4] Press Ctrl+C to stop\n")

    with open(output_path, 'w', newline='') as f_out:
        reader = SerialReader(ser, frame_queue, music_estimator=music_estimator)
        writer = CSVWriter(frame_queue, f_out, spacing_estimator=spacing_estimator)
        reader.start()
        writer.start()
        start_time = time.time()
        try:
            while True:
                time.sleep(BUCKET_MS / 1000.0)
                frame_queue.put(('flush', dict(meta)))
                if args.duration > 0 and (time.time() - start_time) >= args.duration:
                    print(f"\n[GHV4] Duration elapsed ({args.duration:.0f}s). Stopping.")
                    break
        except KeyboardInterrupt:
            print("\n[GHV4] Stopping…")
        finally:
            reader.stop()
            frame_queue.put(('flush', dict(meta)))  # flush final partial bucket
            frame_queue.put(None)
            writer.join(timeout=2)
            ser.close()
    print(f"[GHV4] Saved to {output_path}")

if __name__ == '__main__':
    main()
