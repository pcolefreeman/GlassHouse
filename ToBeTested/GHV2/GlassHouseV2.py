"""GlassHouseV2.py — bidirectional CSI data collection.

Threads:
  SerialReader — byte stream → frame_queue  (parses [0xAA][0x55] and [0xBB][0xDD])
  CSVWriter    — frame_queue → 200ms bucket → CSV row
"""
import csv
import math
import os
import queue
import serial
import struct
import threading
import time
import csi_parser

# ── Configuration ──────────────────────────────────────────────────────────────
SERIAL_PORT        = "COM3"
BAUD_RATE          = 921600
ACTIVE_SHOUTER_IDS = csi_parser.ACTIVE_SHOUTER_IDS
BUCKET_MS          = csi_parser.BUCKET_MS
OUTPUT_CSV         = "data/processed/capture.csv"


# ── SerialReader ───────────────────────────────────────────────────────────────
class SerialReader(threading.Thread):
    """Reads a byte stream, detects magic sequences, parses frames, enqueues them."""

    def __init__(self, ser, frame_queue: queue.Queue):
        super().__init__(daemon=True, name="SerialReader")
        self._ser     = ser
        self._queue   = frame_queue
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            self._read_one_frame()

    def stop(self):
        self._running = False

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


# ── CSVWriter ──────────────────────────────────────────────────────────────────
class CSVWriter(threading.Thread):
    """Reads frame_queue; correlates (mac, poll_seq) pairs; writes CSV rows on flush."""

    def __init__(self, frame_queue: queue.Queue, output,
                 active_shouter_ids=None):
        super().__init__(daemon=True, name="CSVWriter")
        self._queue   = frame_queue
        self._output  = output
        self._ids     = active_shouter_ids or ACTIVE_SHOUTER_IDS
        self._names   = csi_parser.build_feature_names(self._ids)
        self._writer  = None
        self._lf_pending = {}   # (mac, poll_seq) → listener frame
        self._sf_pending = {}   # (mac, poll_seq) → shouter frame

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
        # extract_feature_vector handles lf=None or sf=None gracefully
        vec = csi_parser.extract_feature_vector(lf, sf, self._names)
        ts  = (lf['timestamp_ms'] if lf else
               sf['listener_ms']  if sf else 0)
        # Fill meta positions (first 5 columns)
        meta_vals = [ts, meta.get('label',''), meta.get('zone_id',''),
                     meta.get('grid_row',''), meta.get('grid_col','')]
        for i, v in enumerate(meta_vals):
            vec[i] = v
        self._writer.writerow(vec)


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="GHV2 data collection")
    parser.add_argument('--port',   default=SERIAL_PORT)
    parser.add_argument('--output', default=OUTPUT_CSV)
    parser.add_argument('--label',  default='unknown')
    parser.add_argument('--zone',   type=int, default=0)
    parser.add_argument('--row',    type=int, default=0)
    parser.add_argument('--col',    type=int, default=0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    meta = {'label': args.label, 'zone_id': args.zone,
            'grid_row': args.row, 'grid_col': args.col}

    frame_queue = queue.Queue()
    ser = serial.Serial(args.port, BAUD_RATE, timeout=1)
    print(f"[GHV2] {args.port}  →  {args.output}")
    print(f"[GHV2] label={args.label}  zone={args.zone}  row={args.row}  col={args.col}")
    print("[GHV2] Press Ctrl+C to stop\n")

    with open(args.output, 'w', newline='') as f_out:
        reader = SerialReader(ser, frame_queue)
        writer = CSVWriter(frame_queue, f_out)
        reader.start()
        writer.start()
        try:
            while True:
                time.sleep(BUCKET_MS / 1000.0)
                frame_queue.put(('flush', dict(meta)))  # copy: avoid aliasing issues
        except KeyboardInterrupt:
            print("\n[GHV2] Stopping…")
        finally:
            reader.stop()
            frame_queue.put(None)
            writer.join(timeout=2)
            ser.close()
    print(f"[GHV2] Saved to {args.output}")

if __name__ == '__main__':
    main()
