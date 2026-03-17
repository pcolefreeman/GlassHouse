"""log_listener.py — Capture listener serial output to a human-readable log file.

Usage:
    python log_listener.py              # uses COM3, logs 60 s
    python log_listener.py COM4         # custom port
    python log_listener.py COM3 120     # custom port + duration (seconds)

Output files (written to current directory):
    listener_log_HHMMSS.txt   — human-readable: parsed frames + text lines
    listener_raw_HHMMSS.bin   — raw bytes (for hex inspection)
"""
import sys
import struct
import time
import serial

PORT     = sys.argv[1] if len(sys.argv) > 1 else 'COM3'
DURATION = int(sys.argv[2]) if len(sys.argv) > 2 else 60
BAUD     = 921600

ts = time.strftime("%H%M%S")
txt_path = f"listener_log_{ts}.txt"
bin_path = f"listener_raw_{ts}.bin"

# ── Frame counters ─────────────────────────────────────────────────────────────
counts = {'AA55': 0, 'BBDD': 0, 'CCDD': 0, 'EEFF': 0, 'text': 0, 'dropped': 0}


def parse_cc_dd(payload: bytes) -> str:
    """Parse [CC][DD] payload (12 bytes after magic)."""
    if len(payload) < 12:
        return f"  [CC][DD] SHORT payload ({len(payload)} bytes)"
    ver, reporter_id = struct.unpack_from('<BB', payload, 0)
    peer_rssi        = list(struct.unpack_from('<5b', payload, 2))
    peer_count       = list(struct.unpack_from('<5B', payload, 7))
    parts = []
    for i in range(1, 5):
        if peer_count[i] > 0:
            parts.append(f"  peer{i}: rssi={peer_rssi[i]} dBm  count={peer_count[i]}")
    body = '\n'.join(parts) if parts else "  (no peer data)"
    return f"  [CC][DD] reporter={reporter_id}\n{body}"


def parse_ee_ff(header: bytes, csi: bytes) -> str:
    """Parse [EE][FF] header (6 bytes after magic) + csi."""
    if len(header) < 6:
        return f"  [EE][FF] SHORT header ({len(header)} bytes)"
    ver, reporter, peer, seq, csi_len = struct.unpack_from('<BBBBH', header)
    return (f"  [EE][FF] reporter={reporter}  peer={peer}  seq={seq}"
            f"  csi_len={csi_len}  actual={len(csi)}")


print(f"Logging {PORT} at {BAUD} baud for {DURATION}s")
print(f"  Text log : {txt_path}")
print(f"  Raw bytes: {bin_path}")
print("  Ctrl+C to stop early\n")

try:
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
except serial.SerialException as e:
    print(f"ERROR: Could not open {PORT}: {e}")
    print("Make sure Arduino Serial Monitor is closed.")
    sys.exit(1)

start = time.time()
buf = bytearray()

with open(txt_path, 'w', encoding='utf-8', errors='replace') as ftxt, \
     open(bin_path, 'wb') as fbin:

    ftxt.write(f"GHV3 Listener Log — {PORT} — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    ftxt.write("=" * 60 + "\n\n")

    def log(msg: str):
        ftxt.write(msg + "\n")
        ftxt.flush()
        print(msg)

    try:
        while time.time() - start < DURATION:
            chunk = ser.read(256)
            if not chunk:
                continue
            fbin.write(chunk)
            fbin.flush()
            buf.extend(chunk)

            while len(buf) >= 2:
                b0, b1 = buf[0], buf[1]

                # ── [AA][55] listener CSI frame ───────────────────────────────
                if b0 == 0xAA and b1 == 0x55:
                    if len(buf) < 2 + 20:
                        break
                    hdr = buf[2:22]
                    csi_len = struct.unpack_from('<H', hdr, 18)[0]
                    need = 2 + 20 + csi_len
                    if len(buf) < need:
                        break
                    counts['AA55'] += 1
                    log(f"[AA][55] #{counts['AA55']:4d}  csi_len={csi_len}")
                    del buf[:need]

                # ── [BB][DD] shouter poll frame ───────────────────────────────
                elif b0 == 0xBB and b1 == 0xDD:
                    if len(buf) < 2 + 29:
                        break
                    hdr = buf[2:31]
                    csi_len = struct.unpack_from('<H', hdr, 27)[0]
                    need = 2 + 29 + csi_len
                    if len(buf) < need:
                        break
                    flags, sid = hdr[1], hdr[4]
                    hit = bool(flags & 0x01)
                    counts['BBDD'] += 1
                    log(f"[BB][DD] #{counts['BBDD']:4d}  sid={sid}  hit={hit}  csi_len={csi_len}")
                    del buf[:need]

                # ── [CC][DD] ranging frame ────────────────────────────────────
                elif b0 == 0xCC and b1 == 0xDD:
                    need = 2 + 12
                    if len(buf) < need:
                        break
                    payload = bytes(buf[2:14])
                    counts['CCDD'] += 1
                    log(f"[CC][DD] #{counts['CCDD']:4d}")
                    log(parse_cc_dd(payload))
                    del buf[:need]

                # ── [EE][FF] MUSIC CSI snap frame ─────────────────────────────
                elif b0 == 0xEE and b1 == 0xFF:
                    if len(buf) < 2 + 6:
                        break
                    header = bytes(buf[2:8])
                    csi_len = struct.unpack_from('<H', header, 4)[0]
                    need = 2 + 6 + csi_len
                    if len(buf) < need:
                        break
                    csi = bytes(buf[8:8 + csi_len])
                    counts['EEFF'] += 1
                    log(parse_ee_ff(header, csi))
                    del buf[:need]

                # ── Text line ─────────────────────────────────────────────────
                elif b0 < 0x80:
                    # Collect until newline
                    nl = buf.find(b'\n')
                    if nl == -1:
                        if len(buf) > 256:
                            del buf[:1]  # no newline found, advance
                        break
                    line_bytes = bytes(buf[:nl + 1])
                    del buf[:nl + 1]
                    # Only print lines that look like text
                    try:
                        line = line_bytes.decode('ascii', errors='strict').rstrip()
                        if line:
                            counts['text'] += 1
                            log(f"TEXT: {line}")
                    except UnicodeDecodeError:
                        counts['dropped'] += 1

                # ── Unknown / binary byte ─────────────────────────────────────
                else:
                    counts['dropped'] += 1
                    del buf[:1]

    except KeyboardInterrupt:
        print("\nStopped by user.")

    elapsed = time.time() - start
    summary = (
        f"\n{'='*60}\n"
        f"SUMMARY after {elapsed:.1f}s on {PORT}\n"
        f"  [AA][55] listener frames : {counts['AA55']}\n"
        f"  [BB][DD] shouter frames  : {counts['BBDD']}\n"
        f"  [CC][DD] ranging frames  : {counts['CCDD']}\n"
        f"  [EE][FF] MUSIC snap frms : {counts['EEFF']}\n"
        f"  Text lines               : {counts['text']}\n"
        f"  Dropped bytes            : {counts['dropped']}\n"
    )
    log(summary)

ser.close()
print(f"\nDone. Log saved to: {txt_path}")
print(f"If [EE][FF] count is 0, MUSIC snapshots are not arriving at the PC.")
print(f"If [CC][DD] count is 0, ranging reports are not arriving at the PC.")
