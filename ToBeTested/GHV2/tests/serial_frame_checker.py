"""Reads the listener Serial stream and validates both frame types."""
import serial, struct, sys

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM3"
ser  = serial.Serial(PORT, 921600, timeout=2)

frames_a = frames_b = misses = 0
for _ in range(400):
    b = ser.read(1)
    if not b: continue
    if b[0] == 0xAA:
        b2 = ser.read(1)
        if not b2 or b2[0] != 0x55: continue
        hdr      = ser.read(20)
        if len(hdr) < 20: break
        poll_seq = struct.unpack_from('<I', hdr, 14)[0]
        csi_len  = struct.unpack_from('<H', hdr, 18)[0]
        ser.read(csi_len)
        print(f"[AA55] poll_seq={poll_seq}  csi_len={csi_len}")
        frames_a += 1
    elif b[0] == 0xBB:
        b2 = ser.read(1)
        if not b2 or b2[0] != 0xDD: continue
        hdr       = ser.read(29)
        if len(hdr) < 29: break
        flags     = hdr[1]
        poll_seq  = struct.unpack_from('<I', hdr, 15)[0]
        csi_len   = struct.unpack_from('<H', hdr, 27)[0]
        ser.read(csi_len)
        tag = "HIT " if flags == 1 else "MISS"
        print(f"[BBDD] {tag}  poll_seq={poll_seq}  csi_len={csi_len}")
        frames_b += 1
        if flags == 0: misses += 1

print(f"\nListener frames: {frames_a}  Shouter frames: {frames_b}  Misses: {misses}")
