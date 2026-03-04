import struct

INPUT_FILE = r"C:\Users\19124\OneDrive\Documents\Senior_Cap\GitRepo\GlassHouse\TestcsiCapture\csi_20260303_162815.bin"

MAGIC = bytes([0xAA, 0x55])

def crc16_ccitt(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc

def parse_csi(csi_bytes):
    csi = []
    for j in range(0, len(csi_bytes) - 1, 2):
        imag = struct.unpack('b', bytes([csi_bytes[j]]))[0]
        real = struct.unpack('b', bytes([csi_bytes[j+1]]))[0]
        csi.append(complex(real, imag))
    return csi

frames = []

with open(INPUT_FILE, "rb") as f:
    raw = f.read()

i = 0
while i < len(raw) - 2:
    if raw[i] == 0xAA and raw[i+1] == 0x55:
        offset = i + 2

        if offset + 15 > len(raw):
            break

        ver       = raw[offset]
        flags     = raw[offset+1]
        timestamp = struct.unpack_from('<I', raw, offset+2)[0]
        rssi      = struct.unpack_from('<b', raw, offset+6)[0]
        mac       = raw[offset+7:offset+13]
        csi_len   = struct.unpack_from('<H', raw, offset+13)[0]

        offset += 15

        if offset + csi_len + 2 > len(raw):
            i += 1
            continue

        csi_bytes = raw[offset:offset+csi_len]
        offset += csi_len

        crc_received = struct.unpack_from('<H', raw, offset)[0]
        crc_calc = crc16_ccitt(raw[i+2:i+2+15])
        crc_calc ^= crc16_ccitt(csi_bytes)

        mac_str = ':'.join(f'{b:02X}' for b in mac)

        # Parse CSI into complex numbers
        csi_complex = parse_csi(csi_bytes)

        frame = {
            'timestamp_ms': timestamp,
            'rssi_dbm':     rssi,
            'mac':          mac_str,
            'csi_len':      csi_len,
            'csi_complex':  csi_complex,   # list of complex numbers
            'crc_ok':       crc_calc == crc_received
        }
        frames.append(frame)

        print(f"[{timestamp}ms] MAC={mac_str} RSSI={rssi}dBm "
              f"Subcarriers={len(csi_complex)} CRC={'OK' if frame['crc_ok'] else 'FAIL'}")

        i = offset + 2
    else:
        i += 1

print(f"\nTotal frames parsed: {len(frames)}")

# ---- Inspect ALL frames' CSI ----
for frame_idx, frame in enumerate(frames):
    print(f"\n--- Frame {frame_idx} | [{frame['timestamp_ms']}ms] "
          f"MAC={frame['mac']} RSSI={frame['rssi_dbm']}dBm "
          f"CRC={'OK' if frame['crc_ok'] else 'FAIL'} ---")
    for idx, c in enumerate(frame['csi_complex']):
        amplitude = abs(c)
        print(f"  Subcarrier {idx:3d}: real={c.real:6.1f}  imag={c.imag:6.1f}  amplitude={amplitude:6.2f}")