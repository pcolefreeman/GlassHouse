import glob
import os

WATCH_DIR = r"C:\Users\19124\OneDrive\Documents\Senior_Cap\GitRepo\GlassHouse\CSIBin3_3"

# Find the most recent .bin file
files = sorted(glob.glob(os.path.join(WATCH_DIR, "*.bin")))
if not files:
    print("No .bin files found!")
else:
    filepath = files[-1]
    print(f"Inspecting: {os.path.basename(filepath)}")

    with open(filepath, "rb") as f:
        raw = f.read()

    print(f"File size: {len(raw)} bytes")
    print(f"\nFirst 64 bytes (hex):")
    print(raw[:64].hex(' '))
    print(f"\nSearching for 0xAA 0x55 magic bytes...")
    
    count = 0
    for i in range(len(raw) - 1):
        if raw[i] == 0xAA and raw[i+1] == 0x55:
            print(f"  Found at byte {i} — next 15 bytes: {raw[i+2:i+17].hex(' ')}")
            count += 1
            if count >= 5:  # show first 5 hits
                break

    if count == 0:
        print("  !! No 0xAA 0x55 found in file at all !!")
        print(f"\nMost common byte values:")
        from collections import Counter
        c = Counter(raw)
        for byte, freq in c.most_common(10):
            print(f"  0x{byte:02X} = {freq} times")