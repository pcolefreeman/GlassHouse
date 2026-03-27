# Hardware End-to-End Testing Guide

This document walks through the complete hardware test for the CSI Presence Detection & Zone Localization system. The system uses a **WiFi SoftAP + UDP** architecture where the coordinator runs as an access point and perimeter nodes connect as STA clients.

---

## 1. Architecture Overview (M002 — SoftAP + UDP)

The coordinator runs a WiFi SoftAP (SSID: `CSI_NET`, channel 11). Perimeter nodes connect as STA WiFi clients. All communication uses UDP:
- **Port 4210** — CSI reports from perimeter nodes → coordinator (unicast to 192.168.4.1)
- **Port 4211** — Turn commands from coordinator → all nodes (broadcast to 192.168.4.255); also used for CSI_TX stimulus broadcasts

This replaced the previous ESP-NOW transport layer. Benefits: standard WiFi infrastructure, no peer management, simpler debugging.

---

## 2. Hardware Required

| Item | Qty | Notes |
|------|-----|-------|
| ESP32-WROOM dev board | 3–5 | DevKitC, NodeMCU-32S, or similar. Minimum 3 for S01 two-node test |
| USB Micro/Type-C cable | 1–5 | At least 1 for flashing; 1 stays connected to coordinator during test |
| 5V power source | 2–4 | USB phone chargers or battery banks for the perimeter nodes |
| Arduino IDE | 1 | With **arduino-esp32** board package v2.x or v3.x installed |

## 3. Node Roles

### Full 4-Node Deployment

```
        A ─────────── B           ← top edge (~5m)
        │             │
   ~8m  │  detection  │
        │    area     │
        │             │
        C ─────────── D           ← bottom edge
```

### S01 Two-Node Test (Minimum Viable)

```
   Coordinator (SoftAP) ← USB serial to PC
        │
   ┌────┴────┐
   A         B          ← 2 perimeter nodes, ~3m apart
```

| Node | Role | Firmware | Factory MAC |
|------|------|----------|-------------|
| **Coordinator** | SoftAP master, USB serial to PC | `coordinator/coordinator.ino` | `68:FE:71:90:66:A8` |
| **Node A** (ID 0) | Perimeter — STA client, top-left | `perimeter_node/perimeter_node.ino` | `68:FE:71:90:68:14` |
| **Node B** (ID 1) | Perimeter — STA client, top-right | `perimeter_node/perimeter_node.ino` | `68:FE:71:90:6B:90` |
| **Node C** (ID 2) | Perimeter — STA client, bottom-left | `perimeter_node/perimeter_node.ino` | `68:FE:71:90:60:A0` |
| **Node D** (ID 3) | Perimeter — STA client, bottom-right | `perimeter_node/perimeter_node.ino` | `20:E7:C8:EC:F5:DC` |

---

## 4. Flash the Firmware

### 4.1 Arduino IDE Setup

1. Open Arduino IDE
2. Go to **Tools → Board → ESP32 Arduino → ESP32 Dev Module**
3. Set **Upload Speed** to `921600`
4. Set **Flash Frequency** to `80MHz`
5. Make sure the **arduino-esp32** board package is installed (Board Manager → search "esp32")

### 4.2 Flash Coordinator

1. Connect the coordinator ESP32 via USB
2. Open `coordinator/coordinator.ino` in Arduino IDE
3. Select the correct COM port (**Tools → Port**)
4. Click **Upload**
5. After upload, open **Serial Monitor** at **921600 baud** — you should see:
   ```
   === Coordinator Starting (SoftAP + UDP) ===
   SoftAP SSID: CSI_NET
   SoftAP IP: 192.168.4.1
   ```
6. **Leave this board connected to the PC via USB** — it stays connected during the entire test

### 4.3 Flash Perimeter Nodes (one at a time)

Each perimeter node uses the same firmware file but with a different `NODE_ID`. You must edit one line before each flash:

**For Node A:**
1. Open `perimeter_node/perimeter_node.ino`
2. Find `#define NODE_ID  1` (near the top, around line 42)
3. Change to `#define NODE_ID  0`
4. Connect the Node A ESP32 via USB, select its COM port, click **Upload**

**For Node B:**
1. Change to: `#define NODE_ID  1`
2. Connect the Node B ESP32, select its COM port, click **Upload**

**For Node C:**
1. Change to: `#define NODE_ID  2`
2. Connect Node C, select port, click **Upload**

**For Node D:**
1. Change to: `#define NODE_ID  3`
2. Connect Node D, select port, click **Upload**

> **Tip:** Label each board with tape (A, B, C, D) after flashing so you don't mix them up during placement.

### 4.4 Verify Flashing

After flashing perimeter nodes, power them on and check the coordinator's Serial Monitor. You should see:
1. `# STA connected to AP` messages as each node joins the WiFi network
2. CSI_DATA CSV lines once the round-robin starts:

```
CSI_DATA,42,A,B,AB,-45,128,0 12 -5 8 ...
CSI_DATA,43,B,A,AB,-44,128,1 -3 7 ...
```

Each line = one CSI measurement. The 5th field is the link ID (AB, AC, etc.). If you see link IDs cycling through, the network is working.

> **Note (M002/S03 — CSI Config Improvement):** The perimeter node firmware now has `htltf_en=true` and `stbc_htltf2_en=true` (previously both false), and `ltf_merge_en=false` (previously true). This enables HT-LTF capture for better subcarrier resolution and preserves per-frame CSI variation needed for presence detection. The Serial Monitor will show `CSI config (LLTF+HT-LTF)` on boot to confirm the new configuration.

---

## 5. M002 S01 Two-Node Test Procedure

This minimal test validates the AP+UDP architecture with just 3 boards: coordinator + Node A + Node B.

### 5.1 Setup

1. Flash the coordinator (Section 4.2)
2. Flash Node A with `NODE_ID 0` and Node B with `NODE_ID 1` (Section 4.3)
3. Power on all 3 boards. Place Node A and Node B ~3m apart.
4. Connect coordinator to PC via USB, open Serial Monitor at 921600 baud.

### 5.2 Expected Output

Within a few seconds of all nodes powering on:

1. Coordinator shows `# STA connected to AP` (twice — once for each perimeter node)
2. CSI_DATA lines appear with link_id `AB`:
   ```
   CSI_DATA,0,A,B,AB,-42,128,3 -7 11 ...
   CSI_DATA,1,B,A,AB,-39,128,1 5 -2 ...
   ```
3. Cycle heartbeat appears every ~5 seconds: `# cycle=25 seq=...`

### 5.3 Pass Criteria

| Check | Expected |
|-------|----------|
| Both STAs connect to AP | Two `# STA connected to AP` messages |
| CSI_DATA lines appear | Continuous CSV output with `AB` link_id |
| RSSI values reasonable | Between -30 and -80 dBm at 3m distance |
| No crashes for 2 minutes | Steady output, no reboots |

### 5.4 Troubleshooting

- **No `STA connected` messages:** Verify SSID/password match (`CSI_NET` / `csi12345`). Power cycle nodes.
- **CSI_DATA lines missing:** Nodes may need 5–10 seconds to associate. Check that both nodes show "Connected!" on their own serial output.
- **Only one direction (e.g., A→B but not B→A):** Verify both nodes have correct `NODE_ID` (0 for A, 1 for B).

---

## 6. Physical Setup (Full 4-Node)

1. Place the 4 perimeter nodes at the corners of a rectangular area, roughly **5m × 8m**
   - Node A = top-left
   - Node B = top-right
   - Node C = bottom-left
   - Node D = bottom-right
2. Mount nodes at consistent height (~1m above ground, table height works)
3. Keep the coordinator anywhere convenient — it just needs USB to the PC. It does not need line-of-sight to perimeter nodes (WiFi AP range is ~50–100m indoors)
4. Power on all 4 perimeter nodes
5. Connect the coordinator to the PC via USB

---

## 7. Install Python Dependencies

```bash
cd python/
pip install -r requirements.txt
```

This installs all required packages:
- `pyserial` — serial port communication
- `pygame-ce` — GUI rendering
- `numpy` — numerical computation (CSI feature extraction)

> **Note (M002/S03):** `numpy>=1.24` is now listed in `requirements.txt`. You no longer need to install it separately.

---

## 8. Run the GUI

```bash
cd python/
python main_gui.py --port COM3
```

Replace `COM3` with your actual coordinator port:
- **Windows:** Check Device Manager → Ports (COM & LPT) for the COM number
- **Linux:** Usually `/dev/ttyUSB0` or `/dev/ttyACM0`
- **macOS:** Usually `/dev/cu.usbserial-*` or `/dev/cu.SLAB_USBtoUART`

### Optional flags:

```bash
python main_gui.py --port COM3 --threshold 0.01 --window 30
```

| Flag | Default | What it does |
|------|---------|--------------|
| `--threshold` | `0.005` | Variance threshold for motion detection. Higher = less sensitive |
| `--window` | `20` | Sample window size. Larger = smoother but slower response |
| `--baud` | `921600` | Baud rate (must match coordinator firmware) |

---

## 9. Full Test Procedure (4-Node)

### Test 1: Empty Room Baseline (2 minutes)

1. Start the GUI with nobody in the detection area
2. **Expected:** GUI shows **EMPTY** status, all links show **idle** in the sidebar
3. Watch for 2 minutes — the status should remain EMPTY with no false triggers
4. ✅ **Pass** if: Status stays EMPTY for 2 minutes with no one present

### Test 2: Presence Detection — Enter Area

1. Walk into the center of the detection area
2. **Expected:** Within ~1 second, the GUI should change from **EMPTY** to **OCCUPIED**
3. Stand still for 10 seconds — status should remain OCCUPIED (even small body movements disturb CSI)
4. ✅ **Pass** if: GUI shows OCCUPIED within a few seconds of entering

### Test 3: Zone Tracking — Move Between Quadrants

The detection area is divided into 4 zones:

```
    Q1 (top-left)    │   Q2 (top-right)
  ────────────────────┼────────────────────
    Q3 (bottom-left)  │   Q4 (bottom-right)
```

1. Stand in the **top-left quadrant** (near Node A) for 10 seconds
   - **Expected:** GUI highlights **Q1** zone
2. Walk to the **top-right quadrant** (near Node B), wait 10 seconds
   - **Expected:** GUI highlights **Q2** zone
3. Walk to the **bottom-right quadrant** (near Node D), wait 10 seconds
   - **Expected:** GUI highlights **Q4** zone
4. Walk to the **bottom-left quadrant** (near Node C), wait 10 seconds
   - **Expected:** GUI highlights **Q3** zone
5. ✅ **Pass** if: Zone detection correctly identifies at least 3 out of 4 quadrants

> **Note:** Zone detection uses link disturbance patterns. Standing near a corner where two links converge gives the strongest signal. The center of the area may show ambiguous zone results — that's expected for v1.

### Test 4: Presence Detection — Leave Area

1. Walk completely out of the detection area
2. **Expected:** Within 5–10 seconds, the GUI should return to **EMPTY**
3. All link states should return to **idle**
4. ✅ **Pass** if: GUI returns to EMPTY after leaving

### Test 5: Stability Run (10 minutes)

1. Leave the GUI running for 10 minutes
2. Periodically walk in and out of the area (3–4 times)
3. **Expected:**
   - No Python crashes or exceptions
   - No GUI freezes
   - FPS counter stays above 0 (visible in GUI)
   - Detection continues to work throughout
4. ✅ **Pass** if: No crashes, GUI remains responsive for 10 minutes

---

## 10. Troubleshooting

### No serial data / GUI shows nothing

- Check COM port is correct (`--port` flag)
- Check baud rate matches firmware (default: 921600)
- Verify coordinator Serial Monitor shows CSV lines
- Try unplugging and reconnecting the coordinator USB

### Nodes not connecting to AP

- Verify SSID matches: coordinator defines `CSI_NET`, perimeter nodes define `CSI_NET`
- Verify password matches: both use `csi12345`
- All nodes use WiFi channel 11 (hardcoded in firmware)
- Coordinator serial should show `# STA connected to AP` when a node joins
- Power cycle the node if it shows "WiFi connect timeout"

### False positives (OCCUPIED when nobody present)

- Try increasing the threshold: `--threshold 0.01` or `--threshold 0.02`
- Ensure no fans, moving objects, or pets in the detection area
- Metal furniture close to nodes can cause reflections

### Detection is too slow

- Try decreasing the window size: `--window 10`
- Smaller window = faster response but noisier readings

### Zone detection seems wrong

- Verify node placement matches the diagram (A=top-left, B=top-right, C=bottom-left, D=bottom-right)
- Zone detection works best when standing near a corner, not in the exact center
- Make sure all 6 link IDs appear in the coordinator output

---

## 11. Recording Results

After completing all tests, fill in this summary:

| Test | Result | Notes |
|------|--------|-------|
| S01: Two-Node AP+UDP | PASS / FAIL | AB link_id visible, both STAs connected |
| 1. Empty Room Baseline | PASS / FAIL | |
| 2. Presence Detection — Enter | PASS / FAIL | |
| 3. Zone Tracking — Quadrants | PASS / FAIL | Quadrants correctly identified: Q1 Q2 Q3 Q4 |
| 4. Presence Detection — Leave | PASS / FAIL | |
| 5. Stability Run (10 min) | PASS / FAIL | |

**Overall:** PASS / FAIL

**Tester:** _______________  
**Date:** _______________  
**Notes:** _______________

---

## 12. Cross-Component Verification

Run the automated verification script to check firmware data contracts:

```bash
python scripts/verify_m002_s01.py
```

This validates 11 categories across coordinator, perimeter node, and Python parser: WiFi mode, AP credentials, UDP ports, no ESP-NOW, message types, packet structure, CSV format, WiFi channel, baud rate, CSI APIs, and factory MAC addresses.

To run the legacy S02 checks (if using the old ESP-NOW firmware):
```bash
python scripts/verify_s02.py
```

---

## 13. Automated Hardware Data Validation

After confirming serial data is flowing (Section 5 or Section 9), run the automated hardware validation script to check live data quality:

```bash
python scripts/validate_live_serial.py --port COM3 --duration 30
```

Replace `COM3` with your coordinator's serial port. The script collects CSI data for the specified duration (default 30 seconds) and runs 5 checks:

| Check | What it verifies |
|-------|-----------------|
| All 6 link IDs present | AB, AC, AD, BC, BD, CD all seen during collection |
| CSI byte count in [64, 256] | Each frame has a reasonable number of CSI bytes |
| RSSI in [-90, -10] dBm | Signal strength values are in the expected range |
| Frame rate ≥ 1 fps per link | Each link is producing data at an adequate rate |
| Overall frame rate ≥ 5 fps | Aggregate data throughput is sufficient |

### Output

The script prints a structured report with per-check pass/fail status and exits with code **0** if all checks pass, **1** if any check fails.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | (required) | Serial port (e.g. COM3, /dev/ttyUSB0) |
| `--baud` | 921600 | Baud rate (must match coordinator firmware) |
| `--duration` | 30 | Collection duration in seconds |
| `--stdin` | off | Read from stdin instead of serial (for testing with piped/synthetic data) |

### When to use

- **After initial flash:** Run this immediately after flashing all nodes to confirm data is flowing correctly
- **After physical rearrangement:** If you move nodes, run this to verify all links are still producing data
- **Debugging detection issues:** If presence detection seems off, this script checks whether the underlying data quality is the problem
