# Hardware End-to-End Testing Guide

This document walks through the complete hardware test for the CSI Presence Detection & Zone Localization system. You need **5 ESP32-WROOM boards**, a USB cable, and a PC with Python 3.10+.

---

## 1. Hardware Required

| Item | Qty | Notes |
|------|-----|-------|
| ESP32-WROOM dev board | 5 | Any ESP32-WROOM module (DevKitC, NodeMCU-32S, etc.) |
| USB Micro/Type-C cable | 1–5 | At least 1 for flashing; 1 stays connected to coordinator during test |
| 5V power source | 4 | USB phone chargers or battery banks for the perimeter nodes |
| Arduino IDE | 1 | With **arduino-esp32** board package v2.x or v3.x installed |

## 2. Node Roles

```
        A ─────────── B           ← top edge (~5m)
        │             │
   ~8m  │  detection  │
        │    area     │
        │             │
        C ─────────── D           ← bottom edge
```

| Node | Role | Firmware | MAC |
|------|------|----------|-----|
| **Coordinator** | Master orchestrator, USB serial to PC | `coordinator/coordinator.ino` | `24:6F:28:AA:00:00` |
| **Node A** (ID 0) | Perimeter — top-left corner | `perimeter_node/perimeter_node.ino` | `24:6F:28:AA:00:01` |
| **Node B** (ID 1) | Perimeter — top-right corner | `perimeter_node/perimeter_node.ino` | `24:6F:28:AA:00:02` |
| **Node C** (ID 2) | Perimeter — bottom-left corner | `perimeter_node/perimeter_node.ino` | `24:6F:28:AA:00:03` |
| **Node D** (ID 3) | Perimeter — bottom-right corner | `perimeter_node/perimeter_node.ino` | `24:6F:28:AA:00:04` |

---

## 3. Flash the Firmware

### 3.1 Arduino IDE Setup

1. Open Arduino IDE
2. Go to **Tools → Board → ESP32 Arduino → ESP32 Dev Module**
3. Set **Upload Speed** to `921600`
4. Set **Flash Frequency** to `80MHz`
5. Make sure the **arduino-esp32** board package is installed (Board Manager → search "esp32")

### 3.2 Flash Coordinator

1. Connect the coordinator ESP32 via USB
2. Open `coordinator/coordinator.ino` in Arduino IDE
3. Select the correct COM port (**Tools → Port**)
4. Click **Upload**
5. After upload, open **Serial Monitor** at **921600 baud** — you should see periodic output once perimeter nodes are active
6. **Leave this board connected to the PC via USB** — it stays connected during the entire test

### 3.3 Flash Perimeter Nodes (one at a time)

Each perimeter node uses the same firmware file but with a different `NODE_ID`. You must edit one line before each flash:

**For Node A:**
1. Open `perimeter_node/perimeter_node.ino`
2. Find line 42: `#define NODE_ID  0`
3. Verify it says `0` (Node A)
4. Connect the Node A ESP32 via USB, select its COM port, click **Upload**

**For Node B:**
1. Change line 42 to: `#define NODE_ID  1`
2. Connect the Node B ESP32, select its COM port, click **Upload**

**For Node C:**
1. Change line 42 to: `#define NODE_ID  2`
2. Connect Node C, select port, click **Upload**

**For Node D:**
1. Change line 42 to: `#define NODE_ID  3`
2. Connect Node D, select port, click **Upload**

> **Tip:** Label each board with tape (A, B, C, D) after flashing so you don't mix them up during placement.

### 3.4 Verify Flashing

After flashing all 4 perimeter nodes, power them all on (USB charger/battery) and check the coordinator's Serial Monitor. You should see CSV lines like:

```
CSI_DATA,42,A,B,AB,-45,128,0 12 -5 8 ...
CSI_DATA,42,A,C,AC,-52,128,3 -7 11 ...
CSI_DATA,42,A,D,AD,-48,128,-2 9 4 ...
CSI_DATA,43,B,A,BA,-44,128,1 -3 7 ...
```

Each line = one CSI measurement. The 5th field is the link ID (AB, AC, AD, BA, BC, etc.). If you see all 6 link IDs cycling through, the network is working.

---

## 4. Physical Setup

1. Place the 4 perimeter nodes at the corners of a rectangular area, roughly **5m × 8m**
   - Node A = top-left
   - Node B = top-right
   - Node C = bottom-left
   - Node D = bottom-right
2. Mount nodes at consistent height (~1m above ground, table height works)
3. Keep the coordinator anywhere convenient — it just needs USB to the PC. It doesn't need line-of-sight to perimeter nodes (ESP-NOW range is ~50–100m indoors)
4. Power on all 4 perimeter nodes
5. Connect the coordinator to the PC via USB

---

## 5. Install Python Dependencies

```bash
cd python/
pip install -r requirements.txt
```

This installs:
- `pyserial` — serial port communication
- `pygame-ce` — GUI rendering

You also need `numpy` (should already be installed; if not: `pip install numpy`).

---

## 6. Run the GUI

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

## 7. Test Procedure

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

## 8. Troubleshooting

### No serial data / GUI shows nothing

- Check COM port is correct (`--port` flag)
- Check baud rate matches firmware (default: 921600)
- Verify coordinator Serial Monitor shows CSV lines
- Try unplugging and reconnecting the coordinator USB

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

### Perimeter node not connecting

- All nodes must be on **Wi-Fi channel 11** (hardcoded in firmware)
- Verify the `NODE_ID` was set correctly before flashing
- Power cycle the node
- Check coordinator Serial Monitor — missing link IDs indicate a node isn't communicating

---

## 9. Recording Results

After completing all tests, fill in this summary:

| Test | Result | Notes |
|------|--------|-------|
| 1. Empty Room Baseline | PASS / FAIL | |
| 2. Presence Detection — Enter | PASS / FAIL | |
| 3. Zone Tracking — Quadrants | PASS / FAIL | Quadrants correctly identified: Q1 Q2 Q3 Q4 |
| 4. Presence Detection — Leave | PASS / FAIL | |
| 5. Stability Run (10 min) | PASS / FAIL | |

**Overall:** PASS / FAIL

**Tester:** _______________  
**Date:** _______________  
**Notes:** _______________
