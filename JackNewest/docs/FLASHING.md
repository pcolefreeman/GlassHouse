# GlassHouse v2 — Flashing & Provisioning Guide

## Prerequisites

- ESP-IDF 5.2 installed at `C:\Users\incre\esp\esp-idf`
- 5x ESP32-S3 modules connected via USB
- Know which COM port each board is on (check Device Manager > Ports)

## Sourcing ESP-IDF

ESP-IDF **must** be sourced before any build/flash commands. Git Bash doesn't work with ESP-IDF on Windows — use **PowerShell** instead.

**Open PowerShell and run:**
```powershell
C:\Users\incre\esp\esp-idf\export.ps1
```

You'll see output like:
```
Using Python interpreter in: C:\Users\incre\.espressif\...
Added idf.py to PATH.
Done! You can now compile ESP-IDF projects.
```

All flash commands below assume you're in this sourced PowerShell session.

## Network Topology

```
  [Node 2]----[Node 3]
     |    \  /    |
     |     \/     |
     |     /\     |
     |    /  \    |
  [Node 1]----[Node 4]
         \
     [Coordinator] ---USB serial---> [Raspberry Pi]
```

- **Coordinator**: Creates WiFi AP "CSI_NET_V2", forwards UDP packets to Pi via COBS serial
- **Perimeter nodes 1-4**: Connect to coordinator AP, send heartbeats + link reports

## Step 1: Flash the Coordinator

```powershell
cd C:\Users\incre\Class\Capstone\Claude\GSDprojects\glasshouse-v2\firmware\coordinator
idf.py set-target esp32s3
idf.py build
idf.py -p COM3 flash monitor
```

Replace `COM3` with your coordinator's actual port.

The serial monitor should show:
```
I (xxx) coordinator: SoftAP started: SSID=CSI_NET_V2 CH=1
I (xxx) coordinator: UDP bridge listening on port 4210
I (xxx) coordinator: GlassHouse v2 Coordinator ready
```

Press `Ctrl+]` to exit the monitor.

## Step 2: Flash Perimeter Nodes (First Pass)

Flash each node one at a time. On this first pass, just flash and note each node's MAC address from the boot log.

```powershell
cd C:\Users\incre\Class\Capstone\Claude\GSDprojects\glasshouse-v2\firmware\perimeter

# Only need to set target + build once (same binary for all nodes)
idf.py set-target esp32s3
idf.py build

# Flash each node (change COM port per node)
idf.py -p COM4 flash monitor   # Node 1
idf.py -p COM5 flash monitor   # Node 2
idf.py -p COM6 flash monitor   # Node 3
idf.py -p COM7 flash monitor   # Node 4
```

From each node's boot log, record the MAC address. It appears as:
```
I (xxx) wifi:mode : sta (AA:BB:CC:DD:EE:FF)
```

Write down all 4 MACs:
```
Node 1: __:__:__:__:__:__
Node 2: __:__:__:__:__:__
Node 3: __:__:__:__:__:__
Node 4: __:__:__:__:__:__
```

## Step 3: Provision Each Node

Still in PowerShell, provision each node with its WiFi credentials, node ID, and target IP:

```powershell
cd C:\Users\incre\Class\Capstone\Claude\GSDprojects\glasshouse-v2\firmware\perimeter

# Node 1
python provision.py --port COM4 --ssid "CSI_NET_V2" --password "glasshouse" --target-ip "192.168.4.1" --target-port 4210 --node-id 1

# Node 2
python provision.py --port COM5 --ssid "CSI_NET_V2" --password "glasshouse" --target-ip "192.168.4.1" --target-port 4210 --node-id 2

# Node 3
python provision.py --port COM6 --ssid "CSI_NET_V2" --password "glasshouse" --target-ip "192.168.4.1" --target-port 4210 --node-id 3

# Node 4
python provision.py --port COM7 --ssid "CSI_NET_V2" --password "glasshouse" --target-ip "192.168.4.1" --target-port 4210 --node-id 4
```

### Setting the MAC Filter

Each node needs to know its peers' MACs for CSI filtering. You have two options:

**Option A: Hardcode peer MACs (simplest for demo)**

Edit `firmware/perimeter/main/csi_collector.c` — add an init function that populates `s_peer_macs[]` and `s_peer_node_ids[]` with the known MACs from Step 2. Then rebuild and reflash all 4 nodes.

**Option B: Use NVS single-MAC filter per node**

Add `--filter-mac` to each provision command above:
```powershell
python provision.py --port COM4 --ssid "CSI_NET_V2" --password "glasshouse" --target-ip "192.168.4.1" --target-port 4210 --node-id 1 --filter-mac "AA:BB:CC:DD:EE:FF"
```

## Step 4: Verify with Debug Capture

Connect the coordinator to the Pi/PC via USB serial. Back in Git Bash or PowerShell:

```bash
python tools/capture_debug.py --port COM3 --duration 60
```

Expected output:
```
[  1.2s] link 12: var=0.001234 IDLE   n=20
[  1.4s] link 13: var=0.045678 MOTION n=20
[  1.6s] VITALS: presence=YES
...
========================================
47 link reports in 60.0s
Links seen: ['12', '13', '14', '23', '24', '34'] (6/6)
All 6 links detected!
========================================
```

If links are missing, check:
- Is the node powered on and connected to "CSI_NET_V2"?
- Is the MAC filter configured correctly?
- Is the node close enough for CSI capture?

## Step 5: Run Live Zone Detection

```bash
# Headless (text output)
python python/main.py --port COM3 --headless

# With LCD display (on Pi)
python python/main.py --port COM3
```

Walk around the room — zone output should track your quadrant.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `export.ps1` blocked | PowerShell execution policy | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| "SoftAP started" not shown | Coordinator flash failed | Reflash, check USB port |
| Nodes don't connect to AP | Wrong SSID/password in NVS | Re-provision with correct credentials |
| No link reports | MAC filter blocking all frames | Check `--filter-mac` matches a peer's actual MAC |
| Only some links appear | Node not in range or not powered | Check physical placement, power supply |
| "Forwarded 100 packets" not shown | No UDP traffic reaching coordinator | Verify nodes connected to AP (check coordinator log) |
| Zone stuck on "---" | Variance too low / baseline not established | Wait 5-10 seconds for baseline, then move |
| `idf.py` not found | ESP-IDF not sourced | Run `C:\Users\incre\esp\esp-idf\export.ps1` first |

## Port Reference

| Device | Example Port | Baud |
|--------|-------------|------|
| Coordinator (serial to Pi) | COM3 | 921600 |
| Node 1 (flash only) | COM4 | 921600 |
| Node 2 (flash only) | COM5 | 921600 |
| Node 3 (flash only) | COM6 | 921600 |
| Node 4 (flash only) | COM7 | 921600 |
