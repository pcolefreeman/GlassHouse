#!/usr/bin/env python3
"""
Cross-component data contract verification for S02 (Multi-Node Round-Robin Network).

Reads firmware and Python source files, then checks that MAC addresses, message types,
CSV format, WiFi channel, baud rate, and API calls are all consistent across:
  - coordinator/coordinator.ino
  - perimeter_node/perimeter_node.ino
  - python/serial_csi_reader.py

Exits 0 if all checks pass, 1 otherwise.
"""

from __future__ import annotations

import os
import re
import sys


# ---------------------------------------------------------------------------
# Paths — resolve relative to the project root (parent of scripts/)
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

COORDINATOR_PATH = os.path.join(PROJECT_ROOT, "coordinator", "coordinator.ino")
PERIMETER_PATH = os.path.join(PROJECT_ROOT, "perimeter_node", "perimeter_node.ino")
PYTHON_PARSER_PATH = os.path.join(PROJECT_ROOT, "python", "serial_csi_reader.py")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_file(path: str) -> str:
    """Read a file and return its contents as a string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record a check result and print PASS/FAIL."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        msg = f"  FAIL  {name}"
        if detail:
            msg += f"  — {detail}"
        print(msg)


# ---------------------------------------------------------------------------
# Load sources
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("S02 Cross-Component Data Contract Verification")
    print("=" * 60)

    # Check files exist
    for label, path in [
        ("coordinator.ino", COORDINATOR_PATH),
        ("perimeter_node.ino", PERIMETER_PATH),
        ("serial_csi_reader.py", PYTHON_PARSER_PATH),
    ]:
        if not os.path.isfile(path):
            print(f"FATAL: {label} not found at {path}")
            return 1

    coord = read_file(COORDINATOR_PATH)
    perim = read_file(PERIMETER_PATH)
    pyparser = read_file(PYTHON_PARSER_PATH)

    # ===================================================================
    # 1. MAC address consistency
    # ===================================================================
    print("\n--- 1. MAC Address Consistency ---")

    # Coordinator MAC 24:6F:28:AA:00:00 as byte array in both files
    coord_mac_bytes = "{0x24, 0x6F, 0x28, 0xAA, 0x00, 0x00}"
    check(
        "Coordinator MAC in coordinator.ino",
        coord_mac_bytes in coord,
        f"Expected {coord_mac_bytes}",
    )
    check(
        "Coordinator MAC in perimeter_node.ino",
        coord_mac_bytes in perim,
        f"Expected {coord_mac_bytes}",
    )

    # Perimeter node MACs 01 through 04
    for i in range(1, 5):
        mac_bytes = f"{{0x24, 0x6F, 0x28, 0xAA, 0x00, 0x0{i}}}"
        node_label = chr(ord("A") + i - 1)
        check(
            f"Node {node_label} MAC (0x0{i}) in coordinator.ino",
            mac_bytes in coord,
        )
        check(
            f"Node {node_label} MAC (0x0{i}) in perimeter_node.ino",
            mac_bytes in perim,
        )

    # ===================================================================
    # 2. Message type bytes
    # ===================================================================
    print("\n--- 2. Message Type Bytes ---")

    msg_types = {
        "MSG_TURN_CMD": "0x01",
        "MSG_CSI_TX": "0x02",
        "MSG_CSI_REPORT": "0x03",
    }
    for name, value in msg_types.items():
        # Check define in both firmware files
        pattern = rf"#define\s+{name}\s+{re.escape(value)}"
        check(
            f"{name} = {value} in coordinator.ino",
            bool(re.search(pattern, coord)),
        )
        check(
            f"{name} = {value} in perimeter_node.ino",
            bool(re.search(pattern, perim)),
        )

    # ===================================================================
    # 3. CSI_REPORT packet structure
    # ===================================================================
    print("\n--- 3. CSI_REPORT Packet Structure ---")

    # Perimeter node packs: report[0]=MSG_CSI_REPORT, [1]=tx_id, [2]=NODE_ID, [3]=rssi, [4]=hi, [5]=lo
    check(
        "Perimeter packs report[0] = MSG_CSI_REPORT",
        "report[0] = MSG_CSI_REPORT" in perim,
    )
    check(
        "Perimeter packs report[1] = tx_node_id",
        "report[1] = frame->tx_node_id" in perim,
    )
    check(
        "Perimeter packs report[2] = NODE_ID (rx)",
        "report[2] = (uint8_t)NODE_ID" in perim,
    )
    check(
        "Perimeter packs report[3] = rssi",
        "report[3] = (uint8_t)frame->rssi" in perim,
    )
    check(
        "Perimeter packs report[4] = len high byte",
        "report[4]" in perim and ">> 8" in perim,
    )
    check(
        "Perimeter packs report[5] = len low byte",
        "report[5]" in perim and "& 0xFF" in perim,
    )

    # Coordinator unpacks: data[0]=type, data[1]=tx_id, data[2]=rx_id, data[3]=rssi, data[4/5]=len
    check(
        "Coordinator reads data[0] as msg_type",
        "data[0]" in coord and "msg_type" in coord,
    )
    check(
        "Coordinator reads data[1] as tx_id",
        "data[1]" in coord and "tx_id" in coord,
    )
    check(
        "Coordinator reads data[2] as rx_id",
        "data[2]" in coord and "rx_id" in coord,
    )
    check(
        "Coordinator reads data[3] as rssi",
        "(int8_t)data[3]" in coord,
    )
    check(
        "Coordinator reads data[4]<<8|data[5] as csi_len",
        "data[4]" in coord and "data[5]" in coord,
    )

    # ===================================================================
    # 4. Serial CSV format consistency
    # ===================================================================
    print("\n--- 4. Serial CSV Format ---")

    # Coordinator printf format: CSI_DATA,%u,%c,%c,%s,%d,%u,
    check(
        "Coordinator CSV prefix: CSI_DATA,seq,tx,rx,link,rssi,len",
        'CSI_DATA,%u,%c,%c,%s,%d,%u,' in coord,
    )

    # Python parser expects S02: CSI_DATA,<seq>,<tx_node>,<rx_node>,<link_id>,<rssi>,<data_len>,<bytes>
    check(
        "Python parser documents S02 format in docstring",
        "CSI_DATA,<seq>,<tx_node>,<rx_node>,<link_id>,<rssi>,<data_len>" in pyparser,
    )

    # Python parser returns tx_node, rx_node, link_id keys
    check(
        "Python parser returns 'tx_node' key",
        '"tx_node"' in pyparser,
    )
    check(
        "Python parser returns 'rx_node' key",
        '"rx_node"' in pyparser,
    )
    check(
        "Python parser returns 'link_id' key",
        '"link_id"' in pyparser,
    )

    # Link ID generation — coordinator uses alphabetical ordering
    check(
        "Coordinator link_id uses alphabetical ordering (tx_label <= rx_label)",
        "tx_label <= rx_label" in coord,
    )

    # ===================================================================
    # 5. WiFi channel
    # ===================================================================
    print("\n--- 5. WiFi Channel ---")

    check(
        "Coordinator uses WIFI_CHANNEL 11",
        bool(re.search(r"#define\s+WIFI_CHANNEL\s+11\b", coord)),
    )
    check(
        "Perimeter uses WIFI_CHANNEL 11",
        bool(re.search(r"#define\s+WIFI_CHANNEL\s+11\b", perim)),
    )

    # ===================================================================
    # 6. Baud rate
    # ===================================================================
    print("\n--- 6. Baud Rate ---")

    check(
        "Coordinator SERIAL_BAUD = 921600",
        bool(re.search(r"#define\s+SERIAL_BAUD\s+921600\b", coord)),
    )
    check(
        "Python default baud = 921600",
        "default=921600" in pyparser,
    )

    # ===================================================================
    # 7. API calls present
    # ===================================================================
    print("\n--- 7. Required API Calls ---")

    perim_apis = [
        "esp_wifi_set_csi_rx_cb",
        "esp_wifi_set_promiscuous",
        "esp_now_init",
        "esp_now_send",
        "esp_now_register_recv_cb",
        "esp_wifi_set_mac",
    ]
    for api in perim_apis:
        check(
            f"Perimeter calls {api}",
            api in perim,
        )

    coord_apis = [
        "esp_now_init",
        "esp_now_send",
        "esp_now_register_recv_cb",
        "esp_wifi_set_mac",
    ]
    for api in coord_apis:
        check(
            f"Coordinator calls {api}",
            api in coord,
        )

    # ===================================================================
    # Summary
    # ===================================================================
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
