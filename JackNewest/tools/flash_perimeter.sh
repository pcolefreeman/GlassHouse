#!/bin/bash
# tools/flash_perimeter.sh — Build, flash, and provision a perimeter node
# Usage: ./flash_perimeter.sh <port> <node_id>
set -e
PORT="${1:-/dev/ttyUSB0}"
NODE_ID="${2:-1}"
cd "$(dirname "$0")/../firmware/perimeter"
idf.py set-target esp32s3
idf.py build
idf.py -p "$PORT" flash

# Provision via RuView's provision.py
python provision.py --port "$PORT" \
    --ssid "CSI_NET_V2" \
    --password "glasshouse" \
    --target-ip "192.168.4.1" \
    --target-port 4210 \
    --node-id "$NODE_ID"
echo "Node $NODE_ID provisioned on $PORT"
