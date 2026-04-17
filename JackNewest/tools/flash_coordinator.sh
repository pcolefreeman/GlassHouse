#!/bin/bash
# tools/flash_coordinator.sh — Build and flash coordinator firmware
# Usage: ./flash_coordinator.sh <port>
set -e
cd "$(dirname "$0")/../firmware/coordinator"
idf.py set-target esp32s3
idf.py build
idf.py -p "${1:-/dev/ttyUSB0}" flash monitor
