#!/bin/bash
# S.A.R — Auto-start dispatcher
#
# Detects which Pi this is and runs the matching script.
# Pi 2 has two USB sound cards (Play! 3 + G3), Pi 1 has one.
#
# Install on each Pi:
#   mkdir -p ~/.config/autostart
#   cp ~/SAR_system/scripts/sar.desktop ~/.config/autostart/

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if lsusb 2>/dev/null | grep -qi "Sound Blaster G3"; then
    exec "$SCRIPT_DIR/sar_autostart_pi2.sh"
else
    exec "$SCRIPT_DIR/sar_autostart_pi1.sh"
fi
