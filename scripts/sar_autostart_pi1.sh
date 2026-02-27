#!/bin/bash
# S.A.R — Auto-start for Pi 1
#
# Hardware: discone antenna, drone synth, Sound Blaster Play! 3
# Audio:    SDR + SuperCollider both output to Play! 3 (single card)
#
# Install:
#   mkdir -p ~/.config/autostart
#   cp ~/SAR_system/scripts/sar.desktop ~/.config/autostart/

sleep 5

export DISPLAY=:0
export XDG_RUNTIME_DIR="/run/user/$(id -u)"

cd ~/SAR_system
source .venv/bin/activate

wpctl set-default "$(wpctl status 2>/dev/null \
    | grep -i 'sound blaster' | grep -oP '^\s*\K\d+' | head -1)" 2>/dev/null

exec python -m python_app.gui_main \
    --synth drone \
    --antenna discone \
    >> /tmp/sar.log 2>&1
