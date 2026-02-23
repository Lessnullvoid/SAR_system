#!/bin/bash
# S.A.R — Auto-start script for Raspberry Pi
# Waits for the desktop to be ready, then launches the SAR application.
#
# Install: copy sar.desktop to ~/.config/autostart/
#   mkdir -p ~/.config/autostart
#   cp ~/SAR/scripts/sar.desktop ~/.config/autostart/

sleep 5

export DISPLAY=:0
export XDG_RUNTIME_DIR="/run/user/$(id -u)"

cd ~/SAR
source .venv/bin/activate

# Ensure USB sound card is default output
wpctl set-default "$(wpctl status 2>/dev/null | grep -i 'sound blaster' | grep -oP '^\s*\K\d+' | head -1)" 2>/dev/null

# Select synth: both (default), drone, or resonator
# Override by setting SAR_SYNTH env var before this script runs
SAR_SYNTH="${SAR_SYNTH:-both}"

exec python -m python_app.gui_main --synth "$SAR_SYNTH" >> /tmp/sar.log 2>&1
