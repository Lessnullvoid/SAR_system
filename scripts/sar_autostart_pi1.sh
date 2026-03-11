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

# Wait for USB sound card to appear (may take a few seconds after boot)
for _ in $(seq 1 10); do
    PLAY3_ID=$(wpctl status 2>/dev/null \
        | grep -i 'play.*3.*analog stereo' | grep -oP '^\s*\K\d+' | head -1)
    [ -n "$PLAY3_ID" ] && break
    sleep 2
done

if [ -n "$PLAY3_ID" ]; then
    wpctl set-default "$PLAY3_ID" 2>/dev/null
    wpctl set-volume "$PLAY3_ID" 1.0 2>/dev/null
    echo "$(date) Play! 3 set as default (ID $PLAY3_ID)" >> /tmp/sar.log
else
    echo "$(date) WARNING: Play! 3 not found, using system default" >> /tmp/sar.log
fi

# Re-apply volume 30s after app starts (overrides WirePlumber's saved state)
(
    sleep 30
    P3_ID=$(wpctl status 2>/dev/null \
        | grep -i 'play.*3.*analog stereo' | grep -oP '^\s*\K\d+' | head -1)
    [ -n "$P3_ID" ] && wpctl set-default "$P3_ID" 2>/dev/null
    [ -n "$P3_ID" ] && wpctl set-volume "$P3_ID" 1.0 2>/dev/null
    echo "$(date) Play! 3 volume re-applied (ID $P3_ID)" >> /tmp/sar.log
) &

exec python -m python_app.gui_main \
    --synth drone \
    --antenna discone \
    >> /tmp/sar.log 2>&1
