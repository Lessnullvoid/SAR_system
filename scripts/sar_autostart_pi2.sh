#!/bin/bash
# S.A.R — Auto-start for Pi 2
#
# Hardware: loop antenna, resonator synth,
#           Sound Blaster Play! 3 (SDR audio) + G3 (SuperCollider)
# Audio:    SDR -> Play! 3,  SuperCollider -> G3 (routed via pw-link)
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

# Route SuperCollider (JACK) output to G3 once scsynth ports appear.
# Polls every 2s for up to 60s, then gives up silently.
G3_SINK="alsa_output.usb-Creative_Technology_Ltd_Sound_Blaster_G3_27676C972BD491DC-03.analog-stereo"
(
    for _ in $(seq 1 30); do
        if pw-link -o 2>/dev/null | grep -q "SuperCollider:out_1"; then
            sleep 1
            pw-link -d "SuperCollider:out_1" "*Play*:playback_FL" 2>/dev/null
            pw-link -d "SuperCollider:out_2" "*Play*:playback_FR" 2>/dev/null
            pw-link "SuperCollider:out_1" "${G3_SINK}:playback_FL" 2>/dev/null
            pw-link "SuperCollider:out_2" "${G3_SINK}:playback_FR" 2>/dev/null
            echo "$(date) scsynth routed to G3" >> /tmp/sar.log
            exit 0
        fi
        sleep 2
    done
) &

exec python -m python_app.gui_main \
    --synth resonator \
    --antenna loop_antenna \
    >> /tmp/sar.log 2>&1
