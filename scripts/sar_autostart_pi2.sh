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

# Set Play! 3 as default sink (SDR audio) and ensure G3 is at full volume
PLAY3_ID=$(wpctl status 2>/dev/null | grep -i 'play.*3.*analog stereo' | grep -oP '^\s*\K\d+' | head -1)
G3_ID=$(wpctl status 2>/dev/null | grep -i 'g3.*analog stereo' | grep -oP '^\s*\K\d+' | head -1)
[ -n "$PLAY3_ID" ] && wpctl set-default "$PLAY3_ID" 2>/dev/null
[ -n "$PLAY3_ID" ] && wpctl set-volume "$PLAY3_ID" 1.0 2>/dev/null
[ -n "$G3_ID" ] && wpctl set-volume "$G3_ID" 2.5 2>/dev/null

# Route SuperCollider (JACK) output to G3 once scsynth ports appear.
# Polls every 2s for up to 60s, then gives up silently.
G3_SINK="alsa_output.usb-Creative_Technology_Ltd_Sound_Blaster_G3_27676C972BD491DC-03.analog-stereo"
(
    for _ in $(seq 1 30); do
        if pw-link -o 2>/dev/null | grep -q "SuperCollider:out_1"; then
            sleep 1
            PLAY3_FL=$(pw-link -i 2>/dev/null | grep -i "Play__3.*playback_FL" | head -1 | xargs)
            PLAY3_FR=$(pw-link -i 2>/dev/null | grep -i "Play__3.*playback_FR" | head -1 | xargs)
            [ -n "$PLAY3_FL" ] && pw-link -d "SuperCollider:out_1" "$PLAY3_FL" 2>/dev/null
            [ -n "$PLAY3_FR" ] && pw-link -d "SuperCollider:out_2" "$PLAY3_FR" 2>/dev/null
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
