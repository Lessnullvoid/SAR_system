#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# S.A.R  —  Start SuperCollider drone on Raspberry Pi
# ─────────────────────────────────────────────────────────────
# Prerequisites (install once):
#   sudo apt install supercollider jackd2
#   pip install python-osc
#
# This script:
#   1. Finds the USB sound card (or falls back to default)
#   2. Starts JACK on that device (low-latency, 48 kHz)
#   3. Launches sclang with the drone SynthDef
#
# Usage:
#   chmod +x supercollider/start_sc.sh
#   ./supercollider/start_sc.sh
#
# To stop: press Ctrl+C (kills JACK + sclang cleanly)
# ─────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DRONE_SCD="$SCRIPT_DIR/sar_drone.scd"

# ── Find USB audio card ──────────────────────────────────────
USB_CARD=""
while IFS= read -r line; do
    if echo "$line" | grep -iq "usb"; then
        USB_CARD=$(echo "$line" | grep -oP '^\s*\K\d+')
        break
    fi
done < /proc/asound/cards 2>/dev/null || true

if [ -z "$USB_CARD" ]; then
    echo "[SAR] No USB audio card detected — using default ALSA device."
    ALSA_DEV="hw:0"
else
    echo "[SAR] USB audio card found: card $USB_CARD"
    ALSA_DEV="hw:$USB_CARD"
fi

# ── Kill any existing JACK / sclang instances ────────────────
pkill -9 jackd 2>/dev/null || true
pkill -9 sclang 2>/dev/null || true
sleep 1

# ── Start JACK ───────────────────────────────────────────────
echo "[SAR] Starting JACK on $ALSA_DEV (48kHz, 1024 frames, 2 periods)..."
jackd -d alsa \
    -d "$ALSA_DEV" \
    -r 48000 \
    -p 1024 \
    -n 2 \
    -S &

JACK_PID=$!
sleep 2

if ! kill -0 "$JACK_PID" 2>/dev/null; then
    echo "[SAR] ERROR: JACK failed to start. Check audio device."
    exit 1
fi
echo "[SAR] JACK running (PID $JACK_PID)."

# ── Launch sclang with the drone ─────────────────────────────
echo "[SAR] Launching SuperCollider drone..."
sclang "$DRONE_SCD" &
SC_PID=$!

echo "[SAR] sclang running (PID $SC_PID)."
echo "[SAR] Drone active — waiting for OSC data on port 57120."
echo "[SAR] Press Ctrl+C to stop."

# ── Cleanup on exit ──────────────────────────────────────────
cleanup() {
    echo ""
    echo "[SAR] Shutting down..."
    kill "$SC_PID" 2>/dev/null || true
    sleep 1
    kill "$JACK_PID" 2>/dev/null || true
    echo "[SAR] Done."
}

trap cleanup INT TERM

# Wait for sclang to exit
wait "$SC_PID" 2>/dev/null
cleanup
