"""
OSC bridge — sends tile state data to SuperCollider for sonification.

Uses the python-osc library to send structured OSC messages to
SuperCollider (or any OSC-compatible audio engine).

OSC Address Space
─────────────────
  /sar/tile/<tile_id>/score <float>
      Composite anomaly score (0.0–1.0).

  /sar/tile/<tile_id>/seismic <events_24h> <max_mag>
      Seismic activity summary.

  /sar/tile/<tile_id>/geomag <kp> <dst>
      Geomagnetic indices.

  /sar/tile/<tile_id>/tec <tecu> <delta>
      Ionospheric TEC.

  /sar/tile/<tile_id>/rf <anomaly_score> <peak_freq_mhz>
      RF anomaly score.

  /sar/global/kp <float>
      Current Kp index.

  /sar/global/dst <float>
      Current Dst index (nT).

  /sar/alert <tile_id> <level> <score> <description>
      Fusion alert notification.

  /sar/heartbeat <timestamp>
      Periodic heartbeat.

Usage
-----
    from sar.osc.bridge import OSCBridge
    bridge = OSCBridge(host="127.0.0.1", port=57120)
    bridge.send_tile_state(tile_state)
    bridge.send_alert(fusion_alert)
"""
from __future__ import annotations

import atexit
import logging
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# Try to import python-osc; gracefully degrade if unavailable
try:
    from pythonosc import udp_client
    _HAS_OSC = True
except ImportError:
    _HAS_OSC = False
    log.info("python-osc not installed — OSC bridge disabled")


class OSCBridge:
    """Sends tile state data to SuperCollider via OSC.

    Falls back to no-op if python-osc is not installed.

    Parameters
    ----------
    host : str
        SuperCollider OSC host (default "127.0.0.1").
    port : int
        SuperCollider OSC port (default 57120).
    enabled : bool
        Set to False to disable without removing from pipeline.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 57120,
        enabled: bool = True,
    ):
        self._host = host
        self._port = port
        self._enabled = enabled and _HAS_OSC
        self._client = None
        self._msg_count = 0

        if self._enabled:
            try:
                self._client = udp_client.SimpleUDPClient(host, port)
                log.info("OSC bridge → %s:%d", host, port)
            except Exception as exc:
                log.warning("OSC client init failed: %s", exc)
                self._enabled = False

    def close(self) -> None:
        """Release the UDP socket."""
        if self._client is not None:
            try:
                self._client._sock.close()
            except Exception:
                pass
            self._client = None
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def send_tile_state(self, state) -> None:
        """Send a TileState's data via OSC."""
        if not self._enabled or not self._client:
            return

        tid = state.tile_id
        try:
            self._client.send_message(
                f"/sar/tile/{tid}/score",
                state.composite_score,
            )
            self._client.send_message(
                f"/sar/tile/{tid}/seismic",
                [state.seismic.events_24h, state.seismic.max_mag],
            )
            self._client.send_message(
                f"/sar/tile/{tid}/geomag",
                [state.magnetic.kp_index, state.magnetic.dst_index],
            )
            self._client.send_message(
                f"/sar/tile/{tid}/tec",
                [state.ionospheric.tec_tecu, state.ionospheric.tec_delta],
            )
            self._client.send_message(
                f"/sar/tile/{tid}/rf",
                [state.rf.anomaly_score, state.rf.peak_freq_mhz],
            )
            self._msg_count += 5
        except Exception as exc:
            log.debug("OSC send error for tile %s: %s", tid, exc)

    def send_alert(self, alert) -> None:
        """Send a FusionAlert via OSC."""
        if not self._enabled or not self._client:
            return

        try:
            self._client.send_message(
                "/sar/alert",
                [alert.tile_id, alert.level,
                 alert.composite_score, alert.description],
            )
            self._msg_count += 1
        except Exception as exc:
            log.debug("OSC alert send error: %s", exc)

    def send_global(self, kp: float, dst: float) -> None:
        """Send global geomagnetic state."""
        if not self._enabled or not self._client:
            return

        try:
            self._client.send_message("/sar/global/kp", kp)
            self._client.send_message("/sar/global/dst", dst)
            self._msg_count += 2
        except Exception as exc:
            log.debug("OSC global send error: %s", exc)

    def heartbeat(self) -> None:
        """Send periodic heartbeat."""
        if not self._enabled or not self._client:
            return

        try:
            self._client.send_message("/sar/heartbeat", time.time())
            self._msg_count += 1
        except Exception as exc:
            log.debug("OSC heartbeat error: %s", exc)

    def send_top_tiles(self, top_tiles: list) -> None:
        """Send the top N tile scores as a batch.

        Parameters
        ----------
        top_tiles : list[(tile_id, score)]
        """
        if not self._enabled or not self._client:
            return

        try:
            for tid, score in top_tiles:
                self._client.send_message(
                    f"/sar/top/{tid}", score
                )
            self._msg_count += len(top_tiles)
        except Exception as exc:
            log.debug("OSC top tiles error: %s", exc)

    _ALERT_LEVEL_MAP = {
        "info": 0.0, "watch": 0.25,
        "warning": 0.6, "critical": 1.0,
    }

    def _send_synth_state(
        self,
        prefix: str,
        max_score: float,
        kp: float,
        dst: float,
        total_events: int,
        alert_level: str,
    ) -> None:
        """Send 5 OSC messages to a synth namespace (drone or resonator)."""
        if not self._enabled or not self._client:
            return

        alert_val = self._ALERT_LEVEL_MAP.get(alert_level, 0.0)
        try:
            self._client.send_message(f"/sar/{prefix}/activity", float(max_score))
            self._client.send_message(f"/sar/{prefix}/kp", float(kp))
            self._client.send_message(f"/sar/{prefix}/dst", float(dst))
            self._client.send_message(f"/sar/{prefix}/events", int(total_events))
            self._client.send_message(f"/sar/{prefix}/alert", alert_val)
            self._msg_count += 5
        except Exception as exc:
            log.warning("OSC %s send error: %s", prefix, exc)

    def send_drone_state(
        self,
        max_score: float,
        kp: float = 0.0,
        dst: float = 0.0,
        total_events: int = 0,
        alert_level: str = "info",
    ) -> None:
        """Send a compact summary for the drone synth.

        Only 5 lightweight OSC messages — designed for the sar_drone
        SynthDef which uses these to slowly modulate a continuous drone.

        Parameters
        ----------
        max_score : float
            Highest composite tile score (0.0–1.0).
        kp : float
            Current Kp index (0–9).
        dst : float
            Current Dst index (nT, typically -100 to +50).
        total_events : int
            Total earthquake events across all tiles in last 24h.
        alert_level : str
            Highest current fusion alert level.
        """
        self._send_synth_state("drone", max_score, kp, dst, total_events, alert_level)
        log.info(
            "OSC drone → activity=%.3f  kp=%.1f  dst=%.0f  "
            "events=%d  alert=%.2f",
            max_score, kp, dst, total_events,
            self._ALERT_LEVEL_MAP.get(alert_level, 0.0),
        )

    def send_resonator_state(
        self,
        max_score: float,
        kp: float = 0.0,
        dst: float = 0.0,
        total_events: int = 0,
        alert_level: str = "info",
    ) -> None:
        """Send a compact summary for the sympathetic string resonator.

        Same 5-parameter interface as the drone but targets
        /sar/resonator/* OSC addresses.
        """
        self._send_synth_state("resonator", max_score, kp, dst, total_events, alert_level)

    def stats(self) -> Dict:
        return {
            "enabled": self._enabled,
            "host": self._host,
            "port": self._port,
            "messages_sent": self._msg_count,
        }


# ─────────────────────────────────────────────────────────────
# SuperCollider process manager
# ─────────────────────────────────────────────────────────────

_SCLANG_PATHS_MAC = [
    "/Applications/SuperCollider.app/Contents/MacOS/sclang",
    os.path.expanduser("~/Applications/SuperCollider.app/Contents/MacOS/sclang"),
]

_SC_DIR = Path(__file__).resolve().parent.parent.parent / "supercollider"
_MAIN_SCD = str(_SC_DIR / "sar_main.scd")
_DRONE_SCD = str(_SC_DIR / "sar_drone.scd")
_RESONATOR_SCD = str(_SC_DIR / "sar_resonator.scd")

def get_scd_path(synth_mode: str = "both") -> str:
    """Return the SCD file path for the requested synth mode."""
    if synth_mode == "drone":
        return _DRONE_SCD
    elif synth_mode == "resonator":
        return _RESONATOR_SCD
    return _MAIN_SCD


def _find_sclang() -> Optional[str]:
    """Locate the sclang binary on the current platform."""
    # Check PATH first (works on Linux / Pi where apt puts it in /usr/bin)
    found = shutil.which("sclang")
    if found:
        return found
    # macOS: check known app bundle locations
    if platform.system() == "Darwin":
        for p in _SCLANG_PATHS_MAC:
            if os.path.isfile(p):
                return p
    return None


class SuperColliderProcess:
    """Manages the lifecycle of sclang (+ JACK on Linux/Pi).

    Launches sclang with the SAR audio engine automatically
    (drone + resonator), and tears it down on exit.

    Parameters
    ----------
    scd_path : str | None
        Path to the .scd file. Defaults to supercollider/sar_main.scd.
    auto_jack : bool
        On Linux, start JACK on the USB audio card before sclang.
    """

    def __init__(
        self,
        scd_path: Optional[str] = None,
        auto_jack: bool = True,
    ):
        self._scd = scd_path or _MAIN_SCD
        self._auto_jack = auto_jack
        self._sc_proc: Optional[subprocess.Popen] = None
        self._jack_proc: Optional[subprocess.Popen] = None
        self._running = False
        self._atexit_registered = False

    @property
    def running(self) -> bool:
        if self._sc_proc and self._sc_proc.poll() is not None:
            self._running = False
        return self._running

    def start(self) -> bool:
        """Start SuperCollider. Returns True on success."""
        if self._running:
            log.info("SuperCollider already running.")
            return True

        sclang = _find_sclang()
        if not sclang:
            log.warning(
                "sclang not found — SuperCollider drone disabled. "
                "Install SuperCollider to enable."
            )
            return False

        if not os.path.isfile(self._scd):
            log.warning("SCD file not found: %s", self._scd)
            return False

        is_linux = platform.system() == "Linux"

        # On Linux/Pi: ensure JACK is available for scsynth
        use_pw_jack = False
        if is_linux and self._auto_jack:
            if self._pipewire_has_jack():
                use_pw_jack = True
                log.info("Will launch sclang via pw-jack (PipeWire)")
            else:
                self._start_jack()

        # Launch sclang (via pw-jack on PipeWire systems so scsynth
        # connects to PipeWire's JACK layer instead of raw ALSA)
        try:
            if use_pw_jack:
                cmd = ["pw-jack", sclang, self._scd]
            else:
                cmd = [sclang, self._scd]
            log.info("Starting sclang: %s", " ".join(cmd))
            self._sc_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self._running = True
            if not self._atexit_registered:
                atexit.register(self.stop)
                self._atexit_registered = True
            log.info("SuperCollider launched (PID %d)", self._sc_proc.pid)

            # Start a background thread to drain stdout so the pipe
            # doesn't fill up and block sclang
            import threading
            t = threading.Thread(
                target=self._drain_output, daemon=True, name="sc-log"
            )
            t.start()

            return True
        except Exception as exc:
            log.error("Failed to start SuperCollider: %s", exc)
            return False

    def _start_jack(self) -> None:
        """Start JACK on a USB audio card (Linux only).

        On PipeWire systems (Pi 5 / Bookworm), PipeWire provides JACK
        compatibility via pipewire-jack — no separate jackd needed.
        On older systems, starts jackd on the USB audio card directly.
        """
        # Check if PipeWire is already providing JACK
        if self._pipewire_has_jack():
            log.info("PipeWire-JACK detected — skipping jackd start")
            return

        usb_dev = "hw:0"
        try:
            cards = Path("/proc/asound/cards").read_text()
            for line in cards.splitlines():
                low = line.lower()
                if "usb" in low or "sound blaster" in low or "creative" in low:
                    card_num = line.strip().split()[0]
                    usb_dev = f"hw:{card_num}"
                    log.info("JACK target: %s (%s)", usb_dev, line.strip())
                    break
        except Exception:
            pass

        # Stop any previous SAR-launched JACK (only our PID, not system-wide)
        if self._jack_proc and self._jack_proc.poll() is None:
            try:
                self._jack_proc.terminate()
                self._jack_proc.wait(timeout=3)
            except (subprocess.TimeoutExpired, Exception):
                try:
                    self._jack_proc.kill()
                    self._jack_proc.wait(timeout=2)
                except Exception:
                    pass
            self._jack_proc = None
        time.sleep(0.5)

        try:
            log.info("Starting JACK on %s", usb_dev)
            self._jack_proc = subprocess.Popen(
                [
                    "jackd", "-d", "alsa",
                    "-d", usb_dev,
                    "-r", "48000",
                    "-p", "1024",
                    "-n", "2",
                    "-S",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(2)  # give JACK time to initialise
            if self._jack_proc.poll() is not None:
                log.warning("JACK exited prematurely (code %d)",
                            self._jack_proc.returncode)
                self._jack_proc = None
            else:
                log.info("JACK running (PID %d)", self._jack_proc.pid)
        except FileNotFoundError:
            log.warning("jackd not found — skipping JACK start")
        except Exception as exc:
            log.warning("JACK start failed: %s", exc)

    @staticmethod
    def _pipewire_has_jack() -> bool:
        """Check if PipeWire is running and provides JACK compatibility."""
        try:
            pw = subprocess.run(
                ["systemctl", "--user", "is-active", "pipewire"],
                capture_output=True, text=True, timeout=3,
            )
            if pw.stdout.strip() != "active":
                return False
            # Check that the JACK socket exists (pipewire-jack installed)
            jack_sock = Path("/run/user") / str(os.getuid()) / "pipewire-0"
            if jack_sock.exists():
                log.info("PipeWire active + JACK socket found: %s", jack_sock)
                return True
            # Alternatively check for libjack override
            pw_jack = shutil.which("pw-jack")
            if pw_jack:
                log.info("PipeWire active + pw-jack found: %s", pw_jack)
                return True
            return False
        except Exception:
            return False

    def _drain_output(self) -> None:
        """Read sclang stdout and log it (runs in daemon thread)."""
        try:
            for line in self._sc_proc.stdout:
                stripped = line.rstrip()
                if stripped:
                    log.debug("[sclang] %s", stripped)
        except Exception:
            pass

    def stop(self) -> None:
        """Shut down sclang and JACK gracefully."""
        if self._sc_proc and self._sc_proc.poll() is None:
            log.info("Stopping SuperCollider (PID %d)...", self._sc_proc.pid)
            self._sc_proc.terminate()
            try:
                self._sc_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._sc_proc.kill()
                self._sc_proc.wait(timeout=2)
            log.info("SuperCollider stopped.")
        self._sc_proc = None

        if self._jack_proc and self._jack_proc.poll() is None:
            log.info("Stopping JACK (PID %d)...", self._jack_proc.pid)
            self._jack_proc.terminate()
            try:
                self._jack_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._jack_proc.kill()
                self._jack_proc.wait(timeout=2)
            log.info("JACK stopped.")
        self._jack_proc = None

        self._running = False
