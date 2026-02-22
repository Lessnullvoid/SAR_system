"""
Audio output via sounddevice **blocking writes** (no Python callback).

Previous approach (callback-based):
    sounddevice calls a Python function ~47 times/sec from a C thread.
    That callback needs the GIL to execute numpy/ring-buffer code.
    When the DSP or GUI thread holds the GIL, the callback is delayed
    → PortAudio's internal buffer runs dry → audible click/dropout.

Current approach (blocking-write thread):
    A dedicated Python thread reads from the ring buffer and calls
    sd.OutputStream.write(chunk).  That write() is a blocking C call
    that releases the GIL while PortAudio plays the audio.  The Python
    thread only needs the GIL for ~0.01 ms per chunk (ring-buffer read),
    so GIL contention is negligible.

    DSP thread ──write()──▶ [ring buffer] ──▶ _play_loop() ──▶ PortAudio

Benefits:
  1. Audio timing is 100 % driven by PortAudio's C code (deterministic).
  2. GIL is held for < 0.02 ms per 43 ms audio chunk → zero contention.
  3. Buffer underruns produce smooth silence (explicit zeros), not
     hardware glitches (clicks/pops).
"""
from __future__ import annotations

import logging
import platform
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

_IS_PI = (
    platform.system() == "Linux"
    and platform.machine().startswith(("aarch64", "arm"))
)


# ── Audio device detection ─────────────────────────────────────────────

def list_output_devices() -> List[Dict]:
    """Return a list of available output audio devices.

    Each entry has keys: index, name, channels, sample_rate, is_usb, is_builtin.
    """
    devices = sd.query_devices()
    results = []
    for i, d in enumerate(devices):
        if d["max_output_channels"] < 1:
            continue
        name = d["name"]
        results.append({
            "index": i,
            "name": name,
            "channels": d["max_output_channels"],
            "sample_rate": d["default_samplerate"],
            "is_usb": any(k in name.lower() for k in ("usb", "sound blaster", "creative")),
            "is_builtin": any(k in name.lower() for k in (
                "bcm", "headphone", "built-in", "macbook", "speakers",
            )),
        })
    return results


def find_sdr_audio_device() -> Optional[int]:
    """Auto-detect the best output device for SDR audio.

    On Raspberry Pi: prefers the built-in bcm2835 headphone jack.
    On macOS/desktop: returns None (use system default).
    """
    if not _IS_PI:
        return None

    devices = list_output_devices()

    # Pi: prefer bcm2835 Headphones (3.5mm jack)
    for d in devices:
        if "bcm" in d["name"].lower() or "headphone" in d["name"].lower():
            log.info("SDR audio → Pi built-in: [%d] %s", d["index"], d["name"])
            return d["index"]

    # Fallback: first non-USB device
    for d in devices:
        if not d["is_usb"]:
            log.info("SDR audio → fallback: [%d] %s", d["index"], d["name"])
            return d["index"]

    return None


def find_usb_audio_device() -> Optional[int]:
    """Auto-detect a USB audio device (for SuperCollider / JACK).

    Returns the sounddevice index, or None if no USB device found.
    """
    devices = list_output_devices()
    for d in devices:
        if d["is_usb"]:
            log.info("USB audio device found: [%d] %s", d["index"], d["name"])
            return d["index"]
    return None


def find_usb_alsa_device() -> str:
    """Find the ALSA hw:N identifier for the first USB audio card (Linux).

    Used by JACK to open the correct hardware device.
    Returns 'hw:0' as fallback.
    """
    from pathlib import Path
    try:
        cards = Path("/proc/asound/cards").read_text()
        for line in cards.splitlines():
            if "usb" in line.lower() or "sound blaster" in line.lower():
                card_num = line.strip().split()[0]
                dev = f"hw:{card_num}"
                log.info("USB ALSA device for JACK: %s (%s)",
                         dev, line.strip())
                return dev
    except Exception:
        pass
    return "hw:0"


class AudioOutput:
    """
    Blocking-write audio output with a pre-filled ring buffer.
    """

    # Pre-fill the ring buffer with this much silence before starting
    # playback.  Gives the DSP pipeline a head start so the playback
    # thread always has data to read even when the GIL is contended.
    # 3 seconds provides enough runway to survive GIL contention from
    # map scanner image loading, sensor fetches, and DSP processing
    # overhead for narrow-band modes (AM/SSB).
    PREFILL_SECONDS = 3.0

    # Frames per blocking write.  2048 / 48000 ≈ 42.7 ms per write.
    # During that 42.7 ms the GIL is RELEASED (C code in PortAudio).
    # Keeping at 2048 for fast thread responsiveness during shutdown.
    WRITE_FRAMES = 2048

    def __init__(
        self,
        sample_rate: float = 48000.0,
        buffer_seconds: float = 4.0,
        device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self._device = device
        buf_size = int(sample_rate * buffer_seconds)
        self._buf = np.zeros(buf_size, dtype=np.float32)
        self._buf_size = buf_size

        prefill = int(sample_rate * self.PREFILL_SECONDS)
        self._write_pos = prefill
        self._read_pos = 0
        self._lock = threading.Lock()

        self._stream: Optional[sd.OutputStream] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Software volume (0.0 = mute, 1.0 = unity, >1.0 = boost)
        # Default 0.9 (90%) — synced by _sync_volume_to_worker after start
        self._volume: float = 0.9

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, val: float) -> None:
        self._volume = max(0.0, min(val, 3.0))

    # ── internal helpers ──────────────────────────────────────────────

    def _readable(self) -> int:
        return (self._write_pos - self._read_pos) % self._buf_size

    def _writable(self) -> int:
        return self._buf_size - 1 - self._readable()

    # ── playback thread ───────────────────────────────────────────────

    @staticmethod
    def _try_set_realtime_priority() -> None:
        """Set SCHED_FIFO on the calling thread if possible (Linux only).

        Must be called from within the audio thread itself so that the
        scheduler change applies to the correct OS thread.
        """
        import sys
        if sys.platform != "linux":
            return
        try:
            import os
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(50))
            log.info("Audio thread: SCHED_FIFO priority 50")
        except (PermissionError, OSError) as exc:
            log.debug("Cannot set FIFO priority (add user to 'audio' group): %s", exc)
        except AttributeError:
            pass  # sched_setscheduler not available

    def _play_loop(self) -> None:
        """Dedicated playback thread.

        Reads chunks from the ring buffer and writes them to the audio
        stream.  ``stream.write()`` is a **blocking** C call that waits
        inside PortAudio until the audio device has consumed the data.
        During that wait the GIL is released, so the DSP and GUI threads
        run freely.

        If the ring buffer is empty we write zeros (smooth silence) rather
        than skipping the write.  This keeps the PortAudio pipeline fed
        continuously — no hardware under-run clicks.
        """
        # Elevate this thread to real-time priority on Linux (Raspberry Pi).
        self._try_set_realtime_priority()

        CHUNK = self.WRITE_FRAMES
        out = np.zeros((CHUNK, 1), dtype=np.float32)
        underrun_count = 0
        underrun_log_interval = 100  # log every N underruns

        while self._running:
            # --- copy from ring buffer (brief GIL hold: ~0.01 ms) ---
            with self._lock:
                avail = self._readable()
                n = min(CHUNK, avail)
                ri = self._read_pos

                if n > 0:
                    end = ri + n
                    if end <= self._buf_size:
                        out[:n, 0] = self._buf[ri:end]
                    else:
                        first = self._buf_size - ri
                        out[:first, 0] = self._buf[ri:]
                        out[first:n, 0] = self._buf[:n - first]
                    self._read_pos = (ri + n) % self._buf_size

                if n < CHUNK:
                    out[n:, 0] = 0.0   # pad with silence
                    if n == 0:
                        underrun_count += 1
                        if underrun_count <= 3 or underrun_count % underrun_log_interval == 0:
                            log.warning(
                                "Audio underrun #%d (buffer empty, wrote %d zeros)",
                                underrun_count, CHUNK,
                            )

            # --- blocking write (releases GIL for ~43 ms) ---
            try:
                self._stream.write(out)
            except Exception:
                # Stream closed or device disconnected — exit cleanly
                break

    # ── public API ────────────────────────────────────────────────────

    def start(self) -> None:
        if self._stream is not None:
            return

        self._running = True

        # Open stream WITHOUT a callback → blocking mode.
        # latency='high' tells PortAudio to use a large internal
        # buffer, giving extra protection against timing jitter.
        kwargs = dict(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            latency="high",
        )
        if self._device is not None:
            kwargs["device"] = self._device

        self._stream = sd.OutputStream(**kwargs)
        self._stream.start()

        self._thread = threading.Thread(
            target=self._play_loop, daemon=True, name="audio-playback"
        )
        self._thread.start()

        dev_name = ""
        if self._device is not None:
            try:
                dev_name = f" → [{self._device}] {sd.query_devices(self._device)['name']}"
            except Exception:
                dev_name = f" → device {self._device}"

        log.info(
            "Audio started: %.0f Hz, pre-fill %.1f s, write %d frames%s",
            self.sample_rate, self.PREFILL_SECONDS, self.WRITE_FRAMES, dev_name,
        )

    def write(self, audio: np.ndarray) -> None:
        """Called from the DSP thread to enqueue audio for playback."""
        if len(audio) == 0:
            return
        if self._stream is None:
            self.start()

        samples = np.ascontiguousarray(audio, dtype=np.float32).ravel()

        # Apply software volume as an absolute ceiling (like a radio knob).
        # All demodulators normalize to ~0.15 peak via AGC, so volume
        # scales from there.  The hard clip guarantees no mode can ever
        # exceed the set volume level — set it to 40%, max output is 40%.
        vol = self._volume
        if vol != 1.0:
            samples = samples * np.float32(vol)
        np.clip(samples, -1.0, 1.0, out=samples)

        n = len(samples)

        # Guard: if write is larger than buffer capacity, keep only the tail
        max_write = self._buf_size - 1
        if n >= max_write:
            samples = samples[-max_write:]
            n = max_write

        with self._lock:
            space = self._writable()
            if n > space:
                drop = n - space
                self._read_pos = (self._read_pos + drop) % self._buf_size

            wi = self._write_pos
            end = wi + n
            if end <= self._buf_size:
                self._buf[wi:end] = samples
            else:
                first = self._buf_size - wi
                self._buf[wi:] = samples[:first]
                self._buf[:n - first] = samples[first:]
            self._write_pos = (wi + n) % self._buf_size

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._stream is not None:
            try:
                self._stream.abort()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
