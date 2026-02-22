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
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)


class AudioOutput:
    """
    Blocking-write audio output with a pre-filled ring buffer.
    """

    # Pre-fill the ring buffer with this much silence before starting
    # playback.  Gives the DSP pipeline a head start so the playback
    # thread always has data to read even when the GIL is contended.
    PREFILL_SECONDS = 1.5

    # Frames per blocking write.  2048 / 48000 ≈ 42.7 ms per write.
    # During that 42.7 ms the GIL is RELEASED (C code in PortAudio).
    WRITE_FRAMES = 2048

    def __init__(
        self,
        sample_rate: float = 48000.0,
        buffer_seconds: float = 4.0,
    ):
        self.sample_rate = sample_rate
        buf_size = int(sample_rate * buffer_seconds)
        self._buf = np.zeros(buf_size, dtype=np.float32)
        self._buf_size = buf_size

        # Pre-fill: advance write pointer so the playback thread starts
        # with 1.5 s of silence in front of it.
        prefill = int(sample_rate * self.PREFILL_SECONDS)
        self._write_pos = prefill
        self._read_pos = 0
        self._lock = threading.Lock()

        self._stream: Optional[sd.OutputStream] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── internal helpers ──────────────────────────────────────────────

    def _readable(self) -> int:
        return (self._write_pos - self._read_pos) % self._buf_size

    def _writable(self) -> int:
        return self._buf_size - 1 - self._readable()

    # ── playback thread ───────────────────────────────────────────────

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
        CHUNK = self.WRITE_FRAMES
        out = np.zeros((CHUNK, 1), dtype=np.float32)

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
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            latency="high",
        )
        self._stream.start()

        self._thread = threading.Thread(
            target=self._play_loop, daemon=True, name="audio-playback"
        )
        self._thread.start()
        log.info(
            "Audio started: %.0f Hz, pre-fill %.1f s, write %d frames",
            self.sample_rate, self.PREFILL_SECONDS, self.WRITE_FRAMES,
        )

    def write(self, audio: np.ndarray) -> None:
        """Called from the DSP thread to enqueue audio for playback."""
        if len(audio) == 0:
            return
        if self._stream is None:
            self.start()

        samples = np.ascontiguousarray(audio, dtype=np.float32).ravel()
        n = len(samples)

        with self._lock:
            space = self._writable()
            if n > space:
                # Buffer overflow — drop oldest samples to make room.
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
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
