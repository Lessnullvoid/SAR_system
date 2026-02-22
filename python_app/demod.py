"""
Demodulators for FM, AM, LSB, USB.

Each demodulator expects CHANNEL-FILTERED IQ (narrow bandwidth),
not raw wideband IQ. The channel filter in gui_main.py handles that.

Signal chain:
    IIR channel filter → decimate → demod.demodulate() → audio samples

Reference implementations studied:
  - gqrx (rx_demod_am.cpp):  complex_to_mag → DC-blocking IIR → AGC on IQ
  - liquid-dsp (ampmodem.c): peak detector (cabs) → FIR DC blocker
                              SSB via Hilbert transform (firhilbf c2r)
  - GNU Radio (am_demod.py): complex_to_mag → add_const(-1) → FIR LPF
  - CubicSDR (ModemFM.cpp):  freqdem phase discriminator → resample + de-emphasis
  - SDR++ (am.h / ssb.h):    envelope → DC block → AGC w/ 100ms attack, 1s release

Key design rules:
  1. ALL modes use slow AGC (1.0s time constant) to normalize output.
     FM target (0.05) is lower than AM/SSB (0.12) because FM broadcast
     audio is heavily compressed — same peak ≠ same perceived loudness.
  2. FM applies 75µs de-emphasis (standard for North America, matches gqrx/CubicSDR)
  3. AM uses persistent DC-blocking IIR (NOT per-block mean subtraction)
  4. SSB: USB = real(IQ), LSB = -imag(IQ) — the channel filter selects band
  5. AGC uses SLOW envelope (≥1.0s time constant) — fast AGC causes pumping
  6. AGC envelope initializes HIGH (0.5) for gentle fade-in on band switch,
     not low (which causes a full-scale blast while the AGC settles)
  7. All filter operations preserve state across blocks (zi parameter)
  8. Stay in float32/complex64 — RTL-SDR data is 8-bit, float32 is plenty.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy import signal as sig


class Demodulator(ABC):
    """Base class for all demodulators."""

    def __init__(self, channel_rate_hz: float, audio_rate_hz: float = 48000.0):
        self.channel_rate_hz = channel_rate_hz
        self.audio_rate_hz = audio_rate_hz
        self._audio_decim = max(1, int(round(channel_rate_hz / audio_rate_hz)))
        self._audio_decim_phase = 0
        self._agc_fast = np.float32(0.5)
        self._agc_slow = np.float32(0.5)
        if self._audio_decim > 1:
            # Anti-alias IIR — use SOS (8th order) for better stopband
            wn = min(0.95, 0.85 / self._audio_decim)
            sos = sig.butter(8, wn, output='sos')
            self._aa_sos = sos.astype(np.float32)
            self._aa_sos_zi = sig.sosfilt_zi(self._aa_sos).astype(np.float32) * 0.0
        else:
            self._aa_sos = self._aa_sos_zi = None

    @abstractmethod
    def demodulate(self, iq: np.ndarray) -> np.ndarray:
        """Demodulate channel-filtered IQ to float32 audio [-1, 1]."""
        pass

    def _slow_agc(
        self,
        audio: np.ndarray,
        target: float = 0.12,
    ) -> np.ndarray:
        """Very slow AGC for stable output volume.

        Uses a TWO-STAGE envelope tracker:
          Stage 1 (fast peak): tracks signal peaks with ~10ms attack for
                  sudden level jumps (prevents clipping).
          Stage 2 (slow smooth): smooths the envelope with ~1.0s time
                  constant so the gain changes are extremely gentle —
                  essentially inaudible to the listener.

        Max gain is capped at 3x — enough to bring up weak stations but
        prevents the startup blast when switching bands (the envelope
        initializes at 0.5, so initial gain ≈ target/0.5 = 0.24).
        """
        if len(audio) == 0:
            return audio

        audio = np.asarray(audio, dtype=np.float32)
        level = np.abs(audio)

        rate = self.audio_rate_hz

        # Stage 1: fast peak tracker (10ms attack)
        alpha_fast = np.float32(min(0.01, 1.0 / (rate * 0.01)))
        b_fast = np.array([alpha_fast], dtype=np.float32)
        a_fast = np.array([1.0, -(1.0 - alpha_fast)], dtype=np.float32)
        zi_fast = np.array([self._agc_fast * (1.0 - alpha_fast)], dtype=np.float32)
        env_fast, zi_fast_out = sig.lfilter(b_fast, a_fast, level, zi=zi_fast)
        env_peak = np.maximum(env_fast, level * alpha_fast + env_fast * (1.0 - alpha_fast))
        self._agc_fast = np.float32(zi_fast_out[0] / (1.0 - alpha_fast))

        # Stage 2: very slow smoother (1.0s time constant)
        alpha_slow = np.float32(min(0.002, 1.0 / (rate * 1.0)))
        b_slow = np.array([alpha_slow], dtype=np.float32)
        a_slow = np.array([1.0, -(1.0 - alpha_slow)], dtype=np.float32)
        zi_slow = np.array([self._agc_slow * (1.0 - alpha_slow)], dtype=np.float32)
        env_smooth, zi_slow_out = sig.lfilter(b_slow, a_slow, env_peak, zi=zi_slow)
        self._agc_slow = np.float32(zi_slow_out[0] / (1.0 - alpha_slow))

        # Clamp envelope floor
        env_smooth = np.maximum(env_smooth, np.float32(1e-5))

        gain = np.float32(target) / env_smooth
        gain = np.minimum(gain, np.float32(3.0))
        out = audio * gain

        return np.clip(out, -1.0, 1.0).astype(np.float32)

    def _decimate_audio(self, audio: np.ndarray) -> np.ndarray:
        """Decimate from channel rate to audio rate, with persistent filter
        state AND decimation phase tracking across blocks.
        """
        if self._audio_decim <= 1 or len(audio) < 4:
            return np.asarray(audio, dtype=np.float32)
        # Anti-alias filter (SOS, 8th order, state preserved across blocks)
        filtered, self._aa_sos_zi = sig.sosfilt(
            self._aa_sos,
            np.asarray(audio, dtype=np.float32),
            zi=self._aa_sos_zi,
        )
        # Decimate with phase continuity across blocks
        D = self._audio_decim
        out = filtered[self._audio_decim_phase :: D]
        n_picked = len(out)
        next_pick = self._audio_decim_phase + n_picked * D
        self._audio_decim_phase = next_pick - len(filtered)
        return np.asarray(out, dtype=np.float32)


class FMDemodulator(Demodulator):
    """
    Wideband FM demodulator (broadcast FM).

    Matches CubicSDR / gqrx signal chain:
      1. Phase discriminator: output = arg{r*[k-1] · r[k]} / π
      2. De-emphasis: 75µs first-order IIR (North America standard)
      3. Resample from channel rate to audio rate
      4. Slow AGC (1s time constant) for consistent level across stations

    The 75µs de-emphasis rolls off high-frequency noise above ~2.1 kHz,
    matching the pre-emphasis applied at the transmitter.  This gives a
    dramatic improvement in perceived audio quality (less hiss).

    NOTE: FM uses a LOWER AGC target (0.05) than AM/SSB (0.12) because
    FM broadcast audio is heavily compressed by stations — at the same
    peak level FM sounds 2-3x louder.  The lower target equalizes
    perceived loudness so band switches are seamless.
    """

    def __init__(self, channel_rate_hz: float, audio_rate_hz: float = 48000.0):
        super().__init__(channel_rate_hz, audio_rate_hz)
        self._prev_sample = np.complex64(0)
        # De-emphasis filter: 75µs time constant (North America)
        # Applied at channel rate BEFORE audio decimation for best accuracy.
        tau = 75e-6  # seconds
        # First-order IIR: y[n] = x[n]*d + y[n-1]*(1-d), d = 1 - exp(-1/(fs*τ))
        d = np.float32(1.0 - np.exp(-1.0 / (channel_rate_hz * tau)))
        self._de_b = np.array([d], dtype=np.float32)
        self._de_a = np.array([1.0, -(1.0 - d)], dtype=np.float32)
        self._de_zi = sig.lfilter_zi(self._de_b, self._de_a).astype(np.float32) * 0.0

    def demodulate(self, iq: np.ndarray) -> np.ndarray:
        if len(iq) == 0:
            return np.array([], dtype=np.float32)

        x = np.asarray(iq, dtype=np.complex64)

        # Phase difference discriminator with block continuity
        x_with_prev = np.empty(len(x) + 1, dtype=np.complex64)
        x_with_prev[0] = self._prev_sample
        x_with_prev[1:] = x
        self._prev_sample = np.complex64(x[-1])
        phase_diff = np.angle(x_with_prev[1:] * np.conj(x_with_prev[:-1]))

        # Normalize to [-1, 1]
        audio = (phase_diff / np.float32(np.pi)).astype(np.float32)

        # De-emphasis (75µs, applied at channel rate for accuracy)
        audio, self._de_zi = sig.lfilter(
            self._de_b, self._de_a, audio, zi=self._de_zi
        )

        # Decimate to audio rate
        audio = self._decimate_audio(audio)

        return self._slow_agc(audio, target=0.05)


class AMDemodulator(Demodulator):
    """AM envelope detector matching gqrx / liquid-dsp / GNU Radio.

    Signal chain (matches gqrx rx_demod_am.cpp):
      1. Envelope detection: |IQ|  (complex_to_mag)
      2. DC-blocking IIR filter with persistent state
         (gqrx uses ff=[1, -1], fb=[1, -0.999])
      3. Decimate to audio rate
      4. Audio high-pass at 50 Hz (removes RTL-SDR carrier offset tone)
      5. Slow AGC (200ms time constant — NOT the 2ms that caused pumping)

    The DC blocker is a first-order IIR high-pass:
        H(z) = (1 - z^-1) / (1 - R·z^-1),  R = 0.999
    This continuously removes the carrier DC offset with zero
    block-boundary discontinuity (unlike per-block mean subtraction).

    The audio high-pass at 50 Hz eliminates the ~20 Hz tone caused by
    the RTL-SDR crystal PPM error (cheap oscillators have 20-50 ppm
    error, which places the AM carrier 20-50 Hz off from DC).  AM
    broadcast audio starts at ~100 Hz, so 50 Hz is safe.
    """

    def __init__(self, channel_rate_hz: float, audio_rate_hz: float = 48000.0):
        super().__init__(channel_rate_hz, audio_rate_hz)
        # DC-blocking IIR (matches gqrx: ff=[1,-1], fb=[1,-0.999])
        self._dc_b = np.array([1.0, -1.0], dtype=np.float64)
        self._dc_a = np.array([1.0, -0.999], dtype=np.float64)
        self._dc_zi = sig.lfilter_zi(self._dc_b, self._dc_a) * 0.0
        # Audio high-pass: removes carrier offset tone from PPM error.
        # 4th-order Butterworth at 50 Hz (AM broadcast audio: 100-5000 Hz)
        self._hp_sos = sig.butter(
            4, 50.0, btype='highpass', fs=audio_rate_hz, output='sos'
        ).astype(np.float32)
        self._hp_zi = sig.sosfilt_zi(self._hp_sos).astype(np.float32) * 0.0

    def demodulate(self, iq: np.ndarray) -> np.ndarray:
        if len(iq) == 0:
            return np.array([], dtype=np.float32)

        # 1. Envelope detection (magnitude)
        envelope = np.abs(iq).astype(np.float64)

        # 2. DC-blocking IIR (persistent state, float64 for precision)
        dc_removed, self._dc_zi = sig.lfilter(
            self._dc_b, self._dc_a, envelope, zi=self._dc_zi
        )
        audio = dc_removed.astype(np.float32)

        # 3. Decimate to audio rate
        audio = self._decimate_audio(audio)

        # 4. Audio high-pass (removes carrier offset from PPM error)
        if len(audio) > 0:
            audio, self._hp_zi = sig.sosfilt(
                self._hp_sos, audio, zi=self._hp_zi
            )

        return self._slow_agc(audio, target=0.12)


class SSBDemodulator(Demodulator):
    """SSB demodulator (LSB or USB).

    The channel filter in gui_main.py already selects the correct
    sideband by frequency offset:
      - USB: VFO is at carrier, low-pass selects 0..+3kHz → real(IQ)
      - LSB: -imag(IQ) extracts the lower sideband content

    DC-blocking IIR + 30 Hz audio high-pass removes carrier leak and
    PPM-offset tone.  SSB voice starts at ~200 Hz so 30 Hz is safe.
    """

    def __init__(
        self, channel_rate_hz: float, audio_rate_hz: float = 48000.0, mode: str = "usb"
    ):
        super().__init__(channel_rate_hz, audio_rate_hz)
        self.mode = mode
        # DC-blocking IIR (same as AM)
        self._dc_b = np.array([1.0, -1.0], dtype=np.float64)
        self._dc_a = np.array([1.0, -0.999], dtype=np.float64)
        self._dc_zi = sig.lfilter_zi(self._dc_b, self._dc_a) * 0.0
        # Audio high-pass: removes carrier offset tone from PPM error.
        # 4th-order Butterworth at 30 Hz (SSB voice: 200-3000 Hz)
        self._hp_sos = sig.butter(
            4, 30.0, btype='highpass', fs=audio_rate_hz, output='sos'
        ).astype(np.float32)
        self._hp_zi = sig.sosfilt_zi(self._hp_sos).astype(np.float32) * 0.0

    def demodulate(self, iq: np.ndarray) -> np.ndarray:
        if len(iq) == 0:
            return np.array([], dtype=np.float32)

        x = np.asarray(iq, dtype=np.complex64)

        if self.mode == "lsb":
            # -imag(IQ) extracts the lower sideband content
            audio = (-np.imag(x)).astype(np.float64)
        else:
            # USB: take real part of analytic signal
            audio = np.real(x).astype(np.float64)

        # DC-blocking IIR
        dc_removed, self._dc_zi = sig.lfilter(
            self._dc_b, self._dc_a, audio, zi=self._dc_zi
        )
        audio = dc_removed.astype(np.float32)

        # Decimate to audio rate
        audio = self._decimate_audio(audio)

        # Audio high-pass (removes carrier offset from PPM error)
        if len(audio) > 0:
            audio, self._hp_zi = sig.sosfilt(
                self._hp_sos, audio, zi=self._hp_zi
            )

        return self._slow_agc(audio, target=0.12)


# Channel bandwidth presets (Hz) for each mode
MODE_CHANNEL_BW = {
    "fm": 200_000,    # Broadcast FM: 200 kHz channel
    "am": 10_000,     # AM broadcast: 10 kHz
    "lsb": 3_000,     # Lower sideband: 3 kHz
    "usb": 3_000,     # Upper sideband: 3 kHz
}


def create_demodulator(mode: str, channel_rate_hz: float, audio_rate_hz: float = 48000.0) -> Demodulator:
    """Factory: create the right demodulator for the given mode."""
    mode = mode.lower()
    if mode == "fm":
        return FMDemodulator(channel_rate_hz, audio_rate_hz)
    elif mode == "am":
        return AMDemodulator(channel_rate_hz, audio_rate_hz)
    elif mode in ("usb", "lsb"):
        return SSBDemodulator(channel_rate_hz, audio_rate_hz, mode=mode)
    else:
        raise ValueError(f"Unknown mode: {mode}")
