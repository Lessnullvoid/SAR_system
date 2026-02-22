"""
Demodulators for FM, AM, LSB, USB.

Each demodulator expects CHANNEL-FILTERED IQ (narrow bandwidth),
not raw wideband IQ. The channel filter in gui_main.py handles that.

Signal chain:
    IIR channel filter → decimate → demod.demodulate() → audio samples

Key design rules:
  1. FM does NOT use AGC (FM has constant envelope; AGC causes
     block-rate amplitude modulation = rhythmic interference)
  2. All filter operations preserve state across blocks (zi parameter)
  3. Stay in float32/complex64 for speed — RTL-SDR data is only 8-bit,
     so float32 (23-bit mantissa) is more than sufficient precision.
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
        # Audio decimation with persistent filter state AND phase tracking
        self._audio_decim = max(1, int(round(channel_rate_hz / audio_rate_hz)))
        self._audio_decim_phase = 0  # decimation phase (offset into next block)
        if self._audio_decim > 1:
            # Anti-alias IIR (float32 for speed)
            wn = min(0.99, 0.9 / self._audio_decim)
            b, a = sig.butter(5, wn)
            self._aa_b = b.astype(np.float32)
            self._aa_a = a.astype(np.float32)
            self._aa_zi = sig.lfilter_zi(self._aa_b, self._aa_a).astype(np.float32) * 0.0
        else:
            self._aa_b = self._aa_a = self._aa_zi = None

        # Smooth AGC state (for AM/SSB only — FM never uses this)
        self._agc_gain: float = 1.0

    @abstractmethod
    def demodulate(self, iq: np.ndarray) -> np.ndarray:
        """Demodulate channel-filtered IQ to float32 audio [-1, 1]."""
        pass

    def _smooth_agc(self, audio: np.ndarray, target: float = 0.3,
                    attack: float = 0.005, release: float = 0.0005) -> np.ndarray:
        """
        Smooth sample-level AGC for AM/SSB modes.
        Tracks envelope sample-by-sample with separate attack/release rates.
        """
        if len(audio) == 0:
            return audio

        out = np.empty(len(audio), dtype=np.float32)
        gain = self._agc_gain

        for i in range(len(audio)):
            sample = float(audio[i])
            level = abs(sample)
            if level > 1e-10:
                desired = target / level
                rate = attack if desired < gain else release
                gain += rate * (desired - gain)
                gain = max(0.01, min(gain, 30.0))
            out[i] = sample * gain

        self._agc_gain = gain
        return np.clip(out, -1.0, 1.0)

    def _decimate_audio(self, audio: np.ndarray) -> np.ndarray:
        """Decimate from channel rate to audio rate, with persistent filter
        state AND decimation phase tracking across blocks.
        """
        if self._audio_decim <= 1 or len(audio) < 4:
            return np.asarray(audio, dtype=np.float32)
        # Anti-alias filter (float32, state preserved across blocks)
        filtered, self._aa_zi = sig.lfilter(
            self._aa_b, self._aa_a,
            np.asarray(audio, dtype=np.float32),
            zi=self._aa_zi,
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

    Matches CubicSDR's exact signal chain (from ModemFM.cpp + ModemAnalog.cpp):
      1. Phase discriminator: output = arg{r*[k-1] · r[k]} / π
         (CubicSDR uses liquid-dsp freqdem_create(0.5) which is identical)
      2. Resample from channel rate to audio rate
         (CubicSDR uses msresamp_rrrf with 60 dB stopband)
      3. NO de-emphasis (CubicSDR has this as optional toggle, off by default)
      4. NO AGC (CubicSDR passes autoGain=false for FM)
    """

    def __init__(self, channel_rate_hz: float, audio_rate_hz: float = 48000.0):
        super().__init__(channel_rate_hz, audio_rate_hz)
        # Previous IQ sample for phase continuity across blocks
        self._prev_sample: complex = None  # None = first block not yet seen

    def demodulate(self, iq: np.ndarray) -> np.ndarray:
        if len(iq) == 0:
            return np.array([], dtype=np.float32)

        # Stay in complex64 — plenty of precision for FM discriminator
        x = np.asarray(iq, dtype=np.complex64)

        # Phase difference discriminator with block continuity
        # Identical to CubicSDR's freqdem(kf=0.5): output = arg{r*[k-1]·r[k]} / π
        if self._prev_sample is None:
            self._prev_sample = complex(x[0])
            if len(x) < 2:
                return np.array([], dtype=np.float32)
            phase_diff = np.angle(x[1:] * np.conj(x[:-1]))
            self._prev_sample = complex(x[-1])
        else:
            x_with_prev = np.empty(len(x) + 1, dtype=np.complex64)
            x_with_prev[0] = self._prev_sample
            x_with_prev[1:] = x
            self._prev_sample = complex(x[-1])
            phase_diff = np.angle(x_with_prev[1:] * np.conj(x_with_prev[:-1]))

        # Normalize to [-1, 1] (matches liquid-dsp freqdem kf=0.5)
        audio = (phase_diff / np.float32(np.pi)).astype(np.float32)

        # Decimate to audio rate
        audio = self._decimate_audio(audio)

        # Direct output — NO de-emphasis, NO AGC (matching CubicSDR defaults)
        return np.clip(audio, -1.0, 1.0)


class AMDemodulator(Demodulator):
    """
    AM envelope detector with smooth sample-level AGC.
    """

    def __init__(self, channel_rate_hz: float, audio_rate_hz: float = 48000.0):
        super().__init__(channel_rate_hz, audio_rate_hz)

    def demodulate(self, iq: np.ndarray) -> np.ndarray:
        if len(iq) == 0:
            return np.array([], dtype=np.float32)

        # Envelope detection (float32)
        envelope = np.abs(iq).astype(np.float32)

        # Remove DC (carrier component)
        envelope -= np.float32(np.mean(envelope))

        # Decimate to audio rate
        audio = self._decimate_audio(envelope)

        # Smooth sample-level AGC
        return self._smooth_agc(audio)


class SSBDemodulator(Demodulator):
    """
    SSB demodulator (LSB or USB).
    The channel filter already selected the sideband;
    we just take the real part of the analytic signal.
    """

    def __init__(
        self, channel_rate_hz: float, audio_rate_hz: float = 48000.0, mode: str = "usb"
    ):
        super().__init__(channel_rate_hz, audio_rate_hz)
        self.mode = mode  # "usb" or "lsb"

    def demodulate(self, iq: np.ndarray) -> np.ndarray:
        if len(iq) == 0:
            return np.array([], dtype=np.float32)

        # For USB: take real part. For LSB: conjugate first then real.
        if self.mode == "lsb":
            audio = np.real(np.conj(iq)).astype(np.float32)
        else:
            audio = np.real(iq).astype(np.float32)

        # Decimate to audio rate
        audio = self._decimate_audio(audio)

        # Smooth sample-level AGC
        return self._smooth_agc(audio)


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
