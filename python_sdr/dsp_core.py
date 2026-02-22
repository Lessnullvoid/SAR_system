"""
DSP core: FFT for display, channel filtering, and decimation.

Signal chain (like SDR++ / uSDR / CubicSDR):
    RTL-SDR IQ (wideband, e.g. 2.4 MHz)
      ├── remove_dc() → compute_power_spectrum()  → spectrum + waterfall
      └── remove_dc() → channel_filter()           → narrow IQ for demod
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import signal as sig


@dataclass
class FftConfig:
    fft_size: int = 4096
    window: str = "hann"


def remove_dc(samples: np.ndarray) -> np.ndarray:
    """
    Remove DC offset from IQ samples.
    RTL-SDR has a known DC spike at center frequency.
    Every real SDR app (SDR++, CubicSDR, GQRX) does this.
    """
    return samples - np.mean(samples)


def compute_power_spectrum(
    samples: np.ndarray, sample_rate_hz: float, cfg: FftConfig | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (freq_offset_hz, power_db) for display.
    Returns baseband frequencies centered on 0 Hz; caller adds center freq.
    DC is removed before FFT.
    """
    if cfg is None:
        cfg = FftConfig()

    n = min(len(samples), cfg.fft_size)
    x = samples[:n]

    # Remove DC offset (kills the center spike)
    x = x - np.mean(x)

    if cfg.window == "hann":
        w = np.hanning(n)
    elif cfg.window == "blackmanharris":
        w = np.blackman(n)
    else:
        w = np.ones(n, dtype=float)

    xw = x * w
    spectrum = np.fft.fftshift(np.fft.fft(xw, n=n))
    power_db = 20.0 * np.log10(np.abs(spectrum) + 1e-12)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sample_rate_hz))
    return freqs, power_db


def design_channel_filter(
    sample_rate_hz: float, channel_bw_hz: float, num_taps: int = 101
) -> np.ndarray:
    """
    Design a low-pass FIR filter for channel selection.
    cutoff = channel_bw_hz / 2, relative to sample_rate_hz.
    """
    cutoff = channel_bw_hz / 2.0
    nyquist = sample_rate_hz / 2.0
    normalized_cutoff = cutoff / nyquist
    normalized_cutoff = np.clip(normalized_cutoff, 0.001, 0.999)
    taps = sig.firwin(num_taps, normalized_cutoff, window="hamming")
    return taps


def channel_filter_and_decimate(
    iq_samples: np.ndarray,
    sample_rate_hz: float,
    channel_bw_hz: float,
    filter_taps: np.ndarray | None = None,
) -> Tuple[np.ndarray, float]:
    """
    Extract a channel from wideband IQ:
    1. Remove DC offset
    2. Low-pass filter to channel_bw_hz / 2
    3. Decimate to just above channel_bw_hz

    Returns (filtered_iq, new_sample_rate_hz).
    """
    if len(iq_samples) == 0:
        return iq_samples, sample_rate_hz

    # Remove DC offset first (critical for clean demod)
    iq_clean = iq_samples - np.mean(iq_samples)

    # Calculate decimation factor.
    # Cap so output rate never drops below 48 kHz (audio rate).
    max_decim = max(1, int(sample_rate_hz / 48000.0))
    decim = max(1, min(int(sample_rate_hz / channel_bw_hz), max_decim))

    # Design filter if not provided
    if filter_taps is None:
        filter_taps = design_channel_filter(sample_rate_hz, channel_bw_hz)

    # Apply FIR filter
    filtered = sig.lfilter(filter_taps, 1.0, iq_clean)

    # Decimate (just take every Nth sample -- filter already applied)
    decimated = filtered[::decim]
    new_rate = sample_rate_hz / decim

    return decimated, new_rate
