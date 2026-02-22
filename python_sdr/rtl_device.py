from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from rtlsdr import RtlSdr
except ImportError:  # pragma: no cover - hardware environment
    RtlSdr = None  # type: ignore[misc, assignment]


@dataclass
class RtlDeviceConfig:
    device_index: int
    center_freq_hz: float
    sample_rate_hz: float
    gain_db: float
    direct_sampling_q: bool = False
    buffer_size: int = 131072  # 128k samples – better SNR + smoother FFT


class RtlDevice:
    """
    Thin wrapper around pyrtlsdr's RtlSdr to provide a clean, testable API.
    """

    def __init__(self, config: RtlDeviceConfig) -> None:
        self.config = config
        self._sdr: Optional[RtlSdr] = None

    def open(self) -> None:
        if RtlSdr is None:
            raise RuntimeError(
                "pyrtlsdr is not installed. Install it with 'pip install pyrtlsdr'."
            )

        if self._sdr is not None:
            return

        sdr = RtlSdr(self.config.device_index)
        sdr.sample_rate = self.config.sample_rate_hz
        sdr.center_freq = self.config.center_freq_hz
        sdr.gain = self.config.gain_db

        if self.config.direct_sampling_q:
            # 2 = Q branch on most RTL-SDRs
            sdr.set_direct_sampling(2)

        self._sdr = sdr

    def close(self) -> None:
        if self._sdr is not None:
            self._sdr.close()
            self._sdr = None

    def __enter__(self) -> "RtlDevice":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def set_center_frequency(self, freq_hz: float) -> None:
        if self._sdr is None:
            raise RuntimeError("Device not opened")
        self._sdr.center_freq = freq_hz
        self.config.center_freq_hz = freq_hz

    def read_samples(self, num_samples: Optional[int] = None) -> np.ndarray:
        if self._sdr is None:
            raise RuntimeError("Device not opened")
        n = num_samples or self.config.buffer_size
        samples = self._sdr.read_samples(n)
        return np.asarray(samples, dtype=np.complex64)

    def read_power_spectrum(
        self, num_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience method that returns (freq_axis_hz, power_db) for a single read.
        """
        samples = self.read_samples(num_samples)
        n = len(samples)
        window = np.hanning(n)
        windowed = samples * window
        spectrum = np.fft.fftshift(np.fft.fft(windowed))
        power = 20.0 * np.log10(np.abs(spectrum) + 1e-12)

        freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / self.config.sample_rate_hz))
        freqs_hz = freqs + self.config.center_freq_hz
        return freqs_hz, power


