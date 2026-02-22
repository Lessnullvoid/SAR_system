"""
Feature extraction from power spectra for seismo-EM anomaly detection.

For each monitored band visible in the current spectrum, we extract 8
features that capture different aspects of the RF environment:

  1. mean_power   – average power in the band (dB)
  2. max_power    – peak power (dB)
  3. noise_floor  – 10th percentile power ≈ noise floor (dB)
  4. std_power    – standard deviation of power (dB) — signal variability
  5. snr          – signal-to-noise ratio (max − noise_floor, dB)
  6. entropy      – spectral entropy (0 = pure tone, high = wideband noise)
  7. occupancy    – fraction of band above (noise_floor + 6 dB)
  8. kurtosis     – spectral kurtosis (>3 = impulsive / transient signals)

These 8 features per band are fed into the anomaly detector.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .bands import Band, bands_in_range

# Number of features per band
N_FEATURES = 8

# Feature names (for display / logging)
FEATURE_NAMES = [
    "mean_power",
    "max_power",
    "noise_floor",
    "std_power",
    "snr",
    "entropy",
    "occupancy",
    "kurtosis",
]


@dataclass
class BandFeatures:
    """Feature vector for a single band at a single time instant."""

    band_name: str
    timestamp: float                     # UNIX epoch
    values: np.ndarray                   # shape (N_FEATURES,), float32
    center_hz: float = 0.0              # SDR center freq when captured
    sample_rate_hz: float = 0.0         # SDR sample rate when captured
    peak_freq_hz: float = 0.0          # frequency of strongest signal in band

    def as_dict(self) -> Dict[str, float]:
        return {n: float(self.values[i]) for i, n in enumerate(FEATURE_NAMES)}


def extract_features(
    freqs_hz: np.ndarray,
    power_db: np.ndarray,
    center_hz: float,
    sample_rate_hz: float,
    timestamp: Optional[float] = None,
) -> List[BandFeatures]:
    """Extract features for every monitored band visible in the spectrum.

    Parameters
    ----------
    freqs_hz : ndarray
        Frequency axis of the power spectrum (Hz), monotonically increasing.
    power_db : ndarray
        Power spectrum (dB), same length as *freqs_hz*.
    center_hz : float
        Current LO center frequency.
    sample_rate_hz : float
        SDR sample rate (determines visible bandwidth).
    timestamp : float, optional
        UNIX timestamp; defaults to ``time.time()``.

    Returns
    -------
    list[BandFeatures]
        One entry per band that overlaps the current spectrum.
    """
    if timestamp is None:
        timestamp = time.time()

    visible = bands_in_range(center_hz, sample_rate_hz)
    if not visible:
        return []

    results: List[BandFeatures] = []
    for band in visible:
        feats = _extract_band(freqs_hz, power_db, band, timestamp,
                              center_hz, sample_rate_hz)
        if feats is not None:
            results.append(feats)
    return results


def _extract_band(
    freqs_hz: np.ndarray,
    power_db: np.ndarray,
    band: Band,
    timestamp: float,
    center_hz: float,
    sample_rate_hz: float,
) -> Optional[BandFeatures]:
    """Extract features for a single band from the spectrum."""
    # Select frequency bins inside this band
    mask = (freqs_hz >= band.freq_lo_hz) & (freqs_hz <= band.freq_hi_hz)
    n_bins = int(np.sum(mask))
    if n_bins < 4:
        return None  # not enough data for meaningful features

    pwr = power_db[mask].astype(np.float64)
    freqs_in_band = freqs_hz[mask]

    # 1. Mean power
    mean_pwr = float(np.mean(pwr))

    # 2. Max power + its frequency (used to auto-tune VFO to anomalies)
    peak_idx = int(np.argmax(pwr))
    max_pwr = float(pwr[peak_idx])
    peak_freq = float(freqs_in_band[peak_idx])

    # 3. Noise floor (10th percentile)
    noise_floor = float(np.percentile(pwr, 10))

    # 4. Standard deviation
    std_pwr = float(np.std(pwr))

    # 5. SNR (peak above noise floor)
    snr = max_pwr - noise_floor

    # 6. Spectral entropy
    #    Convert to linear power, normalise to a probability distribution,
    #    then compute Shannon entropy.  Normalise by log(N) so entropy ∈ [0,1].
    pwr_lin = 10.0 ** (pwr / 10.0)
    pwr_sum = np.sum(pwr_lin)
    if pwr_sum > 0:
        p = pwr_lin / pwr_sum
        p = p[p > 0]  # avoid log(0)
        entropy = float(-np.sum(p * np.log(p)) / np.log(max(n_bins, 2)))
    else:
        entropy = 0.0

    # 7. Band occupancy (fraction of bins > noise_floor + 6 dB)
    occupancy = float(np.mean(pwr > noise_floor + 6.0))

    # 8. Spectral kurtosis (excess kurtosis; normal = 0, impulsive > 0)
    if std_pwr > 1e-12:
        kurtosis = float(np.mean(((pwr - mean_pwr) / std_pwr) ** 4) - 3.0)
    else:
        kurtosis = 0.0

    values = np.array(
        [mean_pwr, max_pwr, noise_floor, std_pwr, snr, entropy, occupancy, kurtosis],
        dtype=np.float32,
    )

    return BandFeatures(
        band_name=band.name,
        timestamp=timestamp,
        values=values,
        center_hz=center_hz,
        sample_rate_hz=sample_rate_hz,
        peak_freq_hz=peak_freq,
    )
