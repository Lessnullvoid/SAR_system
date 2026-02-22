"""
Frequency band definitions for seismo-EM monitoring.

Each band corresponds to a row from the monitoring plan (README §Frequency
Monitoring Plan).  The ``BANDS`` list is used by the feature extractor to
slice the wideband spectrum into scientifically relevant sub-bands.

Antenna mapping
───────────────
  loop_antenna  → VLF, LF, MF, HF  (direct-sampling Q-branch)
  fm_broadcast  → FM broadcast 88–108 MHz
  discone       → VHF, 2 m, 70 cm
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Band:
    """One monitored frequency band."""

    name: str               # short label, e.g. "VLF"
    description: str        # scientific role
    freq_lo_hz: float       # lower edge (Hz)
    freq_hi_hz: float       # upper edge (Hz)
    antenna: str            # which antenna key covers this band
    priority: int = 1       # 1 = primary, 2 = secondary interest

    @property
    def center_hz(self) -> float:
        return (self.freq_lo_hz + self.freq_hi_hz) / 2.0

    @property
    def bandwidth_hz(self) -> float:
        return self.freq_hi_hz - self.freq_lo_hz


# ── Band definitions (from README §Frequency Monitoring Plan) ─────────

BANDS: List[Band] = [
    # ── Loop antenna (RTL-SDR #1, direct-sampling) ──
    Band(
        name="VLF",
        description="SID, beacon perturbations",
        freq_lo_hz=300.0,
        freq_hi_hz=30_000.0,
        antenna="loop_antenna",
        priority=1,
    ),
    Band(
        name="LF",
        description="DCF77 analysis, conductivity shifts",
        freq_lo_hz=30_000.0,
        freq_hi_hz=300_000.0,
        antenna="loop_antenna",
        priority=1,
    ),
    Band(
        name="MF",
        description="AM-band propagation",
        freq_lo_hz=300_000.0,
        freq_hi_hz=1_700_000.0,
        antenna="loop_antenna",
        priority=1,
    ),
    Band(
        name="HF",
        description="Ionospheric reflection anomalies",
        freq_lo_hz=2_000_000.0,
        freq_hi_hz=30_000_000.0,
        antenna="loop_antenna",
        priority=1,
    ),
    # ── Discone / VHF antenna (RTL-SDR #2, quadrature) ──
    Band(
        name="VHF",
        description="Atmospheric RF anomalies",
        freq_lo_hz=30_000_000.0,
        freq_hi_hz=300_000_000.0,
        antenna="discone",
        priority=2,
    ),
    Band(
        name="FM",
        description="Multipath fluctuations",
        freq_lo_hz=88_000_000.0,
        freq_hi_hz=108_000_000.0,
        antenna="fm_broadcast",
        priority=1,
    ),
    Band(
        name="2m",
        description="Ham beacon stability",
        freq_lo_hz=144_000_000.0,
        freq_hi_hz=148_000_000.0,
        antenna="discone",
        priority=1,
    ),
    Band(
        name="70cm",
        description="RF noise mapping",
        freq_lo_hz=430_000_000.0,
        freq_hi_hz=450_000_000.0,
        antenna="discone",
        priority=2,
    ),
]


def bands_in_range(center_hz: float, sample_rate_hz: float) -> List[Band]:
    """Return bands that overlap with the current SDR tuning window.

    Parameters
    ----------
    center_hz : float
        Current LO (center) frequency.
    sample_rate_hz : float
        SDR sample rate (bandwidth = sample_rate_hz, centered on LO).

    Returns
    -------
    list[Band]
        Bands whose [lo, hi] interval overlaps [center - sr/2, center + sr/2].
    """
    half_bw = sample_rate_hz / 2.0
    view_lo = center_hz - half_bw
    view_hi = center_hz + half_bw
    return [
        b for b in BANDS
        if b.freq_lo_hz < view_hi and b.freq_hi_hz > view_lo
    ]


def band_by_name(name: str) -> Band:
    """Look up a band by its short name (case-insensitive)."""
    key = name.strip().upper()
    for b in BANDS:
        if b.name.upper() == key:
            return b
    raise KeyError(f"Unknown band: {name!r}")
