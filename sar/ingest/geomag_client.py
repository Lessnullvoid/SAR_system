"""
Geomagnetic data client — real-time magnetometer data.

Sources:
- INTERMAGNET (via NOAA): Near-real-time magnetometer data from observatories.
- NOAA SWPC magnetometer plots:
    https://services.swpc.noaa.gov/products/solar-wind/mag-2-hour.json

For the California corridor, the most relevant observatories are:
- FRN (Fresno, CA)  — directly on the fault
- TUC (Tucson, AZ)  — southern anchor
- NEW (Newport, WA) — northern reference

In Phase 4+, we may use the INTERMAGNET REST API.  For now, we rely
on NOAA SWPC summaries which are easier to access and don't require
API keys.

Usage
-----
    from sar.ingest.geomag_client import get_magnetic_state
    mag = get_magnetic_state()
    print(mag.kp_index, mag.dst_index)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .noaa_client import get_current_kp, get_current_dst

log = logging.getLogger(__name__)


@dataclass
class MagneticReading:
    """Current global geomagnetic conditions."""
    kp_index: float       # 0–9 scale
    dst_nt: float         # Dst index in nT (negative = storm)
    storm_level: str      # "quiet", "unsettled", "minor", "moderate", "severe"


def _classify_storm(kp: float, dst: float) -> str:
    """Classify geomagnetic storm level."""
    if kp >= 7 or dst <= -200:
        return "severe"
    if kp >= 5 or dst <= -100:
        return "moderate"
    if kp >= 4 or dst <= -50:
        return "minor"
    if kp >= 3 or dst <= -30:
        return "unsettled"
    return "quiet"


def get_magnetic_state() -> MagneticReading:
    """Get current global geomagnetic conditions.

    Combines Kp and Dst from NOAA SWPC into a single reading.
    """
    kp = get_current_kp()
    dst = get_current_dst()
    level = _classify_storm(kp, dst)

    log.info("Geomag state: Kp=%.1f, Dst=%.0f nT, %s", kp, dst, level)
    return MagneticReading(kp_index=kp, dst_nt=dst, storm_level=level)
