"""
NOAA Space Weather and NWS Weather API client.

Fetches two types of data:

1. **Geomagnetic Kp index** — from NOAA SWPC (Space Weather Prediction Center).
   Used to assess global geomagnetic disturbance level.
   https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json

2. **Dst index (near-real-time)** — from NOAA SWPC.
   Indicates the intensity of the globally symmetric part of the
   equatorial ring current.
   https://services.swpc.noaa.gov/products/kyoto-dst.json

3. **Local weather** — from NWS API (api.weather.gov).
   Used as environmental context.
   https://api.weather.gov/

Usage
-----
    from sar.ingest.noaa_client import fetch_kp_index, fetch_dst_index
    kp_data = fetch_kp_index()
    dst_data = fetch_dst_index()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import requests

log = logging.getLogger(__name__)

# ── NOAA SWPC endpoints ──────────────────────────────────────────────

_KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
_DST_URL = "https://services.swpc.noaa.gov/products/kyoto-dst.json"
_SOLAR_WIND_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-2-hour.json"


@dataclass
class KpReading:
    """One Kp index reading."""
    time_tag: str      # "2026-02-12 12:00:00"
    kp: float          # 0-9 scale
    kp_fraction: float # decimal Kp (e.g., 2.33)

@dataclass
class DstReading:
    """One Dst index reading."""
    time_tag: str
    dst_nt: float      # nanotesla

@dataclass
class SolarWind:
    """Solar wind magnetic field reading."""
    time_tag: str
    bz_gsm_nt: float   # Bz component in GSM coordinates (negative = geoeffective)
    bt_nt: float        # Total field magnitude


def fetch_kp_index(timeout: float = 15.0) -> List[KpReading]:
    """Fetch recent Kp index values from NOAA SWPC.

    Returns the last ~3 days of 3-hour Kp values.
    """
    try:
        resp = requests.get(_KP_URL, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("NOAA Kp fetch failed: %s", exc)
        return []

    readings = []
    # First row is header: ["time_tag", "Kp", "Kp_fraction", ...]
    for row in data[1:]:
        try:
            readings.append(KpReading(
                time_tag=str(row[0]),
                kp=float(row[1]),
                kp_fraction=float(row[2]) if len(row) > 2 else float(row[1]),
            ))
        except (ValueError, IndexError):
            continue

    log.info("NOAA Kp: %d readings", len(readings))
    return readings


def fetch_dst_index(timeout: float = 15.0) -> List[DstReading]:
    """Fetch recent Dst index values from NOAA/Kyoto.

    Returns the last ~30 days of hourly Dst values.
    """
    try:
        resp = requests.get(_DST_URL, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("NOAA Dst fetch failed: %s", exc)
        return []

    readings = []
    # First row is header
    for row in data[1:]:
        try:
            readings.append(DstReading(
                time_tag=str(row[0]),
                dst_nt=float(row[1]),
            ))
        except (ValueError, IndexError):
            continue

    log.info("NOAA Dst: %d readings", len(readings))
    return readings


def fetch_solar_wind(timeout: float = 15.0) -> List[SolarWind]:
    """Fetch last 2 hours of solar wind magnetic field data."""
    try:
        resp = requests.get(_SOLAR_WIND_URL, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("NOAA Solar Wind fetch failed: %s", exc)
        return []

    readings = []
    for row in data[1:]:
        try:
            readings.append(SolarWind(
                time_tag=str(row[0]),
                bz_gsm_nt=float(row[3]) if row[3] else 0.0,
                bt_nt=float(row[6]) if len(row) > 6 and row[6] else 0.0,
            ))
        except (ValueError, IndexError):
            continue

    log.info("NOAA SolarWind: %d readings", len(readings))
    return readings


def get_current_kp() -> float:
    """Get the most recent Kp value (convenience function)."""
    readings = fetch_kp_index()
    if readings:
        return readings[-1].kp
    return 0.0


def get_current_dst() -> float:
    """Get the most recent Dst value in nT (convenience function)."""
    readings = fetch_dst_index()
    if readings:
        return readings[-1].dst_nt
    return 0.0
