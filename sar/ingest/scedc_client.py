"""
SCEDC — Southern California Earthquake Data Center client.

Fetches earthquake data from the SCEDC FDSN event web service at Caltech.
Complements the USGS feed with higher-resolution Southern California seismic
data going back to 1932.

Data source
-----------
  SCEDC — Southern California Earthquake Data Center
  https://service.scedc.caltech.edu/fdsnws/event/1/

  No API key required.  Returns text/CSV.  Rate limit: reasonable use.
  Catalog: SCEDC (default), covers Southern California.

Usage
-----
    quakes = fetch_scedc_events(days=7, min_mag=2.0)
    # → list of SCEDCEvent with lat, lon, depth, mag, time
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List

log = logging.getLogger(__name__)

_BASE_URL = "https://service.scedc.caltech.edu/fdsnws/event/1/query"

# Bounding box for Southern California (SAF southern + central segments)
_MIN_LAT = 32.0
_MAX_LAT = 37.0
_MIN_LON = -121.0
_MAX_LON = -114.5


@dataclass
class SCEDCEvent:
    """One seismic event from SCEDC catalog."""
    event_id: str
    time: str            # ISO format
    lat: float
    lon: float
    depth_km: float
    magnitude: float
    mag_type: str = ""
    place: str = ""


def fetch_scedc_events(
    days: int = 7,
    min_mag: float = 2.0,
    max_results: int = 500,
) -> List[SCEDCEvent]:
    """Fetch recent seismic events from SCEDC.

    Parameters
    ----------
    days : int
        Look-back window in days.
    min_mag : float
        Minimum magnitude filter.
    max_results : int
        Maximum number of events to return.

    Returns
    -------
    list[SCEDCEvent]
    """
    import requests

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)

    params = {
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": now.strftime("%Y-%m-%dT%H:%M:%S"),
        "minlatitude": _MIN_LAT,
        "maxlatitude": _MAX_LAT,
        "minlongitude": _MIN_LON,
        "maxlongitude": _MAX_LON,
        "minmagnitude": min_mag,
        "limit": max_results,
        "format": "text",
        "orderby": "time",
    }

    try:
        resp = requests.get(_BASE_URL, params=params, timeout=30)
        if resp.status_code == 204:
            # No events found
            log.info("SCEDC: no events in last %d days (M>=%.1f)", days, min_mag)
            return []
        resp.raise_for_status()
    except Exception as exc:
        log.warning("SCEDC API error: %s", exc)
        return []

    lines = resp.text.strip().split("\n")
    if len(lines) < 2:
        return []

    # Parse header
    header = lines[0].split("|")
    results: List[SCEDCEvent] = []

    for line in lines[1:]:
        fields = line.split("|")
        if len(fields) < len(header):
            continue

        row = dict(zip(header, fields))

        try:
            results.append(SCEDCEvent(
                event_id=row.get("EventID", ""),
                time=row.get("Time", ""),
                lat=float(row.get("Latitude", 0)),
                lon=float(row.get("Longitude", 0)),
                depth_km=float(row.get("Depth/km", 0)),
                magnitude=float(row.get("Magnitude", 0)),
                mag_type=row.get("MagType", ""),
                place=row.get("EventLocationName", ""),
            ))
        except (ValueError, TypeError):
            continue

    log.info("SCEDC: fetched %d events (last %d days, M>=%.1f)",
             len(results), days, min_mag)
    return results
