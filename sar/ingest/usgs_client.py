"""
USGS Earthquake Catalog client.

Fetches recent earthquakes from the USGS GeoJSON feed and returns them
as structured dicts.  Supports two endpoints:

1. **Real-time feeds** — pre-built GeoJSON files updated every minute.
   Best for routine polling (no query params, fast, reliable).
   https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php

2. **FDSN query** — custom bounding-box + time-range queries.
   Used for initial tile population or focused searches.
   https://earthquake.usgs.gov/fdsnws/event/1/query

Usage
-----
    from sar.ingest.usgs_client import fetch_recent_earthquakes
    quakes = fetch_recent_earthquakes(period="day", min_mag=1.0)
    for q in quakes:
        print(q["mag"], q["place"], q["lat"], q["lon"])
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)

# ── USGS real-time feed URLs ──────────────────────────────────────────

_FEED_BASE = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"

# period → (all magnitudes URL, significant only URL)
_FEEDS = {
    "hour":  f"{_FEED_BASE}/all_hour.geojson",
    "day":   f"{_FEED_BASE}/all_day.geojson",
    "week":  f"{_FEED_BASE}/all_week.geojson",
    "month": f"{_FEED_BASE}/all_month.geojson",
}

_FDSN_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


@dataclass
class Earthquake:
    """One earthquake event."""
    event_id: str
    time_ms: int            # origin time (ms since epoch)
    lat: float
    lon: float
    depth_km: float
    mag: float
    mag_type: str           # e.g. "ml", "mw"
    place: str              # human description
    status: str             # "automatic" or "reviewed"
    url: str                # USGS event page

    @property
    def time_utc(self) -> str:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(
            self.time_ms / 1000.0, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

    @property
    def age_hours(self) -> float:
        return (time.time() - self.time_ms / 1000.0) / 3600.0


def _parse_feature(feat: dict) -> Optional[Earthquake]:
    """Parse one GeoJSON feature into an Earthquake object."""
    try:
        props = feat["properties"]
        coords = feat["geometry"]["coordinates"]
        return Earthquake(
            event_id=feat.get("id", ""),
            time_ms=props.get("time", 0),
            lon=coords[0],
            lat=coords[1],
            depth_km=coords[2] if len(coords) > 2 else 0.0,
            mag=props.get("mag", 0.0) or 0.0,
            mag_type=props.get("magType", "") or "",
            place=props.get("place", "") or "",
            status=props.get("status", "") or "",
            url=props.get("url", "") or "",
        )
    except (KeyError, IndexError, TypeError) as exc:
        log.debug("Failed to parse earthquake feature: %s", exc)
        return None


# ── Public API ────────────────────────────────────────────────────────

def fetch_recent_earthquakes(
    period: str = "day",
    min_mag: float = 0.0,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    timeout: float = 15.0,
) -> List[Earthquake]:
    """Fetch recent earthquakes from USGS real-time feed.

    Parameters
    ----------
    period : str
        One of "hour", "day", "week", "month".
    min_mag : float
        Minimum magnitude filter (applied client-side).
    bbox : (lon_min, lat_min, lon_max, lat_max), optional
        Bounding box filter (applied client-side).
    timeout : float
        HTTP request timeout in seconds.

    Returns
    -------
    list[Earthquake]
        Earthquakes sorted by time (newest first).
    """
    url = _FEEDS.get(period, _FEEDS["day"])

    try:
        from . import fetch_with_retry
        resp = fetch_with_retry(url, timeout=timeout, retries=2)
        data = resp.json()
    except Exception as exc:
        log.warning("USGS feed fetch failed: %s", exc)
        return []

    features = data.get("features", [])
    quakes: List[Earthquake] = []

    for feat in features:
        q = _parse_feature(feat)
        if q is None:
            continue
        if q.mag < min_mag:
            continue
        if bbox:
            lon_min, lat_min, lon_max, lat_max = bbox
            if not (lon_min <= q.lon <= lon_max and lat_min <= q.lat <= lat_max):
                continue
        quakes.append(q)

    # Sort newest first
    quakes.sort(key=lambda q: q.time_ms, reverse=True)

    log.info(
        "USGS %s feed: %d earthquakes (min_mag=%.1f, bbox=%s)",
        period, len(quakes), min_mag,
        f"{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}" if bbox else "all",
    )
    return quakes


def fetch_fault_corridor_earthquakes(
    period: str = "week",
    min_mag: float = 1.0,
) -> List[Earthquake]:
    """Fetch earthquakes within the San Andreas corridor bounding box.

    Uses a generous bounding box covering the full fault system:
    Lat 32.5–40.5, Lon -125.0 to -115.0.
    """
    return fetch_recent_earthquakes(
        period=period,
        min_mag=min_mag,
        bbox=(-125.0, 32.5, -115.0, 40.5),
    )
