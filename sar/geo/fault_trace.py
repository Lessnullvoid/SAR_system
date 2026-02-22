"""
San Andreas Fault trace geometry.

The trace is defined as a sequence of (latitude, longitude) waypoints
running roughly NW → SE through California.  These are simplified from
the USGS Quaternary Fault and Fold Database.

Functions
---------
fault_trace_lonlat()
    Returns the fault trace as a list of (lon, lat) tuples.
fault_corridor(buffer_km=25.0)
    Returns a buffered corridor polygon around the trace (shapely Polygon
    in lon/lat coordinates).
"""
from __future__ import annotations

from typing import List, Tuple

from shapely.geometry import LineString, Polygon
from shapely.ops import transform
import pyproj

# ── Simplified San Andreas Fault trace (lat, lon) waypoints ───────────
# From the Salton Sea (SE) to Cape Mendocino (NW).
# ~30 key waypoints capturing the main trace geometry.

_TRACE_LATLON: List[Tuple[float, float]] = [
    # Southern section (Salton Sea → Cajon Pass)
    (33.18, -115.60),  # Salton Sea (SE terminus)
    (33.30, -115.72),
    (33.50, -115.90),
    (33.70, -116.08),
    (33.92, -116.30),  # Palm Springs area
    (34.05, -116.45),
    (34.15, -116.62),
    (34.20, -116.82),
    (34.25, -117.10),
    (34.30, -117.35),  # Cajon Pass

    # Central section (Cajon Pass → Parkfield)
    (34.42, -117.65),
    (34.55, -117.85),
    (34.68, -118.08),
    (34.80, -118.40),
    (34.88, -118.62),
    (34.95, -118.88),
    (35.05, -119.15),
    (35.10, -119.45),
    (35.30, -119.65),
    (35.60, -119.90),
    (35.90, -120.30),  # Parkfield

    # Northern section (Parkfield → Cape Mendocino)
    (36.15, -120.55),
    (36.45, -120.85),
    (36.75, -121.20),
    (37.00, -121.50),
    (37.30, -121.80),
    (37.50, -122.00),
    (37.75, -122.25),  # San Francisco Peninsula
    (38.00, -122.50),
    (38.40, -122.80),
    (38.80, -123.00),
    (39.20, -123.25),
    (39.60, -123.45),
    (40.00, -123.70),
    (40.30, -124.10),  # Cape Mendocino (NW terminus)
]


def fault_trace_lonlat() -> List[Tuple[float, float]]:
    """Return the fault trace as (lon, lat) tuples (GeoJSON order)."""
    return [(lon, lat) for lat, lon in _TRACE_LATLON]


def fault_trace_line() -> LineString:
    """Return the fault trace as a Shapely LineString (lon, lat)."""
    return LineString(fault_trace_lonlat())


def fault_corridor(buffer_km: float = 25.0) -> Polygon:
    """Return a buffered corridor polygon around the fault trace.

    Parameters
    ----------
    buffer_km : float
        Half-width of the corridor in kilometres (default 25 km = 50 km total).

    Returns
    -------
    shapely.geometry.Polygon
        Corridor polygon in lon/lat (WGS84) coordinates.
    """
    # Project to a local metric CRS, buffer, project back
    wgs84 = pyproj.CRS("EPSG:4326")
    # California Albers Equal Area — good for distance calculations along the fault
    aea_ca = pyproj.CRS("EPSG:3310")

    to_metric = pyproj.Transformer.from_crs(wgs84, aea_ca, always_xy=True).transform
    to_lonlat = pyproj.Transformer.from_crs(aea_ca, wgs84, always_xy=True).transform

    trace = fault_trace_line()
    trace_m = transform(to_metric, trace)
    corridor_m = trace_m.buffer(buffer_km * 1000.0, cap_style="round")
    corridor_ll = transform(to_lonlat, corridor_m)

    return corridor_ll


def fault_length_km() -> float:
    """Approximate length of the fault trace in kilometres."""
    wgs84 = pyproj.CRS("EPSG:4326")
    aea_ca = pyproj.CRS("EPSG:3310")
    to_metric = pyproj.Transformer.from_crs(wgs84, aea_ca, always_xy=True).transform

    trace = fault_trace_line()
    trace_m = transform(to_metric, trace)
    return trace_m.length / 1000.0
