"""
Vector overlay data for the fault corridor map.

Provides geographic features as projected coordinates for rendering
on the QGraphicsScene:
  - Polylines: coastline, highways, borders, secondary faults
  - Point labels: cities, towns, geographic features, road names

Data is embedded as simplified coordinate lists — no external file
dependencies.  All features within the San Andreas corridor bounding
box (lon -125 to -115, lat 32.5 to 40.5).

Usage
-----
    from sar.geo.vector_data import get_vector_layers, get_labels
    layers = get_vector_layers()   # polylines
    labels = get_labels()          # point labels
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pyproj

log = logging.getLogger(__name__)

WGS84 = pyproj.CRS("EPSG:4326")
AEA_CA = pyproj.CRS("EPSG:3310")
_to_m = pyproj.Transformer.from_crs(WGS84, AEA_CA, always_xy=True)


def _project(coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Convert (lon, lat) pairs to (x_m, y_m) in EPSG:3310."""
    result = []
    for lon, lat in coords:
        x, y = _to_m.transform(lon, lat)
        result.append((x, y))
    return result


def _project_pt(lon: float, lat: float) -> Tuple[float, float]:
    """Project a single (lon, lat) to (x_m, y_m)."""
    return _to_m.transform(lon, lat)


@dataclass
class MapLabel:
    """A text label to render on the map."""
    name: str
    x_m: float
    y_m: float
    category: str       # "city", "town", "road", "feature", "fault", "region"
    font_size: float    # in scene units (will be scaled)
    rotation: float = 0.0  # degrees


# ═══════════════════════════════════════════════════════════════════════
# POLYLINE DATA
# ═══════════════════════════════════════════════════════════════════════

# ── California coastline (simplified) ────────────────────────────────

_COASTLINE_LONLAT = [
    (-117.12, 32.53), (-117.18, 32.72), (-117.25, 32.88),
    (-117.36, 33.10), (-117.59, 33.38), (-117.88, 33.60),
    (-117.93, 33.63), (-118.00, 33.72), (-118.19, 33.77),
    (-118.28, 33.74), (-118.52, 33.98), (-118.81, 34.03),
    (-119.22, 34.21), (-119.70, 34.40), (-120.47, 34.77),
    (-120.65, 35.12), (-121.00, 36.00), (-121.80, 36.60),
    (-122.00, 36.95), (-122.33, 37.57), (-122.49, 37.78),
    (-122.43, 37.83), (-122.48, 37.89), (-122.97, 38.30),
    (-123.42, 38.80), (-123.78, 39.42), (-124.10, 40.00),
    (-124.36, 40.44),
]

# ── Highways ─────────────────────────────────────────────────────────

_I5_LONLAT = [
    (-117.05, 32.55), (-117.16, 32.72), (-117.15, 33.13),
    (-117.65, 33.53), (-117.78, 33.68), (-118.17, 33.77),
    (-118.24, 34.05), (-118.38, 34.20), (-118.50, 34.40),
    (-118.72, 34.73), (-118.87, 34.90), (-119.05, 35.05),
    (-119.28, 35.39), (-119.64, 35.90), (-120.06, 36.33),
    (-120.37, 36.78), (-120.73, 37.28), (-121.15, 37.65),
    (-121.29, 37.95), (-121.50, 38.60), (-122.00, 39.15),
    (-122.19, 40.00), (-122.39, 40.58),
]

_I15_LONLAT = [
    (-117.05, 32.55), (-117.10, 32.78), (-117.16, 33.13),
    (-117.25, 33.53), (-117.42, 33.85), (-117.45, 34.06),
    (-117.30, 34.33), (-117.30, 34.55), (-116.98, 34.84),
    (-115.90, 35.47), (-115.14, 36.17),
]

_I10_LONLAT = [
    (-118.24, 34.05), (-117.90, 34.00), (-117.65, 33.98),
    (-117.38, 33.97), (-117.05, 33.95), (-116.55, 33.83),
    (-116.18, 33.77), (-115.52, 33.65), (-115.00, 33.52),
]

_US101_LONLAT = [
    (-118.25, 34.06), (-118.38, 34.15), (-118.60, 34.18),
    (-118.80, 34.22), (-119.18, 34.28), (-119.70, 34.43),
    (-120.20, 34.63), (-120.46, 34.95), (-120.60, 35.27),
    (-120.70, 35.60), (-120.93, 35.97), (-121.40, 36.37),
    (-121.65, 36.60), (-121.80, 36.67), (-121.90, 36.97),
    (-122.04, 37.37), (-122.40, 37.78), (-122.46, 37.89),
    (-122.70, 38.04), (-122.72, 38.33), (-122.71, 38.50),
    (-123.00, 38.80), (-123.34, 39.15), (-123.75, 39.76),
    (-123.90, 40.15), (-124.15, 40.80),
]

_CA99_LONLAT = [
    (-119.28, 35.39), (-119.37, 35.75), (-119.62, 36.30),
    (-119.78, 36.74), (-120.00, 37.00), (-120.46, 37.50),
    (-120.98, 37.96), (-121.26, 38.10), (-121.50, 38.58),
]

_I8_LONLAT = [
    (-117.16, 32.72), (-116.90, 32.75), (-116.60, 32.68),
    (-116.20, 32.67), (-115.70, 32.72), (-115.50, 32.85),
]

# ── State boundaries ─────────────────────────────────────────────────

_CA_NV_BORDER_LONLAT = [
    (-120.00, 39.00), (-120.00, 39.50), (-120.00, 40.00),
    (-120.00, 40.50), (-120.00, 41.00), (-120.00, 41.50),
    (-120.00, 42.00),
]

_CA_OR_BORDER_LONLAT = [
    (-124.20, 42.00), (-123.00, 42.00), (-122.00, 42.00),
    (-121.00, 42.00), (-120.00, 42.00),
]

_CA_MX_BORDER_LONLAT = [
    (-117.12, 32.54), (-116.50, 32.62), (-116.10, 32.62),
    (-115.50, 32.70), (-115.00, 32.72),
]

# ── Secondary fault segments ─────────────────────────────────────────

_GARLOCK_FAULT_LONLAT = [
    (-117.75, 34.88), (-117.50, 34.93), (-117.20, 35.00),
    (-117.00, 35.05), (-116.60, 35.13), (-116.20, 35.20),
    (-115.80, 35.30),
]

_HAYWARD_FAULT_LONLAT = [
    (-121.87, 37.30), (-121.92, 37.45), (-122.05, 37.60),
    (-122.10, 37.70), (-122.15, 37.80), (-122.17, 37.90),
]

_SAN_JACINTO_FAULT_LONLAT = [
    (-116.30, 33.00), (-116.50, 33.20), (-116.70, 33.40),
    (-116.80, 33.55), (-116.96, 33.72), (-117.08, 33.88),
    (-117.20, 34.05), (-117.28, 34.15),
]

_ELSINORE_FAULT_LONLAT = [
    (-115.80, 32.65), (-116.05, 32.90), (-116.30, 33.15),
    (-116.70, 33.40), (-117.10, 33.58), (-117.32, 33.68),
    (-117.50, 33.82),
]

_CALAVERAS_FAULT_LONLAT = [
    (-121.40, 36.85), (-121.50, 37.05), (-121.70, 37.30),
    (-121.82, 37.48), (-121.85, 37.60),
]

_RODGERS_CREEK_LONLAT = [
    (-122.50, 38.07), (-122.60, 38.25), (-122.68, 38.45),
    (-122.72, 38.58),
]

# ═══════════════════════════════════════════════════════════════════════
# LABEL DATA — (name, lon, lat, category)
# ═══════════════════════════════════════════════════════════════════════

# ── Major cities ─────────────────────────────────────────────────────

_CITIES = [
    ("San Diego",           -117.16, 32.72, "city"),
    ("Tijuana",             -117.04, 32.53, "city"),
    ("Los Angeles",         -118.24, 34.05, "city"),
    ("San Francisco",       -122.42, 37.78, "city"),
    ("Sacramento",          -121.49, 38.58, "city"),
    ("San Jose",            -121.89, 37.34, "city"),
    ("Oakland",             -122.27, 37.80, "city"),
    ("Fresno",              -119.79, 36.74, "city"),
    ("Bakersfield",         -119.02, 35.37, "city"),
    ("Long Beach",          -118.19, 33.77, "city"),
    ("Santa Barbara",       -119.70, 34.42, "city"),
    ("Riverside",           -117.40, 33.95, "city"),
    ("San Bernardino",      -117.29, 34.11, "city"),
    ("Stockton",            -121.29, 37.96, "city"),
    ("Redding",             -122.39, 40.59, "city"),
    ("Eureka",              -124.16, 40.80, "city"),
]

# ── Smaller cities and towns ─────────────────────────────────────────

_TOWNS = [
    ("Palm Springs",        -116.55, 33.83, "town"),
    ("Indio",               -116.22, 33.72, "town"),
    ("El Centro",           -115.56, 32.79, "town"),
    ("Brawley",             -115.53, 32.98, "town"),
    ("Palmdale",            -118.12, 34.58, "town"),
    ("Lancaster",           -118.14, 34.70, "town"),
    ("Victorville",         -117.29, 34.54, "town"),
    ("Barstow",             -117.02, 34.90, "town"),
    ("Ridgecrest",          -117.67, 35.62, "town"),
    ("Tehachapi",           -118.45, 35.13, "town"),
    ("Parkfield",           -120.43, 35.90, "town"),
    ("Coalinga",            -120.36, 36.14, "town"),
    ("Hollister",           -121.40, 36.85, "town"),
    ("Salinas",             -121.65, 36.68, "town"),
    ("Monterey",            -121.89, 36.60, "town"),
    ("Santa Cruz",          -122.03, 36.97, "town"),
    ("Santa Rosa",          -122.71, 38.44, "town"),
    ("Ukiah",               -123.21, 39.15, "town"),
    ("Paso Robles",         -120.69, 35.63, "town"),
    ("San Luis Obispo",     -120.66, 35.28, "town"),
    ("Santa Maria",         -120.44, 34.95, "town"),
    ("Ventura",             -119.23, 34.28, "town"),
    ("Oxnard",              -119.18, 34.20, "town"),
    ("Santa Clarita",       -118.54, 34.39, "town"),
    ("Temecula",            -117.15, 33.49, "town"),
    ("Oceanside",           -117.38, 33.20, "town"),
    ("Escondido",           -117.09, 33.12, "town"),
    ("Garberville",         -123.79, 40.10, "town"),
    ("Willits",             -123.36, 39.41, "town"),
    ("Fort Bragg",          -123.80, 39.45, "town"),
    ("Bodega Bay",          -123.05, 38.33, "town"),
    ("Petaluma",            -122.64, 38.23, "town"),
    ("Novato",              -122.57, 38.11, "town"),
    ("San Rafael",          -122.53, 37.97, "town"),
    ("Half Moon Bay",       -122.43, 37.46, "town"),
    ("Palo Alto",           -122.14, 37.44, "town"),
    ("Modesto",             -120.99, 37.64, "town"),
    ("Merced",              -120.48, 37.30, "town"),
    ("Madera",              -120.06, 36.96, "town"),
    ("Visalia",             -119.29, 36.33, "town"),
    ("Porterville",         -119.02, 36.07, "town"),
    ("Ontario",             -117.65, 34.06, "town"),
    ("Fontana",             -117.44, 34.09, "town"),
    ("Cajon Pass",          -117.38, 34.32, "town"),
    ("Wrightwood",          -117.63, 34.36, "town"),
]

# ── Road labels (positioned along routes) ────────────────────────────

_ROAD_LABELS = [
    ("I-5",      -118.50, 34.42, "road", -45),
    ("I-5",      -119.55, 35.70, "road", -20),
    ("I-5",      -120.60, 37.05, "road", -15),
    ("I-5",      -121.70, 38.10, "road", 0),
    ("I-15",     -117.18, 33.30, "road", -10),
    ("I-15",     -117.35, 34.45, "road", 0),
    ("I-15",     -116.40, 35.15, "road", 30),
    ("I-10",     -117.50, 33.98, "road", 5),
    ("I-10",     -116.00, 33.78, "road", 10),
    ("I-8",      -116.40, 32.68, "road", 5),
    ("US-101",   -119.45, 34.35, "road", -25),
    ("US-101",   -121.00, 36.20, "road", -35),
    ("US-101",   -122.55, 38.15, "road", -10),
    ("CA-99",    -119.50, 36.00, "road", -10),
    ("CA-99",    -120.20, 37.25, "road", -15),
]

# ── Geographic features ──────────────────────────────────────────────

_GEO_FEATURES = [
    # Mountain passes
    ("Tejon Pass",          -118.88, 34.94, "feature"),
    ("Cajon Pass",          -117.45, 34.30, "feature"),
    ("San Gorgonio Pass",   -116.85, 33.90, "feature"),

    # Valleys
    ("San Fernando Valley", -118.46, 34.22, "feature"),
    ("Antelope Valley",     -118.10, 34.75, "feature"),
    ("Coachella Valley",    -116.30, 33.65, "feature"),
    ("Imperial Valley",     -115.55, 32.85, "feature"),
    ("San Joaquin Valley",  -120.10, 36.50, "feature"),
    ("Salinas Valley",      -121.40, 36.40, "feature"),
    ("Central Valley",      -120.70, 37.60, "feature"),
    ("Death Valley",        -116.87, 36.46, "feature"),

    # Mountain ranges
    ("San Gabriel Mtns",    -117.90, 34.30, "feature"),
    ("San Bernardino Mtns", -117.00, 34.20, "feature"),
    ("Sierra Nevada",       -119.00, 37.50, "feature"),
    ("Coast Ranges",        -121.50, 36.00, "feature"),
    ("Transverse Ranges",   -118.50, 34.50, "feature"),
    ("Tehachapi Mtns",      -118.60, 35.10, "feature"),
    ("Santa Lucia Range",   -121.30, 36.20, "feature"),

    # Water bodies / bays
    ("Salton Sea",          -115.83, 33.30, "feature"),
    ("San Francisco Bay",   -122.15, 37.70, "feature"),
    ("Monterey Bay",        -121.90, 36.80, "feature"),
    ("San Pablo Bay",       -122.38, 38.05, "feature"),
    ("Lake Tahoe",          -120.00, 39.10, "feature"),

    # Canyons and geological
    ("San Andreas Rift",    -118.00, 34.50, "feature"),
    ("Carrizo Plain",       -119.85, 35.15, "feature"),
    ("Pinnacles",           -121.15, 36.49, "feature"),
    ("Point Reyes",         -122.88, 38.07, "feature"),
    ("Tomales Bay",         -122.87, 38.18, "feature"),
    ("Bodega Head",         -123.07, 38.31, "feature"),
    ("Cape Mendocino",      -124.36, 40.44, "feature"),
    ("Point Conception",    -120.47, 34.45, "feature"),
    ("Big Sur",             -121.50, 36.20, "feature"),
    ("Mojave Desert",       -117.00, 35.30, "feature"),
]

# ── Fault name labels ────────────────────────────────────────────────

_FAULT_LABELS = [
    ("Garlock Fault",       -116.90, 35.08, "fault", 10),
    ("Hayward Fault",       -122.08, 37.65, "fault", -70),
    ("San Jacinto Fault",   -116.85, 33.60, "fault", -55),
    ("Elsinore Fault",      -117.20, 33.45, "fault", -55),
    ("Calaveras Fault",     -121.60, 37.20, "fault", -60),
    ("Rodgers Creek Fault", -122.60, 38.35, "fault", -75),
    ("San Andreas Fault",   -120.40, 35.50, "fault", -50),
]


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

def get_vector_layers() -> Dict[str, List[List[Tuple[float, float]]]]:
    """Get all vector polyline layers as projected coordinate paths.

    Returns
    -------
    dict[str, list[list[(x_m, y_m)]]]
        Layer name -> list of polylines in EPSG:3310.
    """
    layers: Dict[str, List[List[Tuple[float, float]]]] = {}

    layers["coastline"] = [_project(_COASTLINE_LONLAT)]
    layers["highways"] = [
        _project(_I5_LONLAT),
        _project(_I15_LONLAT),
        _project(_I10_LONLAT),
        _project(_US101_LONLAT),
        _project(_CA99_LONLAT),
        _project(_I8_LONLAT),
    ]
    layers["borders"] = [
        _project(_CA_NV_BORDER_LONLAT),
        _project(_CA_OR_BORDER_LONLAT),
        _project(_CA_MX_BORDER_LONLAT),
    ]
    layers["faults"] = [
        _project(_GARLOCK_FAULT_LONLAT),
        _project(_HAYWARD_FAULT_LONLAT),
        _project(_SAN_JACINTO_FAULT_LONLAT),
        _project(_ELSINORE_FAULT_LONLAT),
        _project(_CALAVERAS_FAULT_LONLAT),
        _project(_RODGERS_CREEK_LONLAT),
    ]

    log.info(
        "Vector layers: %s",
        ", ".join(f"{k}({len(v)})" for k, v in layers.items()),
    )
    return layers


def get_labels() -> List[MapLabel]:
    """Get all map text labels as projected MapLabel objects.

    Returns
    -------
    list[MapLabel]
        Labels with positions in EPSG:3310.
    """
    labels: List[MapLabel] = []

    # Cities (larger text)
    for name, lon, lat, cat in _CITIES:
        x, y = _project_pt(lon, lat)
        labels.append(MapLabel(name, x, y, cat, font_size=5.0))

    # Towns (smaller text)
    for name, lon, lat, cat in _TOWNS:
        x, y = _project_pt(lon, lat)
        labels.append(MapLabel(name, x, y, cat, font_size=3.5))

    # Road labels (with rotation)
    for name, lon, lat, cat, rot in _ROAD_LABELS:
        x, y = _project_pt(lon, lat)
        labels.append(MapLabel(name, x, y, cat, font_size=3.0, rotation=rot))

    # Geographic features
    for name, lon, lat, cat in _GEO_FEATURES:
        x, y = _project_pt(lon, lat)
        labels.append(MapLabel(name, x, y, cat, font_size=3.0))

    # Fault labels (with rotation)
    for name, lon, lat, cat, rot in _FAULT_LABELS:
        x, y = _project_pt(lon, lat)
        labels.append(MapLabel(name, x, y, cat, font_size=3.5, rotation=rot))

    log.info(
        "Labels: %d cities, %d towns, %d roads, %d features, %d faults",
        len(_CITIES), len(_TOWNS), len(_ROAD_LABELS),
        len(_GEO_FEATURES), len(_FAULT_LABELS),
    )
    return labels
