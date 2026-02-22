"""
Tile grid system for the San Andreas Fault corridor.

The fault corridor is subdivided into a regular grid of square tiles.
Tiles are indexed by (row, col) where row 0 is the northernmost row.
Each tile has a unique ID like ``SAF_023``.

The grid is built in a projected coordinate system (EPSG:3310 — California
Albers Equal Area) so distances are in metres, then tile corners are
converted back to lon/lat for display.

Usage
-----
    grid = build_tile_grid(tile_km=15.0, buffer_km=25.0)
    print(f"{len(grid)} tiles covering the San Andreas corridor")
    for tile in grid:
        print(tile.tile_id, tile.centroid_lonlat)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pyproj
from shapely.geometry import Polygon, box
from shapely.ops import transform

from .fault_trace import fault_corridor, fault_trace_line

# Coordinate reference systems
WGS84 = pyproj.CRS("EPSG:4326")
AEA_CA = pyproj.CRS("EPSG:3310")  # California Albers Equal Area (metres)

_to_metric = pyproj.Transformer.from_crs(WGS84, AEA_CA, always_xy=True).transform
_to_lonlat = pyproj.Transformer.from_crs(AEA_CA, WGS84, always_xy=True).transform


@dataclass
class Tile:
    """One geographic tile in the fault corridor grid."""

    tile_id: str                          # e.g. "SAF_023"
    row: int                              # grid row (0 = northernmost)
    col: int                              # grid column (0 = westernmost)
    index: int                            # sequential N→S index

    # Geometry in projected metres (EPSG:3310)
    bounds_m: Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)

    # Geometry in lon/lat (WGS84)
    centroid_lonlat: Tuple[float, float] = (0.0, 0.0)
    corners_lonlat: List[Tuple[float, float]] = field(default_factory=list)

    # Bounding box in lon/lat
    bbox_lonlat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    # Distance along fault from SE terminus (km)
    fault_distance_km: float = 0.0

    # Section label
    section: str = ""   # "southern", "central", "northern"

    # Whether this tile intersects the fault corridor
    on_fault: bool = False


def build_tile_grid(
    tile_km: float = 15.0,
    buffer_km: float = 25.0,
    full_bbox: bool = False,
) -> List[Tile]:
    """Build the tile grid covering the fault corridor.

    Parameters
    ----------
    tile_km : float
        Tile edge length in kilometres (default 15 km).
    buffer_km : float
        Corridor half-width in kilometres (default 25 km).
    full_bbox : bool
        If True, generate tiles for the ENTIRE bounding box (not just
        tiles intersecting the corridor).  Tiles that overlap the fault
        corridor have ``on_fault=True``; others have ``on_fault=False``.
        This fills the map with satellite imagery everywhere.

    Returns
    -------
    list[Tile]
        Tiles sorted north → south (row 0 = northernmost).
    """
    tile_m = tile_km * 1000.0

    # Get corridor polygon in projected coordinates
    corridor_ll = fault_corridor(buffer_km=buffer_km)
    corridor_m = transform(_to_metric, corridor_ll)

    # Get fault trace in projected coordinates (for distance-along-fault)
    trace_ll = fault_trace_line()
    trace_m = transform(_to_metric, trace_ll)

    # Grid bounding box
    minx, miny, maxx, maxy = corridor_m.bounds

    # Align grid to tile_m increments
    grid_minx = math.floor(minx / tile_m) * tile_m
    grid_miny = math.floor(miny / tile_m) * tile_m
    grid_maxx = math.ceil(maxx / tile_m) * tile_m
    grid_maxy = math.ceil(maxy / tile_m) * tile_m

    n_cols = int((grid_maxx - grid_minx) / tile_m)
    n_rows = int((grid_maxy - grid_miny) / tile_m)

    tiles: List[Tile] = []
    seq = 0

    from shapely.geometry import Point

    total_km = trace_m.length / 1000.0

    # Iterate rows top-to-bottom (north → south in projected coords means
    # high Y → low Y since AEA_CA Y increases northward)
    for r in range(n_rows):
        for c in range(n_cols):
            x0 = grid_minx + c * tile_m
            y0 = grid_maxy - (r + 1) * tile_m  # top-down
            x1 = x0 + tile_m
            y1 = y0 + tile_m

            tile_box = box(x0, y0, x1, y1)

            intersects_corridor = corridor_m.intersects(tile_box)

            # Skip non-corridor tiles unless full_bbox requested
            if not full_bbox and not intersects_corridor:
                continue

            # Centroid in lon/lat
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            clon, clat = _to_lonlat(cx, cy)

            # Corners in lon/lat (for rendering)
            corners_m = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            corners_ll = [_to_lonlat(px, py) for px, py in corners_m]

            # Bounding box in lon/lat
            lons = [p[0] for p in corners_ll]
            lats = [p[1] for p in corners_ll]
            bbox_ll = (min(lons), min(lats), max(lons), max(lats))

            # Distance along fault (project centroid onto trace)
            pt = Point(cx, cy)
            dist_m = trace_m.project(pt)
            dist_km = dist_m / 1000.0

            # Section classification
            frac = dist_km / total_km if total_km > 0 else 0
            if frac < 0.33:
                section = "southern"
            elif frac < 0.67:
                section = "central"
            else:
                section = "northern"

            tile = Tile(
                tile_id=f"SAF_{seq:03d}",
                row=r,
                col=c,
                index=seq,
                bounds_m=(x0, y0, x1, y1),
                centroid_lonlat=(clon, clat),
                corners_lonlat=corners_ll,
                bbox_lonlat=bbox_ll,
                fault_distance_km=dist_km,
                section=section,
                on_fault=intersects_corridor,
            )
            tiles.append(tile)
            seq += 1

    # Sort: fault tiles first by distance, then non-fault tiles
    tiles.sort(key=lambda t: (not t.on_fault, t.fault_distance_km))

    # Re-index after sorting
    for i, t in enumerate(tiles):
        t.index = i
        t.tile_id = f"SAF_{i:03d}"

    return tiles


def tiles_to_geojson(tiles: List[Tile]) -> dict:
    """Export tiles as a GeoJSON FeatureCollection for debugging."""
    features = []
    for t in tiles:
        ring = t.corners_lonlat + [t.corners_lonlat[0]]  # close ring
        features.append({
            "type": "Feature",
            "properties": {
                "tile_id": t.tile_id,
                "row": t.row,
                "col": t.col,
                "index": t.index,
                "section": t.section,
                "fault_distance_km": round(t.fault_distance_km, 1),
                "centroid_lon": round(t.centroid_lonlat[0], 5),
                "centroid_lat": round(t.centroid_lonlat[1], 5),
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[(round(lon, 6), round(lat, 6)) for lon, lat in ring]],
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
    }
