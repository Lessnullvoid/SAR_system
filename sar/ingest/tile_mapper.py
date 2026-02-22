"""
Spatial mapper — assigns sensor events to grid tiles.

Given a list of events with (lat, lon) and a tile grid, determines
which tile each event falls into.  Uses projected coordinates for
accuracy.

Usage
-----
    from sar.ingest.tile_mapper import TileMapper
    mapper = TileMapper(tiles)
    tile_id = mapper.find_tile(lat=34.05, lon=-117.25)
    quake_map = mapper.map_earthquakes(quakes)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pyproj

from ..geo.tile_grid import Tile

log = logging.getLogger(__name__)

WGS84 = pyproj.CRS("EPSG:4326")
AEA_CA = pyproj.CRS("EPSG:3310")
_to_metric = pyproj.Transformer.from_crs(WGS84, AEA_CA, always_xy=True)


class TileMapper:
    """Assigns geographic points to grid tiles using projected coordinates."""

    def __init__(self, tiles: List[Tile]):
        self._tiles = tiles
        # Build a spatial index: dict of (row, col) → tile
        self._grid: Dict[Tuple[int, int], Tile] = {}
        for t in tiles:
            self._grid[(t.row, t.col)] = t

        # Compute grid parameters from tiles
        if tiles:
            all_bounds = [t.bounds_m for t in tiles]
            self._tile_size = all_bounds[0][2] - all_bounds[0][0]  # tile width in m
            xs = [b[0] for b in all_bounds]
            ys = [b[1] for b in all_bounds]
            self._grid_minx = min(xs)
            self._grid_miny = min(ys)
            self._grid_maxx = max(b[2] for b in all_bounds)
            self._grid_maxy = max(b[3] for b in all_bounds)
            # Row 0 is at the top (max y), compute accordingly
            self._n_rows = max(t.row for t in tiles) + 1
            self._n_cols = max(t.col for t in tiles) + 1

    def find_tile(self, lat: float, lon: float) -> Optional[str]:
        """Find the tile_id containing a lat/lon point.

        Returns None if the point is outside the grid.
        """
        try:
            mx, my = _to_metric.transform(lon, lat)
        except Exception:
            return None

        # Compute grid row/col
        col = int((mx - self._grid_minx) / self._tile_size)
        # Row is inverted: row 0 = top (max y)
        row = int((self._grid_maxy - my) / self._tile_size)

        tile = self._grid.get((row, col))
        return tile.tile_id if tile else None

    def find_tile_obj(self, lat: float, lon: float) -> Optional[Tile]:
        """Find the Tile object containing a lat/lon point."""
        try:
            mx, my = _to_metric.transform(lon, lat)
        except Exception:
            return None

        col = int((mx - self._grid_minx) / self._tile_size)
        row = int((self._grid_maxy - my) / self._tile_size)

        return self._grid.get((row, col))

    def map_earthquakes(self, quakes) -> Dict[str, list]:
        """Assign earthquakes to tiles.

        Parameters
        ----------
        quakes : list[Earthquake]
            Earthquake objects with .lat and .lon attributes.

        Returns
        -------
        dict[str, list[Earthquake]]
            Mapping of tile_id → list of earthquakes in that tile.
        """
        result: Dict[str, list] = {}
        mapped = 0
        for q in quakes:
            tid = self.find_tile(q.lat, q.lon)
            if tid:
                result.setdefault(tid, []).append(q)
                mapped += 1

        log.info(
            "Mapped %d/%d earthquakes to %d tiles",
            mapped, len(quakes), len(result),
        )
        return result
