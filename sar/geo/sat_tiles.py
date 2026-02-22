"""
Satellite imagery downloader and cache for the fault tile grid.

Downloads satellite imagery from ESRI World Imagery (free for non-commercial
use) for each tile in the grid, converts to grayscale, and caches locally.

The download only needs to happen once — after that, tiles are loaded from
the local cache.  On a Raspberry Pi without internet, the system works fine
with whatever has been cached.

Persistent data (all stored under ``data/sat_cache/``):
  - ``SAF_*.png``         — tile imagery (grayscale, darkened)
  - ``nodata_tiles.txt``  — tile IDs with no satellite coverage (ocean, etc.)
  - ``download_meta.json``— download statistics and timestamp

Usage
-----
    from sar.geo.sat_tiles import download_all_tiles, get_tile_image_path
    from sar.geo.tile_grid import build_tile_grid

    tiles = build_tile_grid()
    download_all_tiles(tiles)          # one-time download (needs internet)

    path = get_tile_image_path("SAF_023")  # returns Path or None
"""
from __future__ import annotations

import json
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .tile_grid import Tile

log = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sat_cache"

# Cache version — bump when download parameters change so stale images
# are automatically cleared.  v1 = WGS84, v2 = EPSG:3310 projected.
_CACHE_VERSION = 2
_CACHE_VERSION_FILE = _CACHE_DIR / ".cache_version"

# ESRI tile services — free for non-commercial / educational use
_ESRI_IMAGERY_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/export"
)
_IS_PI = (
    platform.system() == "Linux"
    and platform.machine().startswith(("aarch64", "arm"))
)
_TILE_PX = 128 if _IS_PI else 512

# Persistent no-data file
_NODATA_FILE = _CACHE_DIR / "nodata_tiles.txt"
_META_FILE = _CACHE_DIR / "download_meta.json"


def get_cache_dir() -> Path:
    """Return the satellite cache directory, creating it if needed."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _check_cache_version() -> None:
    """Clear cache if it was built with an older version (different CRS)."""
    get_cache_dir()
    try:
        if _CACHE_VERSION_FILE.exists():
            v = int(_CACHE_VERSION_FILE.read_text().strip())
            if v == _CACHE_VERSION:
                return
        # Version mismatch or missing → clear all cached images
        old_images = list(_CACHE_DIR.glob("SAF_*.png"))
        if old_images:
            log.info(
                "Cache version changed (v%d) — clearing %d stale images",
                _CACHE_VERSION, len(old_images),
            )
            for f in old_images:
                f.unlink()
        # Also clear nodata list (ocean heuristic is the same, but
        # some tiles might have been marked nodata due to CRS issues)
        if _NODATA_FILE.exists():
            _NODATA_FILE.unlink()
        _CACHE_VERSION_FILE.write_text(str(_CACHE_VERSION))
    except Exception as exc:
        log.warning("Cache version check failed: %s", exc)


def get_tile_image_path(tile_id: str) -> Optional[Path]:
    """Return the cached satellite image path for a tile, or None."""
    path = _CACHE_DIR / f"{tile_id}.png"
    return path if path.exists() else None


# ── No-data tile persistence ─────────────────────────────────────────

def _load_nodata_tiles() -> set:
    """Load the set of tile IDs with no satellite data from disk."""
    if _NODATA_FILE.exists():
        try:
            text = _NODATA_FILE.read_text().strip()
            tiles = {line.strip() for line in text.splitlines() if line.strip()}
            log.info("Loaded %d no-data tile IDs from cache", len(tiles))
            return tiles
        except Exception as exc:
            log.warning("Failed to load nodata_tiles.txt: %s", exc)
    return set()


def _save_nodata_tiles(nodata: set) -> None:
    """Persist the no-data tile set to disk."""
    try:
        get_cache_dir()
        _NODATA_FILE.write_text("\n".join(sorted(nodata)) + "\n")
        log.debug("Saved %d no-data tile IDs to cache", len(nodata))
    except Exception as exc:
        log.warning("Failed to save nodata_tiles.txt: %s", exc)


# Check cache version on import — clears stale images from old CRS
_check_cache_version()

# Load on module import so the set is available immediately
_NODATA_TILES: set = _load_nodata_tiles()


def _save_download_meta(total: int, cached: int, nodata: int, failed: int) -> None:
    """Save download metadata for diagnostics."""
    try:
        meta = {
            "last_download": datetime.now(timezone.utc).isoformat(),
            "total_tiles": total,
            "cached_tiles": cached,
            "nodata_tiles": nodata,
            "failed_tiles": failed,
            "cache_dir": str(_CACHE_DIR),
        }
        _META_FILE.write_text(json.dumps(meta, indent=2) + "\n")
    except Exception as exc:
        log.debug("Failed to save download_meta.json: %s", exc)


# ── Ocean tile heuristic ─────────────────────────────────────────────

def _is_ocean_tile(tile: Tile) -> bool:
    """Heuristic check if a tile is likely entirely over ocean.

    Tiles west of the coastline (roughly) won't have useful satellite imagery.
    We skip these to avoid hammering the server with retries.
    """
    lon_c, lat_c = tile.centroid_lonlat

    # Rough coastline approximation: lon threshold by latitude
    if lat_c < 33.5:
        coast_lon = -117.6
    elif lat_c < 34.5:
        coast_lon = -119.0
    elif lat_c < 35.5:
        coast_lon = -121.0
    elif lat_c < 37.0:
        coast_lon = -122.2
    elif lat_c < 38.5:
        coast_lon = -123.1
    elif lat_c < 40.0:
        coast_lon = -123.8
    else:
        coast_lon = -124.5

    return lon_c < coast_lon - 0.3


# ── Tile download ────────────────────────────────────────────────────

def _download_tile_image(tile: Tile, retries: int = 1) -> Optional[Path]:
    """Download a satellite image for one tile.

    Uses the ESRI World Imagery export API to get a PNG covering the
    tile's bounding box.  Skips tiles that are over open ocean or have
    previously returned no-data responses.

    Returns the cached file path, or None on failure.
    """
    import requests
    from PIL import Image
    from io import BytesIO

    cache_dir = get_cache_dir()
    out_path = cache_dir / f"{tile.tile_id}.png"

    if out_path.exists():
        return out_path

    # Skip known no-data tiles (persisted from previous runs)
    if tile.tile_id in _NODATA_TILES:
        return None

    # Skip ocean tiles by heuristic
    if _is_ocean_tile(tile):
        log.debug("Skipping ocean tile %s", tile.tile_id)
        _NODATA_TILES.add(tile.tile_id)
        return None

    # Bounding box in projected metres (EPSG:3310 — same CRS as tile grid)
    # Using projected coordinates ensures the downloaded image aligns
    # perfectly with the tile grid — no warping or seams between tiles.
    x0, y0, x1, y1 = tile.bounds_m

    params = {
        "bbox": f"{x0},{y0},{x1},{y1}",
        "bboxSR": "3310",
        "imageSR": "3310",
        "size": f"{_TILE_PX},{_TILE_PX}",
        "format": "png",
        "f": "image",
    }

    for attempt in range(retries + 1):
        try:
            resp = requests.get(_ESRI_IMAGERY_URL, params=params, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 500:
                # Convert to high-contrast grayscale
                img = Image.open(BytesIO(resp.content)).convert("L")

                import numpy as np
                arr = np.array(img, dtype=np.float32)

                # Histogram stretch: map [p2, p98] → [0, 255]
                # This maximises contrast — deep blacks and bright whites
                p2 = np.percentile(arr, 2)
                p98 = np.percentile(arr, 98)
                if p98 - p2 > 10:
                    arr = (arr - p2) / (p98 - p2) * 255.0
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                img = Image.fromarray(arr, mode="L")

                img.save(out_path, "PNG")
                return out_path
            elif resp.status_code == 200 and len(resp.content) <= 500:
                # Server returned a tiny response = no imagery for this area
                # Mark as no-data permanently
                log.debug(
                    "Tile %s: no satellite data (ocean/blank), marking persistent",
                    tile.tile_id,
                )
                _NODATA_TILES.add(tile.tile_id)
                return None
            else:
                log.warning(
                    "Tile %s download failed: HTTP %d (%d bytes)",
                    tile.tile_id, resp.status_code, len(resp.content),
                )
        except Exception as exc:
            log.warning("Tile %s download error (attempt %d): %s",
                        tile.tile_id, attempt + 1, exc)
            if attempt < retries:
                time.sleep(1.0)

    return None


def download_all_tiles(
    tiles: List[Tile],
    progress_callback=None,
    max_workers: int = 8,
) -> int:
    """Download satellite imagery for all tiles using parallel workers.

    Parameters
    ----------
    tiles : list[Tile]
        The tile grid.
    progress_callback : callable, optional
        Called with (completed, total) after each batch.
    max_workers : int
        Number of parallel download threads (default 8).

    Returns
    -------
    int
        Number of tiles successfully downloaded (including already cached).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    total = len(tiles)
    success = 0
    skipped = 0

    # Pre-filter: separate cached, nodata, and to-download tiles
    to_download = []
    for tile in tiles:
        if get_tile_image_path(tile.tile_id):
            success += 1
            skipped += 1
        elif tile.tile_id in _NODATA_TILES or _is_ocean_tile(tile):
            _NODATA_TILES.add(tile.tile_id)
        else:
            to_download.append(tile)

    need = len(to_download)
    log.info(
        "Satellite download: %d tiles total, %d cached, %d no-data, %d to download "
        "(using %d workers)",
        total, skipped, len(_NODATA_TILES), need, max_workers,
    )

    if need == 0:
        _save_nodata_tiles(_NODATA_TILES)
        _save_download_meta(total, success, len(_NODATA_TILES), 0)
        return success

    # Parallel download
    lock = threading.Lock()
    completed = [0]

    def _download_one(tile: Tile) -> bool:
        path = _download_tile_image(tile)
        with lock:
            completed[0] += 1
            if completed[0] % 50 == 0:
                log.info(
                    "  satellite progress: %d/%d (%.0f%%)",
                    completed[0], need, completed[0] / need * 100,
                )
        return path is not None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_one, tile): tile
            for tile in to_download
        }
        for future in as_completed(futures):
            try:
                if future.result():
                    success += 1
            except Exception as exc:
                log.debug("Download future error: %s", exc)

    # Persist no-data list and metadata for future runs
    _save_nodata_tiles(_NODATA_TILES)
    _save_download_meta(
        total=total,
        cached=success,
        nodata=len(_NODATA_TILES),
        failed=total - success - len(_NODATA_TILES),
    )

    log.info(
        "Satellite download complete: %d/%d cached, %d no-data, %d skipped",
        success, total, len(_NODATA_TILES), skipped,
    )
    return success


def clear_cache() -> int:
    """Delete all cached satellite images so they can be re-downloaded.

    Returns the number of files deleted.
    """
    cache_dir = get_cache_dir()
    files = list(cache_dir.glob("SAF_*.png"))
    for f in files:
        f.unlink()
    # Also clear the nodata list (tile IDs may have changed)
    _NODATA_TILES.clear()
    if _NODATA_FILE.exists():
        _NODATA_FILE.unlink()
    log.info("Cleared satellite cache: %d files deleted", len(files))
    return len(files)


def cache_stats() -> dict:
    """Return cache statistics."""
    cache_dir = get_cache_dir()
    files = list(cache_dir.glob("SAF_*.png"))
    total_bytes = sum(f.stat().st_size for f in files)

    meta = {}
    if _META_FILE.exists():
        try:
            meta = json.loads(_META_FILE.read_text())
        except Exception:
            pass

    return {
        "cached_tiles": len(files),
        "nodata_tiles": len(_NODATA_TILES),
        "total_mb": round(total_bytes / (1024 * 1024), 1),
        "cache_dir": str(cache_dir),
        "last_download": meta.get("last_download", "never"),
    }

