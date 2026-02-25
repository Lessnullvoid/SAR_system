"""
Fault corridor map widget — QGraphicsScene-based tile visualization.

Renders the San Andreas tile grid as a dark, zoomable map with:
  - Grid of tiles: seismically active tiles render as NEGATIVE/inverted imagery
  - Vector overlay (roads, coastline, faults, borders)
  - Fault trace line with glow
  - SITE crosshair at the instrument location
  - Concentric range circles (antenna coverage)
  - Earthquake markers
  - N/S/E/W compass labels + scale bar
  - Autonomous map scanner that zooms into active areas

Visual style: high-activity tiles show inverted (bright/negative) satellite
imagery like an X-ray view, while quiet areas remain dark.  This matches
the reference aesthetic of a seismic monitoring atlas.

Coordinate system: EPSG:3310 (California Albers Equal Area) projected metres.
"""
from __future__ import annotations

import logging
import math
import os
import platform
import random
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from pathlib import Path

if TYPE_CHECKING:
    from .narrative_engine import Chapter, NarrativeEngine, Shot

from PyQt5 import QtCore, QtGui, QtWidgets

from ..geo.tile_grid import Tile
from ..geo.sat_tiles import get_tile_image_path, cache_stats

log = logging.getLogger(__name__)

# ── Platform detection ─────────────────────────────────────────────────
_IS_PI = (
    platform.system() == "Linux"
    and platform.machine().startswith(("aarch64", "arm"))
)
_IS_KIOSK = _IS_PI or bool(os.environ.get("SAR_KIOSK"))


# ── Colour helpers ────────────────────────────────────────────────────

def _score_to_color(score: float) -> QtGui.QColor:
    """Map 0-1 score to cold-palette colour for overlay."""
    s = max(0.0, min(1.0, score))
    if s < 0.3:
        t = s / 0.3
        r, g, b = 4, 8 + int(12 * t), 16 + int(30 * t)
    elif s < 0.5:
        t = (s - 0.3) / 0.2
        r, g, b = 4, 20 + int(40 * t), 46 + int(80 * t)
    elif s < 0.7:
        t = (s - 0.5) / 0.2
        r, g, b = 4, 60 + int(80 * t), 126 + int(80 * t)
    else:
        t = (s - 0.7) / 0.3
        r, g, b = int(60 * t), 140 + int(115 * t), 206 + int(49 * t)
    return QtGui.QColor(r, g, b)


def _score_to_border(score: float) -> QtGui.QColor:
    s = max(0.0, min(1.0, score))
    base = 15 + int(60 * s)
    return QtGui.QColor(base, base + int(20 * s), base + int(40 * s))






# ── Tile graphics item ────────────────────────────────────────────────

class TileItem(QtWidgets.QGraphicsObject):
    """One tile — renders as negative/inverted imagery when seismically active.

    Rendering layers:
      1. Satellite image — progressively inverted based on score
      2. Score colour tint (subtle overlay)
      3. Grid border
    """

    def __init__(
        self,
        tile: Tile,
        scale_factor: float,
        origin_x: float,
        origin_y: float,
        parent_widget: "FaultMapWidget",
    ):
        super().__init__()
        x0, y0, x1, y1 = tile.bounds_m
        # Position in scene coordinates
        sx = (x0 - origin_x) * scale_factor
        sy = -(y1 - origin_y) * scale_factor
        sw = (x1 - x0) * scale_factor
        sh = (y1 - y0) * scale_factor
        # Set the item position so the bounding rect is in local (0,0)
        self.setPos(sx, sy)
        self._local_rect = QtCore.QRectF(0, 0, sw, sh)
        # Scene-space rect for external queries (scanner, fault rect, etc.)
        self._rect = QtCore.QRectF(sx, sy, sw, sh)

        self.tile = tile
        self._parent_widget = parent_widget
        self._score = 0.0
        self._hovered = False
        self._scanner_focus = False  # True when the auto-scanner targets this tile
        self._quake_count = 0        # earthquakes mapped to this tile
        self._quake_max_mag = 0.0    # largest magnitude in this tile

        # Tile imagery — pixmap loading is deferred (staggered) to avoid
        # blocking the GUI event loop during startup.
        self._pixmap_sat: Optional[QtGui.QPixmap] = None
        self._pixmap_news: Optional[QtGui.QPixmap] = None  # current display pixmap
        self._news_pixmaps: list = []    # ALL loaded pixmaps (one per article with image)
        self._news_pixmap_idx: int = 0   # index into _news_pixmaps for cycling
        self._news_articles: list = []   # ICEArticle list for this tile
        self._news_image_paths: list = []  # paths for lazy loading
        self._news_images_loaded = False   # True after all news pixmaps loaded
        self._sat_loaded = False         # True once we attempted to load sat image

        # History overlay data (social-history events)
        self._history_events: list = []
        self._history_image_paths: list = []
        self._history_pixmaps: list = []
        self._history_pixmap_idx: int = 0
        self._pixmap_history: Optional[QtGui.QPixmap] = None
        self._history_images_loaded = False

        self.setAcceptHoverEvents(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setToolTip(
            f"{tile.tile_id}\n"
            f"Section: {tile.section}\n"
            f"Distance: {tile.fault_distance_km:.0f} km along fault\n"
            f"Lat: {tile.centroid_lonlat[1]:.3f}  Lon: {tile.centroid_lonlat[0]:.3f}"
        )
        # No caching — tiles must repaint cleanly at every zoom level
        self.setCacheMode(QtWidgets.QGraphicsItem.NoCache)

    def boundingRect(self) -> QtCore.QRectF:
        return self._local_rect

    def paint(self, painter: QtGui.QPainter, option, widget=None) -> None:
        r = self._local_rect
        selected = self._scanner_focus or self._hovered

        # ── Layer 1: Tile image (satellite only — news/history shown via overlay) ──
        if self._pixmap_sat:
            src = QtCore.QRectF(self._pixmap_sat.rect())
            painter.drawPixmap(r, self._pixmap_sat, src)
            if not self.tile.on_fault:
                painter.fillRect(r, QtGui.QColor(0, 0, 0, 100))
        else:
            painter.fillRect(r, QtGui.QColor(2, 3, 6))

        # ── Layer 1b: Invert the tile when scanner is focused ──
        if self._scanner_focus and self._pixmap_sat:
            painter.setCompositionMode(
                QtGui.QPainter.CompositionMode_Difference
            )
            painter.fillRect(r, QtGui.QColor(255, 255, 255))
            painter.setCompositionMode(
                QtGui.QPainter.CompositionMode_SourceOver
            )

        # ── Layer 2: Grid — always visible, even zoomed out ──
        if selected:
            pen = QtGui.QPen(QtGui.QColor(0, 220, 255, 240))
            pen.setWidthF(2.0)
        else:
            pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 120))
            pen.setWidthF(3.0)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRect(r)

        # ── Layer 3: News indicator — cyan dot at any zoom ──
        if self._news_articles:
            self._draw_news_indicator(painter, r)

        # ── Layer 4: History indicator — amber diamond at any zoom ──
        if self._history_events:
            self._draw_history_indicator(painter, r)

    def _draw_news_indicator(self, painter: QtGui.QPainter, r: QtCore.QRectF) -> None:
        """Draw a visible NEWS marker in the corner of tiles with news."""
        painter.save()
        # Bright cyan filled square in top-right corner
        sz = min(r.width(), r.height()) * 0.2
        marker_r = QtCore.QRectF(
            r.right() - sz - r.width() * 0.05,
            r.top() + r.height() * 0.05,
            sz, sz,
        )
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(0, 220, 255, 180))
        painter.drawRect(marker_r)
        # "N" letter
        painter.setPen(QtGui.QColor(0, 0, 0))
        font = painter.font()
        font.setPointSizeF(max(0.5, sz * 0.5))
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(marker_r, QtCore.Qt.AlignCenter, "N")
        painter.restore()

    def _draw_history_indicator(self, painter: QtGui.QPainter, r: QtCore.QRectF) -> None:
        """Draw an amber diamond marker in the bottom-right corner of tiles with history."""
        painter.save()
        sz = min(r.width(), r.height()) * 0.2
        cx = r.right() - sz * 0.6 - r.width() * 0.05
        cy = r.bottom() - sz * 0.6 - r.height() * 0.05
        half = sz * 0.5
        diamond = QtGui.QPolygonF([
            QtCore.QPointF(cx, cy - half),
            QtCore.QPointF(cx + half, cy),
            QtCore.QPointF(cx, cy + half),
            QtCore.QPointF(cx - half, cy),
        ])
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(220, 170, 30, 200))
        painter.drawPolygon(diamond)
        painter.setPen(QtGui.QColor(0, 0, 0))
        font = painter.font()
        font.setPointSizeF(max(0.5, sz * 0.4))
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            QtCore.QRectF(cx - half, cy - half, sz, sz),
            QtCore.Qt.AlignCenter, "H",
        )
        painter.restore()

    def set_history_data(self, events: list) -> None:
        """Assign social-history events to this tile."""
        self._history_events = list(events)
        self._history_image_paths = [
            e.image_path for e in events if e.image_path
        ]
        self._history_pixmaps = []
        self._history_pixmap_idx = 0
        self._history_images_loaded = False

        if self._history_image_paths:
            pm = QtGui.QPixmap(self._history_image_paths[0])
            if not pm.isNull():
                _pi_max = 256 if _IS_PI else 512
                if pm.width() > _pi_max or pm.height() > _pi_max:
                    pm = pm.scaled(_pi_max, _pi_max, QtCore.Qt.KeepAspectRatio,
                                   QtCore.Qt.SmoothTransformation)
                self._history_pixmaps.append(pm)
            self._pixmap_history = self._history_pixmaps[0] if self._history_pixmaps else None
        else:
            self._pixmap_history = None
        self.update()

    _LAZY_HIST_BATCH = 3

    def _ensure_all_history_pixmaps(self) -> None:
        """Lazily load remaining history pixmaps in small batches.

        Same batched approach as news images to avoid holding the GIL
        long enough to starve the SDR audio thread.
        """
        if self._history_images_loaded:
            return
        self._history_images_loaded = True
        if len(self._history_image_paths) <= 1:
            return
        self._lazy_hist_idx = 1
        self._lazy_hist_batch()

    def _lazy_hist_batch(self) -> None:
        try:
            end = min(
                self._lazy_hist_idx + self._LAZY_HIST_BATCH,
                len(self._history_image_paths),
            )
            for i in range(self._lazy_hist_idx, end):
                path = self._history_image_paths[i]
                pm = QtGui.QPixmap(path)
                if not pm.isNull():
                    _pi_max = 256 if _IS_PI else 512
                    if pm.width() > _pi_max or pm.height() > _pi_max:
                        pm = pm.scaled(
                            _pi_max, _pi_max, QtCore.Qt.KeepAspectRatio,
                            QtCore.Qt.SmoothTransformation,
                        )
                    self._history_pixmaps.append(pm)
            self._lazy_hist_idx = end
            if end < len(self._history_image_paths):
                QtCore.QTimer.singleShot(5, self._lazy_hist_batch)
            else:
                log.debug(
                    "Lazy-loaded %d/%d history images for %s",
                    len(self._history_pixmaps),
                    len(self._history_image_paths),
                    self.tile.tile_id,
                )
        except Exception as exc:
            log.warning("History image lazy-load error for %s: %s",
                        self.tile.tile_id, exc)

    def advance_history_image(self) -> bool:
        """Advance to the next history image. Returns True if there are more."""
        self._ensure_all_history_pixmaps()
        if len(self._history_pixmaps) <= 1:
            return False
        self._history_pixmap_idx = (self._history_pixmap_idx + 1) % len(self._history_pixmaps)
        self._pixmap_history = self._history_pixmaps[self._history_pixmap_idx]
        self.update()
        return True

    @property
    def history_image_count(self) -> int:
        return len(self._history_pixmaps)

    def set_quake_data(self, count: int, max_mag: float) -> None:
        """Update earthquake count and max magnitude for this tile."""
        if self._quake_count != count or self._quake_max_mag != max_mag:
            self._quake_count = count
            self._quake_max_mag = max_mag
            self.update()

    def set_score(self, score: float) -> None:
        self._score = score
        self.update()

    def set_scanner_focus(self, focused: bool) -> None:
        """Set whether the map scanner is currently focused on this tile."""
        if self._scanner_focus != focused:
            self._scanner_focus = focused
            self.update()

    def load_sat_image(self, path: Path) -> None:
        """Load satellite imagery from disk into the tile's pixmap.

        On Raspberry Pi, scales the pixmap down to 256x256 to balance
        quality and memory (~120 MB for all tiles on Pi 5 with 8 GB).
        """
        self._load_sat_image_no_update(path)
        self.update()

    def _load_sat_image_no_update(self, path: Path) -> None:
        """Load satellite imagery without triggering a repaint.

        Used during bulk initial loading so we can do a single
        scene.update() at the end instead of 7000 individual repaints.
        """
        pm = QtGui.QPixmap(str(path))
        if not pm.isNull():
            if _IS_PI and (pm.width() > 256 or pm.height() > 256):
                pm = pm.scaled(256, 256, QtCore.Qt.KeepAspectRatio,
                               QtCore.Qt.SmoothTransformation)
            self._pixmap_sat = pm
            self._sat_loaded = True

    def set_news_data(self, articles: list) -> None:
        """Set news articles for this tile, sorted oldest->newest.

        Collects unique image paths but does NOT load pixmaps eagerly.
        Images are loaded lazily by _ensure_news_pixmap() when the
        scanner actually visits this tile, keeping startup fast.
        """
        self._news_articles = sorted(articles, key=lambda a: a.date)

        # Collect unique image paths (no QPixmap loading yet)
        self._news_image_paths: list = []
        seen: set = set()
        for art in self._news_articles:
            if art.local_image_path and art.local_image_path not in seen:
                seen.add(art.local_image_path)
                self._news_image_paths.append(art.local_image_path)

        self._news_pixmaps = []
        self._news_pixmap_idx = 0
        self._news_images_loaded = False

        # Load only the FIRST image so the tile has a preview immediately
        if self._news_image_paths:
            pm = QtGui.QPixmap(self._news_image_paths[0])
            if not pm.isNull():
                _pi_max = 256 if _IS_PI else 512
                if pm.width() > _pi_max or pm.height() > _pi_max:
                    pm = pm.scaled(_pi_max, _pi_max, QtCore.Qt.KeepAspectRatio,
                                   QtCore.Qt.SmoothTransformation)
                self._news_pixmaps.append(pm)
            self._pixmap_news = self._news_pixmaps[0] if self._news_pixmaps else None
            log.info("News pixmap loaded for %s: %d images from %d articles",
                     self.tile.tile_id, len(self._news_image_paths),
                     len(self._news_articles))
        else:
            self._pixmap_news = None
            log.debug("No news images for tile %s (%d articles)",
                      self.tile.tile_id, len(articles))
        self.update()

    _LAZY_BATCH = 5  # images per tick during batched lazy-load

    def _ensure_all_news_pixmaps(self) -> None:
        """Lazily load remaining news pixmaps when scanner visits this tile.

        Only the first image is loaded eagerly in set_news_data().
        The rest are loaded here in small batches across QTimer ticks
        so that the GIL is released between batches, keeping the audio
        pipeline fed continuously (429 images in one shot would block
        the GIL for seconds and cause audible buffer underruns).
        """
        if self._news_images_loaded:
            return
        self._news_images_loaded = True

        if len(self._news_image_paths) <= 1:
            return  # already loaded the only image

        self._lazy_load_idx = 1  # skip index 0 (already loaded)
        self._lazy_load_batch()

    def _lazy_load_batch(self) -> None:
        """Load a small batch of news images, then yield to the event loop."""
        try:
            end = min(self._lazy_load_idx + self._LAZY_BATCH,
                      len(self._news_image_paths))
            for i in range(self._lazy_load_idx, end):
                path = self._news_image_paths[i]
                pm = QtGui.QPixmap(path)
                if not pm.isNull():
                    _pi_max = 256 if _IS_PI else 512
                    if pm.width() > _pi_max or pm.height() > _pi_max:
                        pm = pm.scaled(_pi_max, _pi_max, QtCore.Qt.KeepAspectRatio,
                                       QtCore.Qt.SmoothTransformation)
                    self._news_pixmaps.append(pm)
            self._lazy_load_idx = end
            if end < len(self._news_image_paths):
                QtCore.QTimer.singleShot(5, self._lazy_load_batch)
            else:
                log.debug("Lazy-loaded %d/%d news images for %s",
                          len(self._news_pixmaps), len(self._news_image_paths),
                          self.tile.tile_id)
        except Exception as exc:
            log.warning("News image lazy-load error for %s: %s",
                        self.tile.tile_id, exc)

    def advance_news_image(self) -> bool:
        """Advance to the next news image. Returns True if there are more."""
        self._ensure_all_news_pixmaps()  # lazy-load on first use
        if len(self._news_pixmaps) <= 1:
            return False
        self._news_pixmap_idx = (self._news_pixmap_idx + 1) % len(self._news_pixmaps)
        self._pixmap_news = self._news_pixmaps[self._news_pixmap_idx]
        self.update()
        return True

    @property
    def news_image_count(self) -> int:
        return len(self._news_pixmaps)

    def hoverEnterEvent(self, event):
        self._hovered = True
        self.update()
        self._parent_widget.tile_hovered.emit(self.tile.tile_id)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        self._parent_widget.tile_clicked.emit(self.tile.tile_id)
        super().mousePressEvent(event)


# ── Map auto-scanner ─────────────────────────────────────────────────

class MapScanner(QtCore.QObject):
    """Narrative-driven camera controller for the San Andreas Fault map.

    Receives Shot sequences from a NarrativeEngine and executes them as
    smooth camera animations with overlay placement.  The engine decides
    *what* to show; the scanner decides *how* to show it.

    Signals
    -------
    scanning_tile(str)
        Emitted when the scanner focuses on a tile.
    scan_overview()
        Emitted when returning to full-map view.
    """

    scanning_tile = QtCore.pyqtSignal(str)
    scan_overview = QtCore.pyqtSignal()

    def __init__(
        self,
        map_widget: "FaultMapWidget",
        overview_s: float = 3.0,
        parent: Optional[QtCore.QObject] = None,
    ):
        super().__init__(parent)
        self._map = map_widget
        self._overview_ms = int(overview_s * 1000)
        self._running = False

        # Narrative engine (created on start)
        self._engine: Optional["NarrativeEngine"] = None
        self._engine_initialized = False

        # Animation
        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.setInterval(33 if _IS_PI else 16)
        self._anim_timer.timeout.connect(self._animate_step)

        # Shot timer — fires when the current shot's hold time expires
        self._scan_timer = QtCore.QTimer(self)
        self._scan_timer.setSingleShot(True)
        self._scan_timer.timeout.connect(self._advance_shot)

        self._phase = "idle"
        self._current_tile_id: Optional[str] = None

        # Current chapter state
        self._current_chapter: Optional["Chapter"] = None
        self._shot_idx = 0

        # Overlay graphics items
        self._news_overlay: Optional[QtWidgets.QGraphicsPixmapItem] = None
        self._news_border: Optional[QtWidgets.QGraphicsRectItem] = None
        self._history_overlay: Optional[QtWidgets.QGraphicsPixmapItem] = None
        self._history_border: Optional[QtWidgets.QGraphicsRectItem] = None

        # Animation state
        self._anim_start_time = 0.0
        self._anim_duration = 0.7
        self._anim_from_rect: Optional[QtCore.QRectF] = None
        self._anim_to_rect: Optional[QtCore.QRectF] = None

        # Sensor data caches (forwarded to engine)
        self._news_tile_ids: set = set()
        self._history_tile_ids: set = set()

    def _ensure_engine(self) -> "NarrativeEngine":
        from .narrative_engine import NarrativeEngine
        if self._engine is None:
            self._engine = NarrativeEngine()
        if not self._engine_initialized and self._map._tile_items:
            fault_tiles = []
            fault_km: Dict[str, float] = {}
            sections: Dict[str, str] = {}
            for tid, item in self._map._tile_items.items():
                if item.tile.on_fault:
                    fault_tiles.append(tid)
                    fault_km[tid] = item.tile.fault_distance_km
                    sections[tid] = item.tile.section
            fault_tiles.sort(key=lambda t: fault_km[t])
            self._engine.set_tile_info(fault_tiles, fault_km, sections)
            self._engine_initialized = True
            log.info("NarrativeEngine initialized with %d fault tiles", len(fault_tiles))
        return self._engine

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._ensure_engine()
        self._phase = "overview"
        self._scan_timer.start(1500)
        log.info("MapScanner started (narrative mode)")

    def stop(self) -> None:
        self._running = False
        self._anim_timer.stop()
        self._scan_timer.stop()
        self._remove_news_overlay()
        self._remove_history_overlay()
        if self._current_tile_id:
            item = self._map._tile_items.get(self._current_tile_id)
            if item:
                item.set_scanner_focus(False)
        self._current_tile_id = None
        self._current_chapter = None
        self._phase = "idle"
        log.info("MapScanner stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Data updates (forwarded to engine) ────────────────────────────

    def update_scores(self, scores: Dict[str, float]) -> None:
        eng = self._ensure_engine()
        eng.update_scores(scores)

    def update_quake_counts(self, counts: Dict[str, int]) -> None:
        eng = self._ensure_engine()
        eng.update_quake_counts(counts)

    def update_news_tiles(self, tile_ids: set) -> None:
        self._news_tile_ids = set(tile_ids)
        eng = self._ensure_engine()
        img_set = set()
        img_counts: Dict[str, int] = {}
        for tid in tile_ids:
            item = self._map._tile_items.get(tid)
            if item and item._pixmap_news:
                img_set.add(tid)
                img_counts[tid] = max(1, len(item._news_image_paths))
        eng.update_news_tiles(tile_ids, img_set, img_counts)
        total_images = sum(img_counts.values())
        log.info("MapScanner: %d news tiles, %d total images", len(tile_ids), total_images)

    def update_history_tiles(self, tile_ids: set) -> None:
        self._history_tile_ids = set(tile_ids)
        eng = self._ensure_engine()
        img_set = set()
        theme_counts: Dict[str, int] = {}
        img_counts: Dict[str, int] = {}
        for tid in tile_ids:
            item = self._map._tile_items.get(tid)
            if item:
                if item._pixmap_history:
                    img_set.add(tid)
                    img_counts[tid] = max(1, len(item._history_image_paths))
                themes = set()
                for ev in item._history_events:
                    for t in ev.theme.split(","):
                        t = t.strip()
                        if t:
                            themes.add(t)
                theme_counts[tid] = len(themes)
        eng.update_history_tiles(tile_ids, img_set, theme_counts, img_counts)
        total_images = sum(img_counts.values())
        log.info("MapScanner: %d history tiles, %d total images", len(tile_ids), total_images)

    def queue_anomaly_alert(self, tile_ids: list) -> None:
        """Called by GUI when ML detects an anomaly."""
        eng = self._ensure_engine()
        eng.queue_anomaly_alert(tile_ids)

    # ── Geometry helpers ──────────────────────────────────────────────

    def _tile_rect(self, tile_id: str) -> Optional[QtCore.QRectF]:
        item = self._map._tile_items.get(tile_id)
        return item._rect if item else None

    def _bounding_rect(self, tile_ids: List[str]) -> QtCore.QRectF:
        rects = []
        for tid in tile_ids:
            item = self._map._tile_items.get(tid)
            if item:
                rects.append(item._rect)
        if not rects:
            return self._map._fault_rect
        x_min = min(r.x() for r in rects)
        y_min = min(r.y() for r in rects)
        x_max = max(r.x() + r.width() for r in rects)
        y_max = max(r.y() + r.height() for r in rects)
        pw = rects[0].width() * 2.5
        ph = rects[0].height() * 2.5
        return QtCore.QRectF(
            x_min - pw, y_min - ph,
            (x_max - x_min) + 2 * pw, (y_max - y_min) + 2 * ph,
        )

    def _multi_tile_rect(self, tile_id: str, pm: QtGui.QPixmap) -> QtCore.QRectF:
        """Return a scene rect spanning multiple tiles around *tile_id*.

        Square images always span at least 2x2 (4 tiles).  If some
        neighbor tiles are missing at grid edges, the rect is still
        sized to the intended span using the anchor tile dimensions.
        """
        anchor = self._map._tile_items.get(tile_id)
        if not anchor:
            return QtCore.QRectF()

        ar = pm.width() / max(pm.height(), 1)
        if ar > 1.3:
            span_c, span_r = 3, 2
        elif ar < 0.77:
            span_c, span_r = 2, 3
        else:
            span_c, span_r = 2, 2

        row0 = anchor.tile.row
        col0 = anchor.tile.col
        tw = anchor._rect.width()
        th = anchor._rect.height()

        rc_map = self._map._get_rc_map()

        rects: list = []
        for dr in range(span_r):
            for dc in range(span_c):
                nb = rc_map.get((row0 + dr, col0 + dc))
                if nb:
                    rects.append(nb._rect)

        if rects:
            x_min = min(r.x() for r in rects)
            y_min = min(r.y() for r in rects)
            x_max = max(r.x() + r.width() for r in rects)
            y_max = max(r.y() + r.height() for r in rects)
            w = max(x_max - x_min, tw * span_c)
            h = max(y_max - y_min, th * span_r)
            return QtCore.QRectF(x_min, y_min, w, h)

        return QtCore.QRectF(
            anchor._rect.x(), anchor._rect.y(),
            tw * span_c, th * span_r,
        )

    # ── Overlay management ────────────────────────────────────────────

    def _remove_news_overlay(self) -> None:
        if self._news_overlay is not None:
            self._map._scene.removeItem(self._news_overlay)
            self._news_overlay = None
        if self._news_border is not None:
            self._map._scene.removeItem(self._news_border)
            self._news_border = None

    def _show_news_overlay(self, tile_id: str, image_idx: int = -1) -> None:
        if not self._running:
            return
        self._remove_news_overlay()
        item = self._map._tile_items.get(tile_id)
        if not item or not item._pixmap_news:
            return
        item._ensure_all_news_pixmaps()

        pixmaps = item._news_pixmaps
        if not pixmaps:
            return
        if 0 <= image_idx < len(pixmaps):
            pm = pixmaps[image_idx]
        else:
            pm = random.choice(pixmaps)
        if pm.width() <= 0 or pm.height() <= 0:
            return

        span_r = self._multi_tile_rect(tile_id, pm)
        tw, th = span_r.width(), span_r.height()
        scale = min(tw / pm.width(), th / pm.height())
        img_w = pm.width() * scale
        img_h = pm.height() * scale
        cx = span_r.x() + (tw - img_w) / 2.0
        cy = span_r.y() + (th - img_h) / 2.0

        overlay = QtWidgets.QGraphicsPixmapItem(pm)
        overlay.setPos(cx, cy)
        overlay.setTransform(QtGui.QTransform.fromScale(scale, scale))
        overlay.setZValue(50)
        overlay.setOpacity(1.0)
        self._map._scene.addItem(overlay)
        self._news_overlay = overlay

        img_rect = QtCore.QRectF(cx, cy, img_w, img_h)
        border = QtWidgets.QGraphicsRectItem(img_rect)
        pen = QtGui.QPen(QtGui.QColor(0, 220, 255, 220))
        pen.setWidthF(2.0)
        pen.setCosmetic(True)
        border.setPen(pen)
        border.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
        border.setZValue(51)
        self._map._scene.addItem(border)
        self._news_border = border

    def _remove_history_overlay(self) -> None:
        if self._history_overlay is not None:
            self._map._scene.removeItem(self._history_overlay)
            self._history_overlay = None
        if self._history_border is not None:
            self._map._scene.removeItem(self._history_border)
            self._history_border = None

    def _show_history_overlay(self, tile_id: str, image_idx: int = -1) -> None:
        if not self._running:
            return
        self._remove_history_overlay()
        item = self._map._tile_items.get(tile_id)
        if not item or not item._pixmap_history:
            return
        item._ensure_all_history_pixmaps()

        pixmaps = item._history_pixmaps
        if not pixmaps:
            return
        if 0 <= image_idx < len(pixmaps):
            pm = pixmaps[image_idx]
        else:
            pm = random.choice(pixmaps)
        if pm.width() <= 0 or pm.height() <= 0:
            return

        span_r = self._multi_tile_rect(tile_id, pm)
        tw, th = span_r.width(), span_r.height()
        scale = min(tw / pm.width(), th / pm.height())
        img_w = pm.width() * scale
        img_h = pm.height() * scale
        cx = span_r.x() + (tw - img_w) / 2.0
        cy = span_r.y() + (th - img_h) / 2.0

        overlay = QtWidgets.QGraphicsPixmapItem(pm)
        overlay.setPos(cx, cy)
        overlay.setTransform(QtGui.QTransform.fromScale(scale, scale))
        overlay.setZValue(50)
        overlay.setOpacity(1.0)
        self._map._scene.addItem(overlay)
        self._history_overlay = overlay

        img_rect = QtCore.QRectF(cx, cy, img_w, img_h)
        border = QtWidgets.QGraphicsRectItem(img_rect)
        pen = QtGui.QPen(QtGui.QColor(220, 170, 30, 220))
        pen.setWidthF(2.0)
        pen.setCosmetic(True)
        border.setPen(pen)
        border.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
        border.setZValue(51)
        self._map._scene.addItem(border)
        self._history_border = border

    # ── Shot executor (narrative state machine) ───────────────────────

    def _advance_shot(self) -> None:
        """Execute the next shot in the current chapter, or generate a new chapter."""
        if not self._running:
            return

        # Clean up previous shot
        self._remove_news_overlay()
        self._remove_history_overlay()
        if self._current_tile_id:
            item = self._map._tile_items.get(self._current_tile_id)
            if item:
                item.set_scanner_focus(False)
            self._current_tile_id = None

        # Need a new chapter?
        ch = self._current_chapter
        if ch is None or self._shot_idx >= len(ch.shots):
            self._begin_next_chapter()
            return

        shot = ch.shots[self._shot_idx]
        self._shot_idx += 1
        self._execute_shot(shot)

    def _begin_next_chapter(self) -> None:
        """Ask the narrative engine for the next chapter and start it."""
        from .narrative_engine import Chapter
        eng = self._ensure_engine()
        ch = eng.next_chapter()
        self._current_chapter = ch
        self._shot_idx = 0

        if not ch.shots:
            self._scan_timer.start(2000)
            return

        shot = ch.shots[self._shot_idx]
        self._shot_idx += 1
        self._execute_shot(shot)

    def _execute_shot(self, shot: "Shot") -> None:
        """Execute a single Shot — animate camera and manage overlays."""
        from .narrative_engine import ShotKind

        self._anim_duration = shot.anim_s

        if shot.kind == ShotKind.ESTABLISH:
            self._phase = "establish"
            ch = self._current_chapter
            all_tiles = [ch.primary_tile] + ch.context_tiles if ch else []
            if all_tiles:
                target = self._bounding_rect(all_tiles)
            else:
                target = self._map._fault_rect
            self._animate_to_rect(target)
            self.scan_overview.emit()
            self._scan_timer.start(shot.total_ms)

        elif shot.kind == ShotKind.APPROACH:
            self._phase = "approach"
            tid = shot.tile_id
            if tid:
                tr = self._tile_rect(tid)
                if tr:
                    pad = max(tr.width(), tr.height()) * 3.0
                    target = tr.adjusted(-pad, -pad, pad, pad)
                    self._animate_to_rect(target)
            self._scan_timer.start(shot.total_ms)

        elif shot.kind == ShotKind.FOCUS:
            self._phase = "focus"
            tid = shot.tile_id
            if tid:
                item = self._map._tile_items.get(tid)
                if item:
                    self._current_tile_id = tid
                    item.set_scanner_focus(True)
                    self.scanning_tile.emit(tid)

                    src_tile = shot.overlay_source_tile or tid
                    img_idx = shot.overlay_image_idx if shot.overlay_image_idx is not None else -1

                    if shot.overlay == "news":
                        self._show_news_overlay(src_tile, img_idx)
                    elif shot.overlay == "history":
                        self._show_history_overlay(src_tile, img_idx)

                    has_overlay = shot.overlay in ("news", "history")
                    if has_overlay:
                        src_item = self._map._tile_items.get(src_tile)
                        pm = None
                        if src_item:
                            if shot.overlay == "news" and src_item._pixmap_news:
                                pm = src_item._pixmap_news
                            elif shot.overlay == "history" and src_item._pixmap_history:
                                pm = src_item._pixmap_history
                        if pm:
                            overlay_r = self._multi_tile_rect(src_tile, pm)
                            pad = max(overlay_r.width(), overlay_r.height()) * 1.0
                            target = overlay_r.adjusted(-pad, -pad, pad, pad)
                        else:
                            tr = item._rect
                            pad = max(tr.width(), tr.height()) * 3.0
                            target = tr.adjusted(-pad, -pad, pad, pad)
                    else:
                        tr = item._rect
                        pad = max(tr.width(), tr.height()) * 2.0
                        target = tr.adjusted(-pad, -pad, pad, pad)
                    if shot.anim_s > 0:
                        self._animate_to_rect(target)
            self._scan_timer.start(shot.total_ms)

        elif shot.kind == ShotKind.CONTEXT:
            self._phase = "context"
            tid = shot.tile_id
            if tid:
                item = self._map._tile_items.get(tid)
                if item:
                    if self._current_tile_id:
                        prev = self._map._tile_items.get(self._current_tile_id)
                        if prev:
                            prev.set_scanner_focus(False)
                    self._current_tile_id = tid
                    item.set_scanner_focus(True)
                    self.scanning_tile.emit(tid)

                    tr = item._rect
                    pad = max(tr.width(), tr.height()) * 2.0
                    target = tr.adjusted(-pad, -pad, pad, pad)
                    self._animate_to_rect(target)
            self._scan_timer.start(shot.total_ms)

        elif shot.kind == ShotKind.RELEASE:
            self._phase = "release"
            self._animate_to_rect(self._map._fault_rect)
            self.scan_overview.emit()
            self._scan_timer.start(shot.total_ms)

        else:
            self._scan_timer.start(1000)

    # ── Animation ─────────────────────────────────────────────────────

    def _animate_to_rect(self, target: QtCore.QRectF) -> None:
        self._anim_from_rect = self._map._view.mapToScene(
            self._map._view.viewport().rect()
        ).boundingRect()
        self._anim_to_rect = target
        self._anim_start_time = time.time()
        self._anim_timer.start()

    def _animate_step(self) -> None:
        if not self._anim_from_rect or not self._anim_to_rect:
            self._anim_timer.stop()
            return

        elapsed = time.time() - self._anim_start_time
        dur = max(self._anim_duration, 0.01)
        t = min(elapsed / dur, 1.0)

        if t < 0.5:
            ease = 4 * t * t * t
        else:
            ease = 1 - pow(-2 * t + 2, 3) / 2

        f = self._anim_from_rect
        to = self._anim_to_rect
        x = f.x() + (to.x() - f.x()) * ease
        y = f.y() + (to.y() - f.y()) * ease
        w = f.width() + (to.width() - f.width()) * ease
        h = f.height() + (to.height() - f.height()) * ease

        interp = QtCore.QRectF(x, y, w, h)
        self._map._view.fitInView(interp, QtCore.Qt.KeepAspectRatio)

        if t >= 1.0:
            self._anim_timer.stop()


# ── Main map widget ───────────────────────────────────────────────────

class FaultMapWidget(QtWidgets.QWidget):
    """Interactive map of the San Andreas Fault tile grid.

    Signals
    -------
    tile_clicked(str)
        Emitted when a tile is clicked (tile_id).
    tile_hovered(str)
        Emitted on hover (tile_id).
    """

    tile_clicked = QtCore.pyqtSignal(str)
    tile_hovered = QtCore.pyqtSignal(str)
    tiles_loaded = QtCore.pyqtSignal()  # emitted when all tile pixmaps are loaded

    SCENE_SCALE = 1.0 / 100.0
    _TILE_LOAD_BATCH = 10  # pixmaps loaded per event-loop cycle (keep small to avoid GIL starvation)
    _TILE_LOAD_DELAY_MS = 5  # ms between batches — lets DSP/audio threads run

    def __init__(
        self,
        tiles: List[Tile],
        site_lonlat: Optional[Tuple[float, float]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._tiles = tiles
        self._tile_items: Dict[str, TileItem] = {}

        # Compute scene extents
        all_x, all_y = [], []
        for t in tiles:
            x0, y0, x1, y1 = t.bounds_m
            all_x.extend([x0, x1])
            all_y.extend([y0, y1])

        self._origin_x = min(all_x)
        self._origin_y = min(all_y)
        self._extent_x = max(all_x) - self._origin_x
        self._extent_y = max(all_y) - self._origin_y

        sf = self.SCENE_SCALE

        # Site location (projected)
        if site_lonlat:
            import pyproj
            wgs84 = pyproj.CRS("EPSG:4326")
            aea = pyproj.CRS("EPSG:3310")
            tx = pyproj.Transformer.from_crs(wgs84, aea, always_xy=True)
            self._site_m = tx.transform(site_lonlat[0], site_lonlat[1])
        else:
            cx = self._origin_x + self._extent_x / 2
            cy = self._origin_y + self._extent_y / 2
            self._site_m = (cx, cy)

        # Build scene
        self._scene = QtWidgets.QGraphicsScene(self)
        self._scene.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0)))

        # Build view — optimized for smooth navigation
        self._view = QtWidgets.QGraphicsView(self._scene, self)

        self._view.setRenderHints(
            QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform
        )

        if _IS_KIOSK:
            # Kiosk mode: no user interaction with the map (scanner only)
            self._view.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._view.setInteractive(False)
        else:
            self._view.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

        self._view.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self._view.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self._view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._view.setViewportUpdateMode(
            QtWidgets.QGraphicsView.MinimalViewportUpdate
        )

        # Performance optimization flags
        self._view.setOptimizationFlag(
            QtWidgets.QGraphicsView.DontSavePainterState, True
        )
        self._view.setOptimizationFlag(
            QtWidgets.QGraphicsView.DontAdjustForAntialiasing, True
        )

        self._view.setStyleSheet("border: none; background: #000000;")

        # ── Layout: view fills entire widget, controls float on top ──
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._view, 1)

        # Semi-transparent button style
        _BTN_SS = (
            "QPushButton { background: rgba(6,10,16,180); color: #506880; "
            "border: 1px solid rgba(12,26,46,180); padding: 2px 8px; "
            "font-family: 'Helvetica Neue Mono'; font-size: 10px; }"
            "QPushButton:hover { background: rgba(16,42,64,200); color: #00ccff; }"
        )

        # ── Top-right floating controls ──
        self._overlay_top = QtWidgets.QWidget(self._view)
        self._overlay_top.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
        self._overlay_top.setStyleSheet("background: transparent;")
        otl = QtWidgets.QHBoxLayout(self._overlay_top)
        otl.setContentsMargins(6, 4, 6, 0)
        otl.setSpacing(4)

        _LABEL_BG = "background: rgba(0,0,0,180);"

        self._info_label = QtWidgets.QLabel(
            f"S.A.R — {len(tiles)} tiles"
        )
        self._info_label.setStyleSheet(
            "color: rgba(0,204,255,200); font-family: 'Helvetica Neue'; "
            f"font-size: 11px; padding: 2px 4px; {_LABEL_BG}"
        )
        otl.addWidget(self._info_label)

        self._strategy_label = QtWidgets.QLabel("")
        self._strategy_label.setStyleSheet(
            "color: rgba(180,200,220,200); font-family: 'Helvetica Neue Mono'; "
            f"font-size: 9px; padding: 2px 6px; {_LABEL_BG}"
        )
        otl.addWidget(self._strategy_label)

        otl.addStretch(1)

        # Scanner toggle button
        self._btn_scan = QtWidgets.QPushButton("Scan")
        self._btn_scan.setCheckable(True)
        self._btn_scan.setStyleSheet(
            _BTN_SS
            + "QPushButton:checked { background: rgba(0,24,48,200); "
            "color: #00ccff; border: 1px solid #00ccff; }"
        )
        self._btn_scan.toggled.connect(self._on_scan_toggle)
        otl.addWidget(self._btn_scan)

        # Satellite download button
        self._btn_download = QtWidgets.QPushButton("Satellite")
        self._btn_download.setStyleSheet(_BTN_SS)
        self._btn_download.clicked.connect(self._on_download_sat)
        otl.addWidget(self._btn_download)

        # Fit button
        btn_fit = QtWidgets.QPushButton("Fit")
        btn_fit.setStyleSheet(_BTN_SS)
        btn_fit.clicked.connect(self.fit_to_view)
        otl.addWidget(btn_fit)

        # Kiosk mode: hide nav controls (scanner auto-starts),
        # keep satellite download button visible for first-run cache build
        if _IS_KIOSK:
            self._btn_scan.hide()
            btn_fit.hide()

        # ── Bottom floating status labels ──
        self._overlay_bottom = QtWidgets.QWidget(self._view)
        self._overlay_bottom.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self._overlay_bottom.setStyleSheet("background: transparent;")
        obl = QtWidgets.QVBoxLayout(self._overlay_bottom)
        obl.setContentsMargins(6, 0, 6, 4)
        obl.setSpacing(0)

        self._scan_label = QtWidgets.QLabel("")
        self._scan_label.setStyleSheet(
            "color: rgba(80,104,128,220); font-family: 'Helvetica Neue Mono'; "
            f"font-size: 9px; padding: 1px 4px; {_LABEL_BG}"
        )
        obl.addWidget(self._scan_label)

        self._detail_label = QtWidgets.QLabel("")
        self._detail_label.setStyleSheet(
            "color: rgba(112,136,152,220); font-family: 'Helvetica Neue Mono'; "
            f"font-size: 10px; padding: 1px 4px; {_LABEL_BG}"
        )
        obl.addWidget(self._detail_label)

        # Populate scene
        self._add_tiles()
        self._add_vector_overlay()
        self._add_labels()
        self._add_fault_trace()
        self._add_site_marker()
        self._add_compass()
        self._add_scale_bar()

        # Cache static overlay items (paths, labels, markers).
        # TileItems use NoCache (they change with zoom), but vector overlays,
        # city labels, fault trace, etc. never change — caching saves CPU.
        for scene_item in self._scene.items():
            if not isinstance(scene_item, TileItem):
                scene_item.setCacheMode(
                    QtWidgets.QGraphicsItem.DeviceCoordinateCache
                )

        # Connect signals
        self.tile_hovered.connect(self._on_tile_hover)
        self.tile_clicked.connect(self._on_tile_click)

        # Map scanner
        self._scanner = MapScanner(self, overview_s=3.0)
        self._scanner.scanning_tile.connect(self._on_scanner_tile)
        self._scanner.scan_overview.connect(self._on_scanner_overview)

        # Auto-start map scanner once all tiles are loaded
        self.tiles_loaded.connect(self._auto_start_scanner)

        # Compute fault corridor bounding rect
        self._fault_rect = self._compute_fault_rect()

        # Compute a zoomed-in central section for initial view
        self._initial_rect = self._compute_central_rect()

    def _auto_start_scanner(self) -> None:
        """Auto-start the map scanner after tiles are loaded.

        Delayed by 8 seconds so the SDR audio pipeline can fill its buffer
        before the scanner triggers heavy image loading (GIL contention).
        """
        log.info("Map scanner will auto-start in 8 s")
        self._scan_label.setText("Scanner starting in 8 s…")
        QtCore.QTimer.singleShot(8000, self._do_auto_start_scanner)

    def _do_auto_start_scanner(self) -> None:
        if self._scanner._running:
            return  # already running
        log.info("Auto-starting map scanner")
        self._scanner.start()
        self._scan_label.setText("SCANNING — autonomous navigation active")

    # ── Tile grid helpers ─────────────────────────────────────────────

    def _get_rc_map(self) -> Dict[tuple, "TileItem"]:
        """Cached (row, col) → TileItem lookup for multi-tile operations."""
        if not hasattr(self, "_rc_map_cache") or self._rc_map_cache is None:
            self._rc_map_cache: Dict[tuple, TileItem] = {}
            for item in self._tile_items.values():
                self._rc_map_cache[(item.tile.row, item.tile.col)] = item
        return self._rc_map_cache

    # ── Scene construction ────────────────────────────────────────────

    def _add_tiles(self) -> None:
        """Create all TileItem objects (fast, no pixmap loading).

        Pixmaps are loaded in batches via _load_tile_batch() so the GUI
        event loop is never blocked for more than ~100-250 ms.  This is
        essential for keeping the SDR audio pipeline glitch-free on Pi.
        """
        sf = self.SCENE_SCALE
        for tile in self._tiles:
            item = TileItem(tile, sf, self._origin_x, self._origin_y, self)
            self._scene.addItem(item)
            self._tile_items[tile.tile_id] = item

        # Start staggered pixmap loading (non-blocking batches).
        # Small delay between batches keeps the GIL free for the SDR/audio
        # threads that are already running by this point.
        self._tile_load_queue = list(self._tile_items.values())
        self._tile_load_idx = 0
        QtCore.QTimer.singleShot(self._TILE_LOAD_DELAY_MS,
                                 self._load_tile_batch)

    def _load_tile_batch(self) -> None:
        """Load a batch of tile pixmaps, then yield to the event loop.

        Each batch loads _TILE_LOAD_BATCH tiles from disk cache.
        A small delay between batches prevents GIL starvation of the
        SDR/DSP and audio playback threads during startup.
        """
        try:
            end = min(self._tile_load_idx + self._TILE_LOAD_BATCH,
                      len(self._tile_load_queue))

            for i in range(self._tile_load_idx, end):
                item = self._tile_load_queue[i]
                if not item._sat_loaded:
                    img_path = get_tile_image_path(item.tile.tile_id)
                    if img_path:
                        item._load_sat_image_no_update(img_path)
                    else:
                        item._sat_loaded = True

            self._tile_load_idx = end

            if self._tile_load_idx < len(self._tile_load_queue):
                loaded = self._tile_load_idx
                total = len(self._tile_load_queue)
                if loaded % 500 == 0:
                    log.info("Tile loading: %d / %d (%.0f%%)",
                             loaded, total, 100.0 * loaded / total)
                QtCore.QTimer.singleShot(self._TILE_LOAD_DELAY_MS,
                                         self._load_tile_batch)
            else:
                log.info("All %d tile pixmaps loaded", len(self._tile_load_queue))
                self._tile_load_queue = []
                self._scene.update()
                self.tiles_loaded.emit()
        except Exception as exc:
            log.error("Tile batch load error: %s", exc)

    def _add_vector_overlay(self) -> None:
        """Draw bold vector data (roads, coastline, borders, faults).

        Styled to match the reference: bright white/light lines clearly
        visible over dark satellite imagery at every zoom level.
        """
        try:
            from ..geo.vector_data import get_vector_layers
        except ImportError:
            log.warning("Vector data not available")
            return

        sf = self.SCENE_SCALE
        layers = get_vector_layers()

        # Bold style definitions — bright white lines like the reference
        styles = {
            "coastline": {
                "color": QtGui.QColor(200, 210, 220, 180),
                "glow_color": QtGui.QColor(150, 170, 190, 50),
                "width": 2.0,
                "glow_width": 5.0,
                "dash": False,
                "z": 7,
            },
            "highways": {
                "color": QtGui.QColor(180, 190, 200, 150),
                "glow_color": QtGui.QColor(100, 120, 140, 30),
                "width": 1.2,
                "glow_width": 3.5,
                "dash": False,
                "z": 6,
            },
            "borders": {
                "color": QtGui.QColor(160, 170, 180, 120),
                "glow_color": None,
                "width": 1.0,
                "glow_width": 0,
                "dash": True,
                "z": 6,
            },
            "faults": {
                "color": QtGui.QColor(0, 180, 240, 140),
                "glow_color": QtGui.QColor(0, 120, 200, 40),
                "width": 1.2,
                "glow_width": 4.0,
                "dash": True,
                "z": 8,
            },
        }

        for layer_name, polylines in layers.items():
            style = styles.get(layer_name, {
                "color": QtGui.QColor(150, 160, 170, 100),
                "glow_color": None,
                "width": 0.8,
                "glow_width": 0,
                "dash": False,
                "z": 5,
            })

            for coords in polylines:
                if len(coords) < 2:
                    continue

                path = QtGui.QPainterPath()
                for i, (x, y) in enumerate(coords):
                    sx = (x - self._origin_x) * sf
                    sy = -(y - self._origin_y) * sf
                    if i == 0:
                        path.moveTo(sx, sy)
                    else:
                        path.lineTo(sx, sy)

                # Glow layer (wider, semi-transparent backdrop)
                if style["glow_color"]:
                    glow_pen = QtGui.QPen(style["glow_color"])
                    glow_pen.setWidthF(style["glow_width"])
                    glow_pen.setCapStyle(QtCore.Qt.RoundCap)
                    glow_pen.setJoinStyle(QtCore.Qt.RoundJoin)
                    glow_item = self._scene.addPath(path, glow_pen)
                    glow_item.setZValue(style["z"] - 1)

                # Main line
                pen = QtGui.QPen(style["color"])
                pen.setWidthF(style["width"])
                pen.setCapStyle(QtCore.Qt.RoundCap)
                pen.setJoinStyle(QtCore.Qt.RoundJoin)
                if style["dash"]:
                    pen.setStyle(QtCore.Qt.DashLine)

                item = self._scene.addPath(path, pen)
                item.setZValue(style["z"])

    def _add_labels(self) -> None:
        """Draw city names, road labels, geographic features on the map.

        Labels use a dark halo (shadow text underneath) for readability
        over both dark and bright satellite imagery.
        """
        try:
            from ..geo.vector_data import get_labels
        except ImportError:
            log.warning("Label data not available")
            return

        sf = self.SCENE_SCALE
        labels = get_labels()

        # Bold styles — bright and readable like the reference
        styles = {
            "city": {
                "color": QtGui.QColor(240, 245, 250, 220),
                "shadow": QtGui.QColor(0, 0, 0, 180),
                "font_size": 7,
                "bold": True,
                "dot": True,
                "dot_r": 3.0,
                "dot_color": QtGui.QColor(240, 245, 250, 200),
                "z": 22,
            },
            "town": {
                "color": QtGui.QColor(200, 215, 230, 190),
                "shadow": QtGui.QColor(0, 0, 0, 160),
                "font_size": 5,
                "bold": False,
                "dot": True,
                "dot_r": 1.8,
                "dot_color": QtGui.QColor(200, 215, 230, 140),
                "z": 20,
            },
            "road": {
                "color": QtGui.QColor(180, 195, 210, 170),
                "shadow": QtGui.QColor(0, 0, 0, 140),
                "font_size": 4,
                "bold": True,
                "dot": False,
                "z": 19,
            },
            "feature": {
                "color": QtGui.QColor(150, 175, 200, 140),
                "shadow": QtGui.QColor(0, 0, 0, 120),
                "font_size": 4,
                "bold": False,
                "dot": False,
                "z": 17,
                "italic": True,
            },
            "fault": {
                "color": QtGui.QColor(0, 190, 255, 160),
                "shadow": QtGui.QColor(0, 0, 0, 140),
                "font_size": 5,
                "bold": False,
                "dot": False,
                "z": 19,
                "italic": True,
            },
        }

        for lbl in labels:
            style = styles.get(lbl.category, styles["town"])

            sx = (lbl.x_m - self._origin_x) * sf
            sy = -(lbl.y_m - self._origin_y) * sf

            font = QtGui.QFont("Helvetica Neue", int(style["font_size"]))
            if style.get("bold"):
                font.setBold(True)
            if style.get("italic"):
                font.setItalic(True)

            offset_x = 4 if style.get("dot") else 0
            offset_y = -4

            # Shadow text (dark halo behind for readability)
            shadow = self._scene.addText(lbl.name, font)
            shadow.setDefaultTextColor(style["shadow"])
            shadow.setPos(sx + offset_x + 0.5, sy + offset_y + 0.5)
            shadow.setZValue(style["z"] - 1)
            if lbl.rotation != 0:
                shadow.setRotation(lbl.rotation)

            # Main text
            text = self._scene.addText(lbl.name, font)
            text.setDefaultTextColor(style["color"])
            text.setPos(sx + offset_x, sy + offset_y)
            text.setZValue(style["z"])
            if lbl.rotation != 0:
                text.setRotation(lbl.rotation)

            # City/town dot marker
            if style.get("dot"):
                dot_r = style.get("dot_r", 1.5)
                # Outer ring
                ring = QtWidgets.QGraphicsEllipseItem(
                    sx - dot_r, sy - dot_r, dot_r * 2, dot_r * 2
                )
                ring_pen = QtGui.QPen(style["dot_color"])
                ring_pen.setWidthF(0.6)
                ring.setPen(ring_pen)
                ring.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
                ring.setZValue(style["z"])
                self._scene.addItem(ring)
                # Center dot
                inner_r = dot_r * 0.4
                dot = QtWidgets.QGraphicsEllipseItem(
                    sx - inner_r, sy - inner_r, inner_r * 2, inner_r * 2
                )
                dot.setPen(QtGui.QPen(QtCore.Qt.NoPen))
                dot.setBrush(QtGui.QBrush(style["dot_color"]))
                dot.setZValue(style["z"])
                self._scene.addItem(dot)

    def _add_fault_trace(self) -> None:
        from ..geo.fault_trace import fault_trace_line
        from shapely.ops import transform
        import pyproj

        wgs84 = pyproj.CRS("EPSG:4326")
        aea = pyproj.CRS("EPSG:3310")
        to_m = pyproj.Transformer.from_crs(wgs84, aea, always_xy=True).transform

        trace_ll = fault_trace_line()
        trace_m = transform(to_m, trace_ll)

        sf = self.SCENE_SCALE
        path = QtGui.QPainterPath()
        coords = list(trace_m.coords)
        for i, (x, y) in enumerate(coords):
            sx = (x - self._origin_x) * sf
            sy = -(y - self._origin_y) * sf
            if i == 0:
                path.moveTo(sx, sy)
            else:
                path.lineTo(sx, sy)

        # Wide glow
        pen_glow = QtGui.QPen(QtGui.QColor(0, 180, 255, 30))
        pen_glow.setWidthF(12.0)
        pen_glow.setCapStyle(QtCore.Qt.RoundCap)
        pen_glow.setJoinStyle(QtCore.Qt.RoundJoin)
        glow = self._scene.addPath(path, pen_glow)
        glow.setZValue(9)

        # Inner glow
        pen_glow2 = QtGui.QPen(QtGui.QColor(0, 200, 255, 60))
        pen_glow2.setWidthF(5.0)
        pen_glow2.setCapStyle(QtCore.Qt.RoundCap)
        pen_glow2.setJoinStyle(QtCore.Qt.RoundJoin)
        glow2 = self._scene.addPath(path, pen_glow2)
        glow2.setZValue(9)

        # Main line — bold, bright
        pen = QtGui.QPen(QtGui.QColor(0, 220, 255, 200))
        pen.setWidthF(2.0)
        pen.setStyle(QtCore.Qt.DashLine)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        item = self._scene.addPath(path, pen)
        item.setZValue(10)

    def _add_site_marker(self) -> None:
        sf = self.SCENE_SCALE
        sx = (self._site_m[0] - self._origin_x) * sf
        sy = -(self._site_m[1] - self._origin_y) * sf

        arm = 40.0
        pen_cross = QtGui.QPen(QtGui.QColor(255, 255, 255, 200))
        pen_cross.setWidthF(0.8)
        self._scene.addLine(sx - arm, sy, sx + arm, sy, pen_cross).setZValue(20)
        self._scene.addLine(sx, sy - arm, sx, sy + arm, pen_cross).setZValue(20)

        label = self._scene.addText("SITE", QtGui.QFont("Helvetica Neue", 6))
        label.setDefaultTextColor(QtGui.QColor(255, 255, 255, 200))
        label.setPos(sx + 5, sy + 5)
        label.setZValue(20)

        pen_circle = QtGui.QPen(QtGui.QColor(80, 120, 160, 60))
        pen_circle.setWidthF(0.5)
        pen_circle.setStyle(QtCore.Qt.DotLine)

        for radius_km in [25, 50, 100, 200]:
            r_scene = radius_km * 1000.0 * sf
            circle = QtWidgets.QGraphicsEllipseItem(
                sx - r_scene, sy - r_scene, 2 * r_scene, 2 * r_scene
            )
            circle.setPen(pen_circle)
            circle.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
            circle.setZValue(5)
            self._scene.addItem(circle)

            dist_label = self._scene.addText(
                f"{radius_km} km", QtGui.QFont("Helvetica Neue Mono", 4)
            )
            dist_label.setDefaultTextColor(QtGui.QColor(80, 120, 160, 100))
            dist_label.setPos(sx + r_scene * 0.7, sy - r_scene * 0.7 - 8)
            dist_label.setZValue(15)

    def _add_compass(self) -> None:
        rect = self._scene.sceneRect()
        cx = rect.center().x()
        cy = rect.center().y()
        margin = 15
        font = QtGui.QFont("Helvetica Neue", 10)
        color = QtGui.QColor(200, 220, 240, 150)

        for letter, (x, y) in {
            "N": (cx, rect.top() + margin),
            "S": (cx, rect.bottom() - margin - 20),
            "W": (rect.left() + margin, cy),
            "E": (rect.right() - margin - 15, cy),
        }.items():
            label = self._scene.addText(letter, font)
            label.setDefaultTextColor(color)
            label.setPos(x - 6, y)
            label.setZValue(25)

    def _add_scale_bar(self) -> None:
        sf = self.SCENE_SCALE
        rect = self._scene.sceneRect()
        x0 = rect.left() + 30
        y0 = rect.bottom() - 30
        bar_km = 100
        bar_scene = bar_km * 1000.0 * sf

        pen = QtGui.QPen(QtGui.QColor(200, 220, 240, 150))
        pen.setWidthF(1.0)

        self._scene.addLine(x0, y0, x0 + bar_scene, y0, pen).setZValue(25)
        tick = 5
        self._scene.addLine(x0, y0 - tick, x0, y0 + tick, pen).setZValue(25)
        self._scene.addLine(
            x0 + bar_scene, y0 - tick, x0 + bar_scene, y0 + tick, pen
        ).setZValue(25)

        label = self._scene.addText(
            f"{bar_km} km", QtGui.QFont("Helvetica Neue Mono", 5)
        )
        label.setDefaultTextColor(QtGui.QColor(200, 220, 240, 150))
        label.setPos(x0 + bar_scene / 2 - 15, y0 + 3)
        label.setZValue(25)

    # ── Public API ────────────────────────────────────────────────────

    def update_tile(self, tile_id: str, composite: float) -> None:
        item = self._tile_items.get(tile_id)
        if item:
            item.set_score(composite)

    def update_all_tiles(self, scores: Dict[str, float]) -> None:
        for tid, score in scores.items():
            self.update_tile(tid, score)
        # Update scanner queue when scores change
        if self._scanner.is_running:
            self._scanner.update_scores(scores)

    def highlight_tile(self, tile_id: str) -> None:
        item = self._tile_items.get(tile_id)
        if item:
            item._hovered = True
            item.update()

    def _find_tile_at_scene(self, sx: float, sy: float) -> Optional[str]:
        """O(1) tile lookup using cached spatial grid (falls back to linear scan)."""
        if not hasattr(self, "_tile_grid_lookup"):
            grid: Dict[tuple, str] = {}
            for tid, titem in self._tile_items.items():
                r = titem._rect
                cx = r.x() + r.width() / 2
                cy = r.y() + r.height() / 2
                cell_w = r.width() if r.width() > 0 else 1.0
                cell_h = r.height() if r.height() > 0 else 1.0
                key = (int(cx / cell_w), int(cy / cell_h))
                grid[key] = tid
            self._tile_grid_lookup = grid
            self._tile_cell_w = cell_w
            self._tile_cell_h = cell_h

        key = (int(sx / self._tile_cell_w), int(sy / self._tile_cell_h))
        tid = self._tile_grid_lookup.get(key)
        if tid:
            return tid
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                tid = self._tile_grid_lookup.get((key[0] + dx, key[1] + dy))
                if tid and self._tile_items[tid]._rect.contains(sx, sy):
                    return tid
        return None

    def update_earthquakes(self, quakes) -> None:
        import pyproj

        if not hasattr(self, "_quake_markers"):
            self._quake_markers = []
        for item in self._quake_markers:
            self._scene.removeItem(item)
        self._quake_markers.clear()

        tile_quake_counts: Dict[str, int] = {}
        tile_quake_maxmag: Dict[str, float] = {}

        if not quakes:
            for item in self._tile_items.values():
                item.set_quake_data(0, 0.0)
            self._info_label.setText(
                f"S.A.R — {len(self._tiles)} tiles  |  0 earthquakes"
            )
            return

        tx = pyproj.Transformer.from_crs(
            pyproj.CRS("EPSG:4326"), pyproj.CRS("EPSG:3310"), always_xy=True
        )
        sf = self.SCENE_SCALE

        for q in quakes:
            try:
                mx, my = tx.transform(q.lon, q.lat)
            except Exception:
                continue

            sx = (mx - self._origin_x) * sf
            sy = -(my - self._origin_y) * sf

            tid = self._find_tile_at_scene(sx, sy)
            if tid:
                tile_quake_counts[tid] = tile_quake_counts.get(tid, 0) + 1
                tile_quake_maxmag[tid] = max(
                    tile_quake_maxmag.get(tid, 0.0), q.mag
                )

            # ── Draw epicenter marker on the scene ──
            radius = max(1.5, min(q.mag * 2.5, 20.0))

            if q.age_hours < 1:
                color = QtGui.QColor(0, 255, 255, 220)
            elif q.age_hours < 24:
                color = QtGui.QColor(0, 180, 255, 160)
            elif q.age_hours < 72:
                color = QtGui.QColor(60, 120, 180, 120)
            else:
                color = QtGui.QColor(40, 60, 100, 80)

            marker = QtWidgets.QGraphicsEllipseItem(
                sx - radius, sy - radius, radius * 2, radius * 2
            )
            pen = QtGui.QPen(color)
            pen.setWidthF(0.5)
            marker.setPen(pen)
            marker.setBrush(QtGui.QBrush(QtGui.QColor(
                color.red(), color.green(), color.blue(), 40
            )))
            marker.setZValue(15)
            marker.setToolTip(
                f"M{q.mag:.1f} {q.place}\n"
                f"Depth: {q.depth_km:.1f} km\n"
                f"Time: {q.time_utc}\n"
                f"Age: {q.age_hours:.1f}h"
            )
            self._scene.addItem(marker)
            self._quake_markers.append(marker)

        # ── Push per-tile counts to TileItems for badge rendering ──
        for tid, titem in self._tile_items.items():
            cnt = tile_quake_counts.get(tid, 0)
            mmag = tile_quake_maxmag.get(tid, 0.0)
            titem.set_quake_data(cnt, mmag)

        # Update scanner with quake counts for dwell calculation
        self._scanner.update_quake_counts(tile_quake_counts)

        recent = sum(1 for q in quakes if q.age_hours <= 24.0)
        self._info_label.setText(
            f"S.A.R — {len(self._tiles)} tiles  |  "
            f"{len(quakes)} earthquakes ({recent} in 24h)"
        )

    def update_news(self, news_by_tile: dict) -> None:
        """Distribute news articles to tiles and inform the scanner.

        Parameters
        ----------
        news_by_tile : dict
            Maps tile_id → list of ICEArticle objects.
        """
        log.info("update_news called: %d tiles with articles", len(news_by_tile))

        # Clear news from tiles that no longer have articles
        for tid, titem in self._tile_items.items():
            if tid not in news_by_tile:
                if titem._news_articles:
                    titem.set_news_data([])

        # Set news for tiles that have articles
        news_image_tiles = set()
        for tid, articles in news_by_tile.items():
            titem = self._tile_items.get(tid)
            if titem:
                titem.set_news_data(articles)
                if titem._pixmap_news is not None:
                    news_image_tiles.add(tid)
                    log.info("  Tile %s: %d articles, pixmap OK (%dx%d)",
                             tid, len(articles),
                             titem._pixmap_news.width(),
                             titem._pixmap_news.height())
                else:
                    log.warning("  Tile %s: %d articles but NO pixmap loaded!",
                                tid, len(articles))
            else:
                log.warning("  Tile %s not found in tile_items!", tid)

        # Inform scanner so it visits tiles with news images
        self._scanner.update_news_tiles(news_image_tiles)

        total_articles = sum(len(v) for v in news_by_tile.values())
        log.info("News: distributed %d articles across %d tiles (%d with images)",
                 total_articles, len(news_by_tile), len(news_image_tiles))

    def update_history_events(self, events) -> None:
        """Place amber diamond markers for historical events and assign to tiles.

        Parameters
        ----------
        events : list[HistoryEvent]
            Each event has lat, lon, title, description, image_path, period, theme.
        """
        import pyproj

        if not hasattr(self, "_history_markers"):
            self._history_markers = []
        for item in self._history_markers:
            self._scene.removeItem(item)
        self._history_markers.clear()

        if not events:
            return

        tx = pyproj.Transformer.from_crs(
            pyproj.CRS("EPSG:4326"), pyproj.CRS("EPSG:3310"), always_xy=True
        )
        sf = self.SCENE_SCALE

        # Group events by tile for overlay assignment
        tile_events: Dict[str, list] = {}

        for ev in events:
            try:
                mx, my = tx.transform(ev.lon, ev.lat)
            except Exception:
                continue

            sx = (mx - self._origin_x) * sf
            sy = -(my - self._origin_y) * sf

            best_tid = self._find_tile_at_scene(sx, sy)
            if best_tid:
                tile_events.setdefault(best_tid, []).append(ev)

            # Draw amber diamond marker on the scene
            size = 6.0
            diamond = QtGui.QPolygonF([
                QtCore.QPointF(sx, sy - size),
                QtCore.QPointF(sx + size, sy),
                QtCore.QPointF(sx, sy + size),
                QtCore.QPointF(sx - size, sy),
            ])
            marker = QtWidgets.QGraphicsPolygonItem(diamond)
            pen = QtGui.QPen(QtGui.QColor(220, 170, 30, 200))
            pen.setWidthF(0.5)
            pen.setCosmetic(True)
            marker.setPen(pen)
            marker.setBrush(QtGui.QBrush(QtGui.QColor(220, 170, 30, 80)))
            marker.setZValue(14)
            marker.setToolTip(
                f"{ev.title}\n"
                f"{ev.city}, {ev.county}\n"
                f"Date: {ev.date}\n"
                f"Period: {ev.period}\n"
                f"Theme: {ev.theme}"
            )
            self._scene.addItem(marker)
            self._history_markers.append(marker)

        # Assign events to tiles for overlay display
        history_image_tiles = set()
        for tid, evts in tile_events.items():
            titem = self._tile_items.get(tid)
            if titem:
                titem.set_history_data(evts)
                if titem._pixmap_history is not None:
                    history_image_tiles.add(tid)

        self._scanner.update_history_tiles(history_image_tiles)
        log.info(
            "History: %d events, %d markers, %d tiles with images",
            len(events), len(self._history_markers), len(history_image_tiles),
        )

    def set_info(self, text: str) -> None:
        self._info_label.setText(text)

    def _compute_fault_rect(self) -> QtCore.QRectF:
        """Compute the bounding rect of just the on-fault tiles."""
        rects = [
            item._rect for item in self._tile_items.values()
            if item.tile.on_fault
        ]
        if not rects:
            return self._scene.sceneRect()

        x_min = min(r.left() for r in rects)
        y_min = min(r.top() for r in rects)
        x_max = max(r.right() for r in rects)
        y_max = max(r.bottom() for r in rects)

        # Add small padding
        pw = (x_max - x_min) * 0.05
        ph = (y_max - y_min) * 0.05
        return QtCore.QRectF(
            x_min - pw, y_min - ph,
            (x_max - x_min) + 2 * pw, (y_max - y_min) + 2 * ph,
        )

    def _compute_central_rect(self) -> QtCore.QRectF:
        """Compute a zoomed-in rect covering the central 30% of the fault.

        The SAF runs roughly NW-SE, so the "center" is around Parkfield /
        central California (lat ~35-36).  We pick the middle third of the
        on-fault tiles sorted by Y position (which is latitude in scene
        coordinates) so the initial view shows detailed terrain without
        any black borders.
        """
        rects = [
            item._rect for item in self._tile_items.values()
            if item.tile.on_fault
        ]
        if not rects:
            return self._fault_rect

        # Sort by vertical center (scene Y)
        rects.sort(key=lambda r: r.center().y())
        n = len(rects)
        # Central 30% slice
        lo = int(n * 0.35)
        hi = int(n * 0.65)
        if hi <= lo:
            lo, hi = 0, n
        subset = rects[lo:hi]

        x_min = min(r.left() for r in subset)
        y_min = min(r.top() for r in subset)
        x_max = max(r.right() for r in subset)
        y_max = max(r.bottom() for r in subset)

        # Small padding
        pw = (x_max - x_min) * 0.08
        ph = (y_max - y_min) * 0.08
        return QtCore.QRectF(
            x_min - pw, y_min - ph,
            (x_max - x_min) + 2 * pw, (y_max - y_min) + 2 * ph,
        )

    def fit_to_view(self) -> None:
        """Fit the view to show the entire fault corridor."""
        self._view.fitInView(self._fault_rect, QtCore.Qt.KeepAspectRatio)

    # ── Scanner controls ──────────────────────────────────────────────

    _SCAN_SS_ON = (
        "color: #00ccff; font-family: 'Helvetica Neue Mono'; font-size: 9px; "
        "padding: 1px 8px; background: rgba(0,0,0,180);"
    )
    _SCAN_SS_OFF = (
        "color: #506880; font-family: 'Helvetica Neue Mono'; font-size: 9px; "
        "padding: 1px 8px; background: rgba(0,0,0,180);"
    )

    def _on_scan_toggle(self, checked: bool) -> None:
        if checked:
            self._scanner.start()
            self._scan_label.setText("SCANNING — autonomous navigation active")
            self._scan_label.setStyleSheet(self._SCAN_SS_ON)
        else:
            self._scanner.stop()
            self._scan_label.setText("")
            self._scan_label.setStyleSheet(self._SCAN_SS_OFF)
            self._strategy_label.setText("")

    def _update_strategy_label(self) -> None:
        eng = self._scanner._engine
        if not eng or not eng._strategy_history:
            return
        strat = eng._strategy_history[-1].name.replace("_", " ")
        cyc = eng._cycle_count
        total = len(eng._playlist)
        idx = eng._playlist_idx
        self._strategy_label.setText(
            f"C{cyc}  {strat}  {idx}/{total}"
        )

    def _on_scanner_tile(self, tile_id: str) -> None:
        item = self._tile_items.get(tile_id)
        if item:
            t = item.tile
            ch = self._scanner._current_chapter
            ch_type = ch.chapter_type.name if ch else "—"
            qcount = item._quake_count
            self._scan_label.setText(
                f"[{ch_type}] {t.tile_id}  |  {t.section}  |  "
                f"score: {item._score:.3f}  |  quakes: {qcount}  |  "
                f"lat {t.centroid_lonlat[1]:.2f}  lon {t.centroid_lonlat[0]:.2f}"
            )
            self._update_strategy_label()

    def _on_scanner_overview(self) -> None:
        ch = self._scanner._current_chapter
        ch_type = ch.chapter_type.name if ch else "OVERVIEW"
        self._scan_label.setText(f"overview  [{ch_type}]")
        self._update_strategy_label()

    # ── Satellite download ────────────────────────────────────────────

    def _on_download_sat(self) -> None:
        self._btn_download.setEnabled(False)
        self._btn_download.setText("...")
        self._detail_label.setText("Downloading satellite tiles...")

        import threading

        def _worker():
            from ..geo.sat_tiles import download_all_tiles
            sat_count = download_all_tiles(self._tiles)
            QtCore.QMetaObject.invokeMethod(
                self, "_on_download_complete",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(int, sat_count),
            )

        threading.Thread(target=_worker, daemon=True, name="sat-download").start()

    @QtCore.pyqtSlot(int)
    def _on_download_complete(self, count: int) -> None:
        self._btn_download.setText("Satellite")
        self._btn_download.setEnabled(True)

        stats = cache_stats()
        self._detail_label.setText(
            f"Sat: {stats['cached_tiles']} tiles  "
            f"({stats['total_mb']} MB)  |  {stats['nodata_tiles']} no-data"
        )

        # Reload satellite tile images via staggered loading
        self._tile_load_queue = list(self._tile_items.values())
        self._tile_load_idx = 0
        # Reset loaded flags so images are reloaded
        for item in self._tile_load_queue:
            item._sat_loaded = False
        QtCore.QTimer.singleShot(self._TILE_LOAD_DELAY_MS,
                                 self._load_tile_batch)

    # ── Event handlers ────────────────────────────────────────────────

    def _on_tile_hover(self, tile_id: str) -> None:
        item = self._tile_items.get(tile_id)
        if item:
            t = item.tile
            self._detail_label.setText(
                f"{t.tile_id}  |  {t.section}  |  "
                f"{t.fault_distance_km:.0f} km  |  "
                f"lat {t.centroid_lonlat[1]:.3f}  lon {t.centroid_lonlat[0]:.3f}  |  "
                f"score: {item._score:.3f}"
            )

    def _on_tile_click(self, tile_id: str) -> None:
        item = self._tile_items.get(tile_id)
        if item:
            t = item.tile
            self._detail_label.setText(
                f"SELECTED: {t.tile_id}  |  {t.section}  |  "
                f"composite={item._score:.3f}"
            )

    def wheelEvent(self, event):
        """Smooth zoom anchored under the mouse cursor."""
        if _IS_KIOSK:
            event.accept()
            return
        factor = 1.12
        if event.angleDelta().y() > 0:
            self._view.scale(factor, factor)
        else:
            self._view.scale(1.0 / factor, 1.0 / factor)
        event.accept()

    def resizeEvent(self, event):
        """Reposition floating overlays; on first show, zoom into central fault."""
        super().resizeEvent(event)
        vw = self._view.width()
        vh = self._view.height()

        # Reposition top overlay (full width, top)
        self._overlay_top.setGeometry(0, 0, vw, 30)
        # Reposition bottom overlay (full width, bottom)
        self._overlay_bottom.setGeometry(0, vh - 40, vw, 40)

        # Initial fit: zoomed into central fault section
        if not hasattr(self, "_initial_fit_done"):
            self._initial_fit_done = True
            self._view.fitInView(self._initial_rect, QtCore.Qt.KeepAspectRatio)
