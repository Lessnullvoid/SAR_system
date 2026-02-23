"""
Sensor data scheduler — periodic polling of external APIs.

Runs on the Qt event loop using QTimer.  Each sensor source has its own
polling interval.  Results are mapped to tiles and emitted as signals
for the GUI to consume.

Data flow
─────────
  QTimer tick
    → fetch_earthquakes()   (every 2 min)
    → fetch_geomag()        (every 5 min)
    → fetch_gnss()          (every 30 min)
    → fetch_weather()       (every 10 min)
    → fetch_news()          (every 10 min — ICE activity via GDELT)
    → map to tiles / update TileState
    → emit tiles_updated / news_updated signal
    → GUI recolours map + draws markers + shows news

Usage
-----
    scheduler = SensorScheduler(tiles)
    scheduler.tiles_updated.connect(my_handler)
    scheduler.start()
"""
from __future__ import annotations

import logging
import platform
import threading
import time
from typing import Dict, List, Optional

from PyQt5 import QtCore

_IS_PI = (
    platform.system() == "Linux"
    and platform.machine().startswith(("aarch64", "arm"))
)

from ..geo.tile_grid import Tile
from ..geo.tile_state import TileState
from .tile_mapper import TileMapper
from .usgs_client import Earthquake, fetch_fault_corridor_earthquakes
from .noaa_client import get_current_kp, get_current_dst
from .ionospheric_client import get_tec_for_location
from .gnss_client import fetch_gnss_displacements, GNSSReading
from .weather_client import (
    fetch_weather_observations, compute_corridor_summary, WeatherReading,
)
from .ice_client import (
    ICEArticle, fetch_ice_news, backfill_historical, is_backfill_complete,
    download_all_images, save_articles, load_articles,
)
from .census_client import fetch_census_data, CountyDemographics
from .native_land_client import get_saf_territories, IndigenousTerritory
from .scedc_client import fetch_scedc_events, SCEDCEvent

log = logging.getLogger(__name__)


class SensorScheduler(QtCore.QObject):
    """Periodic sensor data poller.

    Signals
    -------
    tiles_updated(dict)
        Emitted when tile states change.  Dict maps tile_id → composite_score.
    earthquakes_updated(list)
        Emitted with the latest earthquake list for map markers.
    geomag_updated(dict)
        Emitted with {'kp': float, 'dst': float, 'storm_level': str}.
    gnss_updated(dict)
        Emitted with GNSS summary info.
    weather_updated(dict)
        Emitted with weather summary info.
    news_updated(dict)
        Emitted with {tile_id: [ICEArticle, ...]} for news feed display.
    status_message(str)
        Informational messages for the status bar.
    """

    tiles_updated = QtCore.pyqtSignal(object)       # dict[str, float]
    earthquakes_updated = QtCore.pyqtSignal(object)  # list[Earthquake]
    geomag_updated = QtCore.pyqtSignal(object)       # dict
    gnss_updated = QtCore.pyqtSignal(object)         # dict
    weather_updated = QtCore.pyqtSignal(object)      # dict
    news_updated = QtCore.pyqtSignal(object)         # dict[str, list[ICEArticle]]
    social_updated = QtCore.pyqtSignal(object)       # dict with census, territories, scedc
    status_message = QtCore.pyqtSignal(str)

    def __init__(
        self,
        tiles: List[Tile],
        usgs_interval_s: float = 120.0,
        geomag_interval_s: float = 300.0,
        gnss_interval_s: float = 1800.0,   # 30 min (daily data, slow update)
        weather_interval_s: float = 600.0,  # 10 min
        news_interval_s: float = 1800.0,    # 30 min (GDELT rate-limits aggressively)
        parent: Optional[QtCore.QObject] = None,
    ):
        super().__init__(parent)
        self._tiles = tiles
        self._mapper = TileMapper(tiles)
        self._tile_states: Dict[str, TileState] = {
            t.tile_id: TileState(tile_id=t.tile_id) for t in tiles
        }
        self._state_lock = threading.Lock()

        # Latest data caches
        self._earthquakes: List[Earthquake] = []
        self._current_kp: float = 0.0
        self._current_dst: float = 0.0
        self._gnss_readings: List[GNSSReading] = []
        self._weather_readings: List[WeatherReading] = []
        self._news_articles: List[ICEArticle] = []
        self._news_by_tile: Dict[str, List[ICEArticle]] = {}

        # Polling timers
        self._usgs_timer = QtCore.QTimer(self)
        self._usgs_timer.setInterval(int(usgs_interval_s * 1000))
        self._usgs_timer.timeout.connect(self._poll_usgs)

        self._geomag_timer = QtCore.QTimer(self)
        self._geomag_timer.setInterval(int(geomag_interval_s * 1000))
        self._geomag_timer.timeout.connect(self._poll_geomag)

        self._gnss_timer = QtCore.QTimer(self)
        self._gnss_timer.setInterval(int(gnss_interval_s * 1000))
        self._gnss_timer.timeout.connect(self._poll_gnss)

        self._weather_timer = QtCore.QTimer(self)
        self._weather_timer.setInterval(int(weather_interval_s * 1000))
        self._weather_timer.timeout.connect(self._poll_weather)

        self._news_timer = QtCore.QTimer(self)
        self._news_timer.setInterval(int(news_interval_s * 1000))
        self._news_timer.timeout.connect(self._poll_news)

        self._running = False

    # ── Control ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start periodic polling."""
        if self._running:
            return
        self._running = True

        # Initial fetch immediately (in background threads)
        self._poll_usgs()
        self._poll_geomag()

        if _IS_PI:
            # Pi: stagger network fetches more aggressively to keep
            # CPU/bandwidth free for the SDR audio pipeline.
            QtCore.QTimer.singleShot(5000, self._poll_news)
            QtCore.QTimer.singleShot(30000, self._poll_gnss)
            QtCore.QTimer.singleShot(20000, self._poll_weather)
            QtCore.QTimer.singleShot(25000, self._poll_social)
        else:
            # Desktop: faster initial data load
            QtCore.QTimer.singleShot(2000, self._poll_news)
            QtCore.QTimer.singleShot(5000, self._poll_gnss)
            QtCore.QTimer.singleShot(8000, self._poll_weather)
            QtCore.QTimer.singleShot(10000, self._poll_social)

        self._usgs_timer.start()
        self._geomag_timer.start()
        self._gnss_timer.start()
        self._weather_timer.start()
        self._news_timer.start()

        log.info(
            "SensorScheduler started (USGS %ds, Geomag %ds, GNSS %ds, "
            "Weather %ds, News %ds)",
            self._usgs_timer.interval() // 1000,
            self._geomag_timer.interval() // 1000,
            self._gnss_timer.interval() // 1000,
            self._weather_timer.interval() // 1000,
            self._news_timer.interval() // 1000,
        )

    def stop(self) -> None:
        self._running = False
        self._usgs_timer.stop()
        self._geomag_timer.stop()
        self._gnss_timer.stop()
        self._weather_timer.stop()
        self._news_timer.stop()
        log.info("SensorScheduler stopped")

    @property
    def tile_states(self) -> Dict[str, TileState]:
        return self._tile_states

    # ── USGS Polling ─────────────────────────────────────────────────

    def _poll_usgs(self) -> None:
        threading.Thread(
            target=self._fetch_usgs, daemon=True, name="usgs-poll"
        ).start()

    def _fetch_usgs(self) -> None:
        """Fetch USGS earthquakes and update tile states (background thread)."""
        try:
            quakes = fetch_fault_corridor_earthquakes(
                period="week", min_mag=1.0
            )
            self._earthquakes = quakes

            # Map to tiles
            tile_quakes = self._mapper.map_earthquakes(quakes)

            # Update tile seismic states (on-fault tiles only)
            scores: Dict[str, float] = {}
            with self._state_lock:
                for tile in self._tiles:
                    if not tile.on_fault:
                        continue
                    tid = tile.tile_id
                    state = self._tile_states[tid]
                    tq = tile_quakes.get(tid, [])

                    state.seismic.events_24h = sum(
                        1 for q in tq if q.age_hours <= 24.0
                    )
                    state.seismic.events_1h = sum(
                        1 for q in tq if q.age_hours <= 1.0
                    )
                    state.seismic.max_mag = max(
                        (q.mag for q in tq), default=0.0
                    )
                    if tq:
                        newest = max(tq, key=lambda q: q.time_ms)
                        state.seismic.latest_mag = newest.mag
                        state.seismic.latest_depth_km = newest.depth_km
                        state.seismic.latest_time_utc = newest.time_utc
                    else:
                        state.seismic.latest_mag = 0.0

                    state.timestamp = time.time()
                    scores[tid] = state.composite_score

            # Emit on main thread
            QtCore.QMetaObject.invokeMethod(
                self, "_emit_usgs",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(object, scores),
                QtCore.Q_ARG(object, quakes),
            )

        except Exception as exc:
            log.error("USGS fetch error: %s", exc)
            QtCore.QMetaObject.invokeMethod(
                self, "_emit_status",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, f"USGS error: {exc}"),
            )

    # ── Geomag + Ionospheric Polling ─────────────────────────────────

    def _poll_geomag(self) -> None:
        threading.Thread(
            target=self._fetch_geomag, daemon=True, name="geomag-poll"
        ).start()

    def _fetch_geomag(self) -> None:
        """Fetch Kp, Dst, TEC and update tile states (background thread)."""
        try:
            kp = get_current_kp()
            dst = get_current_dst()
            self._current_kp = kp
            self._current_dst = dst

            # Classify storm level
            if kp >= 7 or dst <= -200:
                storm = "severe"
            elif kp >= 5 or dst <= -100:
                storm = "moderate"
            elif kp >= 4 or dst <= -50:
                storm = "minor"
            elif kp >= 3 or dst <= -30:
                storm = "unsettled"
            else:
                storm = "quiet"

            # Ionospheric TEC — estimate for fault corridor centroid
            tec = get_tec_for_location(lat=36.0, lon=-120.0, kp=kp)

            # Apply magnetic + ionospheric state only to on-fault tiles
            # (ocean/off-corridor tiles have no relevant data)
            scores: Dict[str, float] = {}
            with self._state_lock:
                for tile in self._tiles:
                    if not tile.on_fault:
                        continue
                    tid = tile.tile_id
                    state = self._tile_states[tid]

                    state.magnetic.kp_index = kp
                    state.magnetic.dst_index = dst
                    state.ionospheric.tec_tecu = tec.tec_tecu
                    state.ionospheric.tec_delta = tec.tec_delta
                    state.timestamp = time.time()
                    scores[tid] = state.composite_score

            # Emit on main thread
            QtCore.QMetaObject.invokeMethod(
                self, "_emit_geomag",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(object, scores),
                QtCore.Q_ARG(object, {
                    "kp": kp, "dst": dst, "storm_level": storm,
                    "tec": tec.tec_tecu, "tec_delta": tec.tec_delta,
                }),
            )

        except Exception as exc:
            log.error("Geomag fetch error: %s", exc)
            QtCore.QMetaObject.invokeMethod(
                self, "_emit_status",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, f"Geomag error: {exc}"),
            )

    # ── GNSS Polling ────────────────────────────────────────────────

    def _poll_gnss(self) -> None:
        threading.Thread(
            target=self._fetch_gnss, daemon=True, name="gnss-poll"
        ).start()

    def _fetch_gnss(self) -> None:
        """Fetch GNSS displacement data and update tile states."""
        try:
            # On Pi, limit to 8 key stations to save bandwidth/CPU
            readings = fetch_gnss_displacements(
                max_stations=8 if _IS_PI else None,
            )
            self._gnss_readings = readings

            if not readings:
                return

            # Map each GNSS station to a tile and update GNSS state
            scores: Dict[str, float] = {}
            # Group readings by tile
            tile_gnss: Dict[str, list] = {}
            for r in readings:
                tid = self._mapper.find_tile(r.lat, r.lon)
                if tid:
                    tile_gnss.setdefault(tid, []).append(r)

            with self._state_lock:
                for tile in self._tiles:
                    if not tile.on_fault:
                        continue
                    tid = tile.tile_id
                    state = self._tile_states[tid]
                    station_readings = tile_gnss.get(tid, [])

                    if station_readings:
                        displacements = [r.displacement_mm_day for r in station_readings]
                        state.gnss.station_count = len(station_readings)
                        sd = sorted(displacements)
                        mid = len(sd) // 2
                        if len(sd) % 2 == 0 and len(sd) > 1:
                            state.gnss.median_mm_day = (sd[mid - 1] + sd[mid]) / 2.0
                        else:
                            state.gnss.median_mm_day = sd[mid]
                        state.gnss.max_mm_day = max(displacements)
                    else:
                        all_displ = [r.displacement_mm_day for r in readings]
                        if all_displ:
                            avg = sum(all_displ) / len(all_displ)
                            state.gnss.median_mm_day = avg
                            state.gnss.max_mm_day = avg
                        state.gnss.station_count = 0

                    state.timestamp = time.time()
                    scores[tid] = state.composite_score

            # Build summary for GUI
            all_rates = [r.displacement_mm_day for r in readings]
            summary = {
                "station_count": len(readings),
                "avg_mm_day": sum(all_rates) / len(all_rates) if all_rates else 0.0,
                "max_mm_day": max(all_rates) if all_rates else 0.0,
                "max_station": max(readings, key=lambda r: r.displacement_mm_day).station_id if readings else "—",
                "readings": readings,
            }

            QtCore.QMetaObject.invokeMethod(
                self, "_emit_gnss",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(object, scores),
                QtCore.Q_ARG(object, summary),
            )

        except Exception as exc:
            log.error("GNSS fetch error: %s", exc)
            QtCore.QMetaObject.invokeMethod(
                self, "_emit_status",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, f"GNSS error: {exc}"),
            )

    # ── Weather Polling ───────────────────────────────────────────────

    def _poll_weather(self) -> None:
        threading.Thread(
            target=self._fetch_weather, daemon=True, name="weather-poll"
        ).start()

    def _fetch_weather(self) -> None:
        """Fetch weather observations and update tile states."""
        try:
            readings = fetch_weather_observations()
            self._weather_readings = readings

            if not readings:
                return

            # Map each weather station to a tile and update state
            scores: Dict[str, float] = {}
            tile_wx: Dict[str, list] = {}
            for r in readings:
                tid = self._mapper.find_tile(r.lat, r.lon)
                if tid:
                    tile_wx.setdefault(tid, []).append(r)

            # Compute corridor-wide averages for tiles without a station
            summary = compute_corridor_summary(readings)
            avg_temp = summary["temperature"]["avg"]
            avg_pressure = summary["pressure"]["avg"]
            avg_wind = summary["wind"]["avg"]
            avg_humidity = summary["humidity"]["avg"]

            with self._state_lock:
                for tile in self._tiles:
                    if not tile.on_fault:
                        continue
                    tid = tile.tile_id
                    state = self._tile_states[tid]
                    station_wx = tile_wx.get(tid, [])

                    if station_wx:
                        w = station_wx[0]
                        state.weather.temperature_c = w.temperature_c if w.temperature_c is not None else avg_temp
                        state.weather.pressure_hpa = w.pressure_hpa if w.pressure_hpa is not None else avg_pressure
                        state.weather.wind_speed_ms = w.wind_speed_ms if w.wind_speed_ms is not None else avg_wind
                        state.weather.humidity_pct = w.humidity_pct if w.humidity_pct is not None else avg_humidity
                    else:
                        state.weather.temperature_c = avg_temp
                        state.weather.pressure_hpa = avg_pressure
                        state.weather.wind_speed_ms = avg_wind
                        state.weather.humidity_pct = avg_humidity

                    state.timestamp = time.time()
                    scores[tid] = state.composite_score

            summary["readings"] = readings

            QtCore.QMetaObject.invokeMethod(
                self, "_emit_weather",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(object, scores),
                QtCore.Q_ARG(object, summary),
            )

        except Exception as exc:
            log.error("Weather fetch error: %s", exc)
            QtCore.QMetaObject.invokeMethod(
                self, "_emit_status",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, f"Weather error: {exc}"),
            )

    # ── News Polling ─────────────────────────────────────────────────

    def _emit_news_data(self, articles: List[ICEArticle]) -> None:
        """Map articles to tiles and emit the news_updated signal."""
        news_by_tile: Dict[str, List[ICEArticle]] = {}
        for art in articles:
            if art.lat and art.lon:
                tid = self._mapper.find_tile(art.lat, art.lon)
                if tid:
                    news_by_tile.setdefault(tid, []).append(art)

        self._news_by_tile = news_by_tile

        total = len(articles)
        with_img = sum(1 for a in articles if a.local_image_path)

        QtCore.QMetaObject.invokeMethod(
            self, "_emit_news",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(object, news_by_tile),
            QtCore.Q_ARG(object, {
                "total": total,
                "with_images": with_img,
                "mapped_to_tiles": sum(len(v) for v in news_by_tile.values()),
                "tile_count": len(news_by_tile),
            }),
        )

    def _poll_news(self) -> None:
        threading.Thread(
            target=self._fetch_news, daemon=True, name="news-poll"
        ).start()

    def _fetch_news(self) -> None:
        """Fetch ICE news from GDELT and map to tiles.

        Every startup cycle:
          1. Load cached archive → emit immediately so map has data ASAP.
          2. Fetch recent 7-day articles, download images, save, re-emit.
          3. Run historical backfill for any missing weeks (Jan 20 2025→today).
             Backfill resumes from where it left off and picks up newly
             elapsed weeks automatically.  After backfill, re-emit full archive.
        """
        try:
            # ── Step 1: Show cached data instantly ──────────────────────
            cached = load_articles()
            if cached:
                self._news_articles = cached
                self._emit_news_data(cached)
                log.info("News: emitted %d cached articles on startup", len(cached))

            # ── Step 2: Fetch recent (last 7 days) ─────────────────────
            recent = fetch_ice_news(max_records=250)
            if recent:
                download_all_images(recent, max_workers=4)
                save_articles(recent)

                # Re-load full archive (cached + recent merged)
                articles = load_articles()
                if articles:
                    download_all_images(articles, max_workers=4)
                    save_articles(articles)
                    self._news_articles = articles
                    self._emit_news_data(articles)
                    log.info("News: emitted %d articles after recent fetch",
                             len(articles))

                # ── Step 3: Historical backfill (only if recent succeeded) ─
                if not is_backfill_complete():
                    log.info("News: backfilling history from 2025-01-20…")
                    new_articles = backfill_historical()
                    if new_articles:
                        download_all_images(new_articles, max_workers=4)
                        save_articles(new_articles)
                        articles = load_articles()
                        if articles:
                            self._news_articles = articles
                            self._emit_news_data(articles)
                            log.info("News: emitted %d articles after backfill",
                                     len(articles))
                else:
                    log.info("News: historical backfill already complete")
            else:
                log.info("News: no recent articles (rate-limited or empty), "
                         "skipping backfill")

        except Exception as exc:
            log.error("News fetch error: %s", exc)
            QtCore.QMetaObject.invokeMethod(
                self, "_emit_status",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, f"News error: {exc}"),
            )

    # ── Social data (Census + Native Land + SCEDC) ─────────────────

    def _poll_social(self) -> None:
        threading.Thread(
            target=self._fetch_social, daemon=True, name="social-poll"
        ).start()

    def _fetch_social(self) -> None:
        """Fetch social data: Census demographics, indigenous territories, SCEDC."""
        try:
            # Census ACS demographics
            census_data = fetch_census_data()

            # Indigenous territories (static, instant)
            territories = get_saf_territories()

            # SCEDC seismic data (complementary to USGS)
            scedc_events = fetch_scedc_events(days=7, min_mag=2.0)

            social_payload = {
                "census": census_data,
                "territories": territories,
                "scedc": scedc_events,
            }

            QtCore.QMetaObject.invokeMethod(
                self, "_emit_social",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(object, social_payload),
            )

        except Exception as exc:
            log.error("Social data fetch error: %s", exc)

    # ── Signal Emitters (main thread) ────────────────────────────────

    @QtCore.pyqtSlot(object, object)
    def _emit_usgs(self, scores: dict, quakes: list) -> None:
        self.tiles_updated.emit(scores)
        self.earthquakes_updated.emit(quakes)

        total = len(quakes)
        recent = sum(1 for q in quakes if q.age_hours <= 24.0)
        self.status_message.emit(
            f"USGS: {total} earthquakes (week), {recent} in 24h"
        )

    @QtCore.pyqtSlot(object, object)
    def _emit_geomag(self, scores: dict, info: dict) -> None:
        self.tiles_updated.emit(scores)
        self.geomag_updated.emit(info)
        self.status_message.emit(
            f"Geomag: Kp={info['kp']:.1f}, Dst={info['dst']:.0f}nT, "
            f"TEC={info['tec']:.1f}TECU ({info['storm_level']})"
        )

    @QtCore.pyqtSlot(object, object)
    def _emit_gnss(self, scores: dict, summary: dict) -> None:
        self.tiles_updated.emit(scores)
        self.gnss_updated.emit(summary)
        self.status_message.emit(
            f"GNSS: {summary['station_count']} stations, "
            f"avg {summary['avg_mm_day']:.3f} mm/day, "
            f"max {summary['max_mm_day']:.3f} mm/day ({summary['max_station']})"
        )

    @QtCore.pyqtSlot(object, object)
    def _emit_weather(self, scores: dict, summary: dict) -> None:
        self.tiles_updated.emit(scores)
        self.weather_updated.emit(summary)
        sc = summary.get("station_count", 0)
        t = summary.get("temperature", {}).get("avg", 0)
        w = summary.get("wind", {}).get("avg", 0)
        p = summary.get("pressure", {}).get("avg", 0)
        self.status_message.emit(
            f"Weather: {sc} stations, {t:.1f}°C, {w:.1f}m/s wind, {p:.0f}hPa"
        )

    @QtCore.pyqtSlot(object, object)
    def _emit_news(self, news_by_tile: dict, summary: dict) -> None:
        self.news_updated.emit(news_by_tile)
        total = summary.get("total", 0)
        with_img = summary.get("with_images", 0)
        tiles = summary.get("tile_count", 0)
        self.status_message.emit(
            f"News: {total} articles, {with_img} images, {tiles} tiles"
        )

    @QtCore.pyqtSlot(object)
    def _emit_social(self, payload: dict) -> None:
        self.social_updated.emit(payload)
        census = payload.get("census", [])
        terr = payload.get("territories", [])
        scedc = payload.get("scedc", [])
        self.status_message.emit(
            f"Social: {len(census)} counties, {len(terr)} territories, "
            f"{len(scedc)} SCEDC events"
        )

    @QtCore.pyqtSlot(str)
    def _emit_status(self, msg: str) -> None:
        self.status_message.emit(msg)
