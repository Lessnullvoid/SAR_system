"""
Tile state data model.

Each tile accumulates sensor readings from multiple sources.  A TileState
snapshot represents the current knowledge about one tile at a point in time.

Example
-------
    state = TileState(tile_id="SAF_023")
    state.seismic.events_24h = 11
    state.seismic.max_mag = 3.2
    state.rf.anomaly_score = 0.71
    state.composite_score  # auto-computed
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SeismicState:
    """Seismic sensor readings for a tile."""
    events_24h: int = 0
    events_1h: int = 0
    max_mag: float = 0.0
    latest_mag: float = 0.0
    latest_depth_km: float = 0.0
    latest_time_utc: str = ""


@dataclass
class GNSSState:
    """GNSS / crustal deformation readings."""
    median_mm_day: float = 0.0
    max_mm_day: float = 0.0
    station_count: int = 0


@dataclass
class WeatherState:
    """Atmospheric context."""
    temperature_c: float = 0.0
    pressure_hpa: float = 0.0
    wind_speed_ms: float = 0.0
    humidity_pct: float = 0.0


@dataclass
class MagneticState:
    """Geomagnetic field state."""
    kp_index: float = 0.0
    local_variation_nt: float = 0.0
    dst_index: float = 0.0


@dataclass
class RFState:
    """RF anomaly state (from the SDR/ML subsystem)."""
    anomaly_score: float = 0.0
    active_bands: List[str] = field(default_factory=list)
    peak_freq_mhz: float = 0.0
    triggered_features: List[str] = field(default_factory=list)


@dataclass
class IonosphericState:
    """Ionospheric TEC readings."""
    tec_tecu: float = 0.0       # Total Electron Content (TECU)
    tec_delta: float = 0.0      # deviation from median


@dataclass
class TileState:
    """Complete sensor state for one tile at one point in time.

    The ``composite_score`` property aggregates all sensor anomaly
    indicators into a single 0–1 value that drives the map visualization
    and sonification.
    """
    tile_id: str = ""
    timestamp: float = field(default_factory=time.time)
    time_utc: str = ""

    seismic: SeismicState = field(default_factory=SeismicState)
    gnss: GNSSState = field(default_factory=GNSSState)
    weather: WeatherState = field(default_factory=WeatherState)
    magnetic: MagneticState = field(default_factory=MagneticState)
    rf: RFState = field(default_factory=RFState)
    ionospheric: IonosphericState = field(default_factory=IonosphericState)

    @property
    def composite_score(self) -> float:
        """Weighted composite anomaly score (0 = quiet, 1 = max anomaly).

        Weights reflect scientific relevance to seismo-EM precursors:
          - Seismic activity:    0.30  (direct indicator)
          - RF anomalies:        0.25  (EM precursors)
          - Geomagnetic:         0.15  (field disturbances)
          - Ionospheric TEC:     0.15  (ionospheric perturbation)
          - GNSS deformation:    0.10  (crustal strain)
          - Weather:             0.05  (environmental context)
        """
        # Weather anomaly: high wind (>15 m/s) or rapid pressure change
        # contribute a small context score
        wx_score = 0.0
        if self.weather.wind_speed_ms > 5.0:
            wx_score = max(wx_score, min(self.weather.wind_speed_ms / 25.0, 1.0))

        scores = {
            "seismic": min(self.seismic.events_24h / 50.0, 1.0),
            "rf": self.rf.anomaly_score,
            "magnetic": min(self.magnetic.kp_index / 9.0, 1.0),
            "ionospheric": min(abs(self.ionospheric.tec_delta) / 20.0, 1.0),
            "gnss": min(self.gnss.max_mm_day / 5.0, 1.0),
            "weather": wx_score,
        }
        weights = {
            "seismic": 0.30,
            "rf": 0.25,
            "magnetic": 0.15,
            "ionospheric": 0.15,
            "gnss": 0.10,
            "weather": 0.05,
        }
        return sum(scores[k] * weights[k] for k in scores)

    def as_dict(self) -> Dict:
        """Serialize to a flat dict (for storage / OSC / JSON)."""
        return {
            "tile_id": self.tile_id,
            "timestamp": self.timestamp,
            "time_utc": self.time_utc,
            "composite_score": round(self.composite_score, 4),
            "seismic_events_24h": self.seismic.events_24h,
            "seismic_max_mag": self.seismic.max_mag,
            "gnss_mm_day": self.gnss.median_mm_day,
            "weather_wind": self.weather.wind_speed_ms,
            "magnetic_kp": self.magnetic.kp_index,
            "rf_anomaly": self.rf.anomaly_score,
            "tec_tecu": self.ionospheric.tec_tecu,
        }
