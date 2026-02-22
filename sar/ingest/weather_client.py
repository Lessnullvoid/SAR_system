"""
NOAA / NWS weather observation client.

Fetches current atmospheric conditions from National Weather Service stations
along the San Andreas Fault corridor.  Weather data provides environmental
context for other sensor readings — temperature inversions, pressure changes,
and high winds can all influence RF propagation and instrument readings.

Data source
-----------
  NOAA National Weather Service API
  https://api.weather.gov

  No API key required.  Returns JSON.  Rate limit: be reasonable (~1 req/s).

  Endpoint chain:
    1. /points/{lat},{lon}         → get nearest observation station
    2. /stations/{id}/observations/latest → current conditions

Usage
-----
    stations = get_fault_corridor_weather_stations()
    readings = fetch_weather_observations(stations)
    # → list of WeatherReading with temperature, pressure, wind, humidity
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

log = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "(SAR-System, research-project)",
    "Accept": "application/geo+json",
}

# ── Weather stations near the San Andreas Fault corridor ───────────────
# NWS observation station IDs along the fault from north to south.
# These are ICAO/NWS identifiers for airports and weather stations
# that provide current observation data.

_SAF_WEATHER_STATIONS = [
    # Station ID, lat, lon, name, region
    ("KEKA", 40.80, -124.16, "Eureka/Arcata", "Northern CA"),
    ("KUKI", 39.13, -123.20, "Ukiah", "Northern CA"),
    ("KSTS", 38.51, -122.81, "Santa Rosa", "North Bay"),
    ("KSFO", 37.62, -122.36, "San Francisco Intl", "Bay Area"),
    ("KSJC", 37.36, -121.93, "San Jose", "Bay Area"),
    ("KHOL", 36.90, -121.41, "Hollister", "Central CA"),
    ("KSNS", 36.66, -121.61, "Salinas", "Central CA"),
    ("KPRB", 35.67, -120.63, "Paso Robles", "Central CA"),
    ("KSBP", 35.24, -120.64, "San Luis Obispo", "Central Coast"),
    ("KVBG", 34.73, -120.57, "Vandenberg AFB", "Central Coast"),
    ("KSBA", 34.43, -119.84, "Santa Barbara", "South Coast"),
    ("KBUR", 34.20, -118.36, "Burbank", "LA Basin"),
    ("KLAX", 34.05, -118.25, "Los Angeles Intl", "LA Basin"),
    ("KONT", 34.06, -117.60, "Ontario", "Inland Empire"),
    ("KSBD", 34.10, -117.24, "San Bernardino", "Inland Empire"),
    ("KPSP", 33.83, -116.51, "Palm Springs", "Coachella Valley"),
    ("KTRM", 33.63, -116.16, "Thermal", "Coachella Valley"),
    ("KIPL", 32.83, -115.57, "Imperial", "Imperial Valley"),
    ("KSDM", 32.57, -117.12, "San Diego / Brown", "San Diego"),
]


@dataclass
class WeatherReading:
    """Current weather observation from one station."""
    station_id: str
    station_name: str
    region: str
    lat: float
    lon: float
    temperature_c: Optional[float] = None
    pressure_hpa: Optional[float] = None
    wind_speed_ms: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    humidity_pct: Optional[float] = None
    visibility_m: Optional[float] = None
    description: str = ""
    timestamp: str = ""


def get_fault_corridor_weather_stations() -> List[tuple]:
    """Return curated list of weather stations along the fault."""
    return list(_SAF_WEATHER_STATIONS)


def fetch_station_observation(
    station_id: str,
    lat: float,
    lon: float,
    name: str,
    region: str,
    timeout: float = 12.0,
) -> Optional[WeatherReading]:
    """Fetch the latest weather observation for one NWS station.

    Uses the NWS API: /stations/{id}/observations/latest

    Returns None if the station data is unavailable.
    """
    url = f"https://api.weather.gov/stations/{station_id}/observations/latest"

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        if resp.status_code != 200:
            log.debug("Weather station %s: HTTP %d", station_id, resp.status_code)
            return None

        data = resp.json()
        props = data.get("properties", {})

        # Extract values — NWS uses {"value": X, "unitCode": "..."} format
        def _val(field_name: str) -> Optional[float]:
            obj = props.get(field_name, {})
            if isinstance(obj, dict):
                v = obj.get("value")
                return float(v) if v is not None else None
            return None

        temp_c = _val("temperature")
        pressure_pa = _val("barometricPressure")
        wind_ms = _val("windSpeed")
        wind_dir = _val("windDirection")
        humidity = _val("relativeHumidity")
        visibility = _val("visibility")

        # Convert pressure from Pa → hPa
        pressure_hpa = pressure_pa / 100.0 if pressure_pa is not None else None

        # Convert wind from km/h to m/s (NWS returns km/h in some cases)
        # Actually NWS API returns m/s with unitCode "wmoUnit:m_s-1"
        # but some stations report km/h — check unitCode
        wind_unit = props.get("windSpeed", {}).get("unitCode", "")
        if wind_ms is not None and "km" in wind_unit.lower():
            wind_ms = wind_ms / 3.6

        description = props.get("textDescription", "")
        timestamp = props.get("timestamp", "")

        return WeatherReading(
            station_id=station_id,
            station_name=name,
            region=region,
            lat=lat,
            lon=lon,
            temperature_c=temp_c,
            pressure_hpa=pressure_hpa,
            wind_speed_ms=wind_ms,
            wind_direction_deg=wind_dir,
            humidity_pct=humidity,
            visibility_m=visibility,
            description=description,
            timestamp=timestamp,
        )

    except requests.RequestException as exc:
        log.debug("Weather station %s: network error: %s", station_id, exc)
        return None
    except (ValueError, KeyError, TypeError) as exc:
        log.debug("Weather station %s: parse error: %s", station_id, exc)
        return None


def fetch_weather_observations(
    stations: Optional[List[tuple]] = None,
    timeout: float = 12.0,
) -> List[WeatherReading]:
    """Fetch current weather from all specified stations.

    Parameters
    ----------
    stations : list of (station_id, lat, lon, name, region) tuples
        If None, uses the default SAF corridor station list.
    timeout : float
        HTTP request timeout per station.

    Returns
    -------
    list[WeatherReading]
        Successfully fetched weather readings.
    """
    if stations is None:
        stations = _SAF_WEATHER_STATIONS

    readings: List[WeatherReading] = []
    for sid, lat, lon, name, region in stations:
        reading = fetch_station_observation(sid, lat, lon, name, region, timeout)
        if reading is not None:
            readings.append(reading)

    log.info(
        "Weather: fetched %d/%d station observations",
        len(readings), len(stations),
    )
    return readings


def compute_corridor_summary(readings: List[WeatherReading]) -> Dict:
    """Compute summary statistics across all stations.

    Returns dict with avg/min/max for key parameters.
    """
    temps = [r.temperature_c for r in readings if r.temperature_c is not None]
    pressures = [r.pressure_hpa for r in readings if r.pressure_hpa is not None]
    winds = [r.wind_speed_ms for r in readings if r.wind_speed_ms is not None]
    humidities = [r.humidity_pct for r in readings if r.humidity_pct is not None]

    def _stats(values: list) -> Dict:
        if not values:
            return {"avg": 0.0, "min": 0.0, "max": 0.0}
        return {
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    return {
        "station_count": len(readings),
        "temperature": _stats(temps),
        "pressure": _stats(pressures),
        "wind": _stats(winds),
        "humidity": _stats(humidities),
    }
