"""
GNSS crustal deformation client.

Fetches daily GPS displacement data from the Nevada Geodetic Laboratory (UNR)
for stations along the San Andreas Fault corridor.  The data provides
millimetre-scale measurements of crustal motion — a key indicator of tectonic
strain accumulation.

Data source
-----------
  University of Nevada, Reno — Nevada Geodetic Laboratory
  https://geodesy.unr.edu/

  Format: tenv3 (daily position time series in IGS14 reference frame)
  URL pattern: https://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/{STATION}.tenv3

  Stations are selected along the San Andreas Fault corridor in California.
  Only the most recent 7 days of data are analysed to compute current
  displacement rates (mm/day).

Usage
-----
    stations = get_fault_corridor_stations()
    readings = fetch_gnss_displacements(stations)
    # → list of GNSSReading with station_id, lat, lon, displacement_mm_day, …
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import requests

log = logging.getLogger(__name__)

# ── San Andreas Fault corridor GNSS stations ───────────────────────────
# Curated list of PBO/NOTA and other permanent GNSS stations within
# ~50 km of the San Andreas Fault, ordered roughly north → south.
# Each entry: (station_id, approx_lat, approx_lon, description)

_SAF_STATIONS = [
    # Northern California
    ("P198", 40.84, -124.08, "Cape Mendocino"),
    ("P162", 40.33, -124.35, "Shelter Cove"),
    ("P159", 39.75, -123.24, "Willits"),
    ("LUTZ", 39.42, -123.01, "Upper Lake"),
    ("P196", 38.84, -122.81, "Cloverdale"),
    ("SVIN", 38.49, -122.82, "Santa Rosa"),
    ("P200", 38.25, -122.57, "Petaluma"),
    # San Francisco Bay Area
    ("TIBB", 37.89, -122.45, "Tiburon"),
    ("P224", 37.76, -122.43, "San Francisco"),
    ("P225", 37.60, -122.39, "Daly City"),
    ("P229", 37.47, -122.17, "Hayward"),
    ("P231", 37.25, -121.88, "San Jose"),
    # Central California
    ("P247", 36.97, -121.57, "Hollister"),
    ("P250", 36.61, -121.20, "Pinnacles"),
    ("P254", 36.25, -120.79, "Coalinga"),
    ("P261", 35.95, -120.50, "Parkfield"),
    ("P263", 35.75, -120.28, "Cholame"),
    ("CARH", 35.53, -120.04, "Carrizo Plain"),
    # Southern California — Transverse Ranges
    ("P504", 35.04, -119.50, "Maricopa"),
    ("VNDN", 34.55, -120.62, "Vandenberg"),
    ("P503", 34.45, -119.73, "Santa Barbara"),
    ("P502", 34.30, -118.95, "Ventura"),
    ("AZU1", 34.13, -117.91, "Azusa"),
    # San Bernardino / Inland Empire
    ("P487", 34.08, -117.25, "San Bernardino"),
    ("P486", 33.90, -116.90, "Banning"),
    ("P497", 33.72, -116.17, "Palm Springs"),
    # Salton Trough / Imperial Valley
    ("P500", 33.25, -115.98, "Salton Sea"),
    ("P494", 33.05, -115.57, "Calipatria"),
    ("P496", 32.89, -115.52, "El Centro"),
    ("IID2", 32.72, -115.50, "Imperial"),
]


@dataclass
class GNSSReading:
    """One displacement reading from a GNSS station."""
    station_id: str
    lat: float
    lon: float
    description: str
    displacement_mm_day: float    # total 3D displacement rate (mm/day)
    north_mm_day: float           # north component rate
    east_mm_day: float            # east component rate
    vertical_mm_day: float        # vertical component rate
    days_of_data: int             # how many recent days were used
    last_epoch: str               # most recent data epoch (YYYY-MM-DD)


def get_fault_corridor_stations() -> List[tuple]:
    """Return the curated list of GNSS stations along the fault.

    Returns list of (station_id, lat, lon, description) tuples.
    """
    return list(_SAF_STATIONS)


def _parse_tenv3_recent(text: str, days: int = 7) -> List[dict]:
    """Parse the most recent N days from a tenv3 time series file.

    tenv3 columns (space-separated):
      0: station  1: date(YYMMMDD)  2: decyr  3-5: dN,dE,dU (m)
      6-8: Sn,Se,Su (sigma in m)  9: corr_ne  10: corr_nu  11: corr_eu
      12: lat  13: lon  14: height

    We only need columns 0, 2 (decimal year), 3-5 (displacements), 12-13 (coords).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_decyr = cutoff.year + (cutoff.timetuple().tm_yday / 365.25)

    rows = []
    for line in text.strip().split("\n"):
        parts = line.split()
        if len(parts) < 15:
            continue
        try:
            decyr = float(parts[2])
            if decyr < cutoff_decyr:
                continue
            rows.append({
                "decyr": decyr,
                "dN": float(parts[3]),   # north offset (m)
                "dE": float(parts[4]),   # east offset (m)
                "dU": float(parts[5]),   # vertical offset (m)
                "lat": float(parts[12]),
                "lon": float(parts[13]),
            })
        except (ValueError, IndexError):
            continue
    return rows


def _compute_rate(rows: List[dict]) -> Optional[dict]:
    """Compute displacement rate from a list of daily position rows.

    Uses simple linear difference between first and last epoch.
    Returns mm/day rates for N, E, U components.
    """
    if len(rows) < 2:
        return None

    first = rows[0]
    last = rows[-1]
    dt_days = (last["decyr"] - first["decyr"]) * 365.25
    if dt_days < 0.5:
        return None

    dN = (last["dN"] - first["dN"]) * 1000.0 / dt_days  # m→mm, per day
    dE = (last["dE"] - first["dE"]) * 1000.0 / dt_days
    dU = (last["dU"] - first["dU"]) * 1000.0 / dt_days
    total = math.sqrt(dN**2 + dE**2 + dU**2)

    return {
        "north_mm_day": dN,
        "east_mm_day": dE,
        "vertical_mm_day": dU,
        "displacement_mm_day": total,
        "days": len(rows),
        "lat": last["lat"],
        "lon": last["lon"],
    }


def fetch_station_displacement(
    station_id: str,
    lat: float,
    lon: float,
    description: str,
    days: int = 7,
    timeout: float = 15.0,
) -> Optional[GNSSReading]:
    """Fetch recent displacement data for a single GNSS station.

    Downloads the tenv3 time series from UNR, parses the most recent
    `days` of data, and computes a displacement rate.

    Returns None if the station data is unavailable or insufficient.
    """
    url = (
        f"https://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/{station_id}.tenv3"
    )
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            log.debug("GNSS station %s: HTTP %d", station_id, resp.status_code)
            return None

        rows = _parse_tenv3_recent(resp.text, days=days)
        rate = _compute_rate(rows)
        if rate is None:
            log.debug("GNSS station %s: insufficient data (%d rows)", station_id, len(rows))
            return None

        # Compute last epoch as approximate date
        last_decyr = rows[-1]["decyr"]
        year = int(last_decyr)
        day_of_year = int((last_decyr - year) * 365.25)
        try:
            last_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
            last_epoch_str = last_date.strftime("%Y-%m-%d")
        except (ValueError, OverflowError):
            last_epoch_str = f"{last_decyr:.3f}"

        return GNSSReading(
            station_id=station_id,
            lat=rate["lat"],
            lon=rate["lon"],
            description=description,
            displacement_mm_day=rate["displacement_mm_day"],
            north_mm_day=rate["north_mm_day"],
            east_mm_day=rate["east_mm_day"],
            vertical_mm_day=rate["vertical_mm_day"],
            days_of_data=rate["days"],
            last_epoch=last_epoch_str,
        )

    except requests.RequestException as exc:
        log.debug("GNSS station %s: network error: %s", station_id, exc)
        return None


def fetch_gnss_displacements(
    stations: Optional[List[tuple]] = None,
    days: int = 30,
    timeout: float = 10.0,
    max_stations: Optional[int] = None,
) -> List[GNSSReading]:
    """Fetch displacement data for all specified GNSS stations.

    Parameters
    ----------
    stations : list of (station_id, lat, lon, description) tuples
        If None, uses the default SAF corridor station list.
    days : int
        Number of recent days to analyze (default 30).  UNR data can
        lag weeks behind real-time, so 30 days captures more stations.
    timeout : float
        HTTP request timeout per station.
    max_stations : int, optional
        Limit the number of stations to fetch (useful on Pi to save
        bandwidth).  None = all stations.

    Returns
    -------
    list[GNSSReading]
        Successfully fetched displacement readings.
    """
    if stations is None:
        stations = _SAF_STATIONS

    if max_stations is not None:
        stations = stations[:max_stations]

    readings: List[GNSSReading] = []
    for sid, lat, lon, desc in stations:
        reading = fetch_station_displacement(sid, lat, lon, desc, days, timeout)
        if reading is not None:
            readings.append(reading)

    log.info(
        "GNSS: fetched %d/%d station displacements (%d-day rate)",
        len(readings), len(stations), days,
    )
    return readings
