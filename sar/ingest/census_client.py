"""
U.S. Census / ACS demographic data client.

Fetches county-level demographic and migration-proxy data from the Census
Bureau's ACS 5-Year API for California counties along the San Andreas Fault.

Data source
-----------
  U.S. Census Bureau — American Community Survey 5-Year Estimates
  https://api.census.gov/data/2022/acs/acs5

  No API key required for <500 queries/day.  Returns JSON arrays.

Variables fetched
-----------------
  B01003_001E — Total population
  B07001_001E — Geographic mobility (total)
  B07001_017E — Moved from different state
  B07001_033E — Moved from abroad
  B19013_001E — Median household income
  B25001_001E — Total housing units
  B03003_003E — Hispanic or Latino population

Usage
-----
    data = fetch_census_data()
    # → list of CountyDemographics for SAF corridor counties
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# ACS 5-Year (2022 is latest stable)
_BASE_URL = "https://api.census.gov/data/2022/acs/acs5"

# California FIPS = 06
# Counties along / near the San Andreas Fault corridor
_SAF_COUNTIES: Dict[str, str] = {
    "037": "Los Angeles",
    "071": "San Bernardino",
    "065": "Riverside",
    "029": "Kern",
    "019": "Fresno",
    "107": "Tulare",
    "069": "San Benito",
    "053": "Monterey",
    "085": "Santa Clara",
    "001": "Alameda",
    "075": "San Francisco",
    "013": "Contra Costa",
    "097": "Sonoma",
    "045": "Mendocino",
    "023": "Humboldt",
    "079": "San Luis Obispo",
    "083": "Santa Barbara",
    "111": "Ventura",
    "059": "Orange",
    "025": "Imperial",
}

# Representative centroid (lat, lon) per county for tile mapping
_COUNTY_CENTROIDS: Dict[str, tuple] = {
    "037": (34.05, -118.24),   # Los Angeles
    "071": (34.85, -116.18),   # San Bernardino
    "065": (33.74, -115.99),   # Riverside
    "029": (35.35, -118.73),   # Kern
    "019": (36.75, -119.65),   # Fresno
    "107": (36.23, -118.78),   # Tulare
    "069": (36.62, -121.08),   # San Benito
    "053": (36.24, -121.31),   # Monterey
    "085": (37.23, -121.70),   # Santa Clara
    "001": (37.65, -121.92),   # Alameda
    "075": (37.76, -122.44),   # San Francisco
    "013": (37.92, -121.95),   # Contra Costa
    "097": (38.52, -122.94),   # Sonoma
    "045": (39.44, -123.39),   # Mendocino
    "023": (40.74, -123.87),   # Humboldt
    "079": (35.38, -120.44),   # San Luis Obispo
    "083": (34.74, -120.02),   # Santa Barbara
    "111": (34.36, -119.13),   # Ventura
    "059": (33.72, -117.78),   # Orange
    "025": (33.04, -115.36),   # Imperial
}

_VARIABLES = [
    "NAME",
    "B01003_001E",   # Total population
    "B07001_001E",   # Geographic mobility total
    "B07001_017E",   # Moved from different state
    "B07001_033E",   # Moved from abroad
    "B19013_001E",   # Median household income
    "B25001_001E",   # Housing units
    "B03003_003E",   # Hispanic/Latino population
]


@dataclass
class CountyDemographics:
    """Demographic profile for one county."""
    fips: str
    name: str
    lat: float = 0.0
    lon: float = 0.0
    population: int = 0
    mobility_total: int = 0
    moved_from_state: int = 0
    moved_from_abroad: int = 0
    median_income: int = 0
    housing_units: int = 0
    hispanic_pop: int = 0
    migration_pct: float = 0.0  # % of population that migrated recently


def fetch_census_data() -> List[CountyDemographics]:
    """Fetch ACS 5-Year demographic data for SAF corridor counties.

    Returns list of CountyDemographics. No API key needed (<500 req/day).
    """
    import requests

    county_fips = ",".join(_SAF_COUNTIES.keys())
    params = {
        "get": ",".join(_VARIABLES),
        "for": f"county:{county_fips}",
        "in": "state:06",
    }

    try:
        resp = requests.get(_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
    except Exception as exc:
        log.warning("Census API error: %s", exc)
        return []

    if not rows or len(rows) < 2:
        log.warning("Census: empty response")
        return []

    header = rows[0]
    results: List[CountyDemographics] = []

    for row in rows[1:]:
        d = dict(zip(header, row))
        fips = d.get("county", "")
        if fips not in _SAF_COUNTIES:
            continue

        def _int(key: str) -> int:
            try:
                return int(d.get(key, 0) or 0)
            except (ValueError, TypeError):
                return 0

        pop = _int("B01003_001E")
        mob_total = _int("B07001_001E")
        from_state = _int("B07001_017E")
        from_abroad = _int("B07001_033E")
        income = _int("B19013_001E")
        housing = _int("B25001_001E")
        hisp = _int("B03003_003E")

        migrated = from_state + from_abroad
        mig_pct = (migrated / pop * 100) if pop > 0 else 0.0

        lat, lon = _COUNTY_CENTROIDS.get(fips, (0.0, 0.0))

        results.append(CountyDemographics(
            fips=fips,
            name=d.get("NAME", _SAF_COUNTIES.get(fips, "")),
            lat=lat, lon=lon,
            population=pop,
            mobility_total=mob_total,
            moved_from_state=from_state,
            moved_from_abroad=from_abroad,
            median_income=income,
            housing_units=housing,
            hispanic_pop=hisp,
            migration_pct=round(mig_pct, 1),
        ))

    log.info("Census: fetched demographics for %d counties", len(results))
    return results
