"""
Native Land Digital — Indigenous territories along the San Andreas Fault.

Provides static data for the indigenous peoples whose territories intersect
the San Andreas Fault corridor in California.  This data is sourced from
Native Land Digital (https://native-land.ca) under CC BY-NC-SA license.

The territories are stored as static records because:
1. The API requires a key (free signup at native-land.ca).
2. Territory boundaries change rarely.
3. This ensures the app works offline for Raspberry Pi deployments.

Attribution
-----------
  Data from Native Land Digital — https://native-land.ca
  License: CC BY-NC-SA 4.0
  "This map does not represent or intend to represent official or legal
   boundaries of any Indigenous nation."

Usage
-----
    territories = get_saf_territories()
    # → list of IndigenousTerritory with name, people, lat/lon coverage
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

log = logging.getLogger(__name__)


@dataclass
class IndigenousTerritory:
    """One indigenous territory along the SAF corridor."""
    name: str               # Territory name
    people: str             # People / nation name
    region: str             # "southern", "central", "northern"
    centroid_lat: float     # Representative latitude
    centroid_lon: float     # Representative longitude
    bbox: Tuple[float, float, float, float]  # (min_lat, min_lon, max_lat, max_lon)
    description: str = ""   # Brief context
    url: str = ""           # Link to native-land.ca page


# ── Static territory data for California SAF corridor ──────────────
# Compiled from Native Land Digital public map data.
# Each entry represents a major indigenous territory that overlaps
# with the San Andreas Fault corridor.

_TERRITORIES = [
    # Southern section (Salton Sea → San Bernardino)
    IndigenousTerritory(
        name="Cahuilla",
        people="Cahuilla Band of Indians",
        region="southern",
        centroid_lat=33.75, centroid_lon=-116.40,
        bbox=(33.2, -117.0, 34.3, -115.8),
        description="Desert Cahuilla homelands in Coachella Valley and surrounding mountains",
        url="https://native-land.ca/maps/territories/cahuilla",
    ),
    IndigenousTerritory(
        name="Serrano",
        people="Serrano / Yuhaviatam",
        region="southern",
        centroid_lat=34.20, centroid_lon=-117.10,
        bbox=(33.8, -117.6, 34.6, -116.5),
        description="San Bernardino Mountains and Mojave Desert edge",
        url="https://native-land.ca/maps/territories/serrano",
    ),
    IndigenousTerritory(
        name="Tongva",
        people="Tongva / Gabrielino",
        region="southern",
        centroid_lat=34.00, centroid_lon=-118.20,
        bbox=(33.5, -118.8, 34.5, -117.5),
        description="Greater Los Angeles Basin — one of the most populated indigenous territories in California",
        url="https://native-land.ca/maps/territories/tongva",
    ),
    IndigenousTerritory(
        name="Kumeyaay",
        people="Kumeyaay / Diegueño",
        region="southern",
        centroid_lat=32.85, centroid_lon=-116.50,
        bbox=(32.3, -117.3, 33.4, -115.5),
        description="Southern border region, Anza-Borrego, and Salton Sea area",
        url="https://native-land.ca/maps/territories/kumeyaay",
    ),
    # Central section (Tehachapi → Parkfield → Hollister)
    IndigenousTerritory(
        name="Yokuts",
        people="Yokuts / Yokoch",
        region="central",
        centroid_lat=36.00, centroid_lon=-119.50,
        bbox=(34.8, -120.5, 37.2, -118.5),
        description="Central Valley — largest territory along the fault, agricultural heartland",
        url="https://native-land.ca/maps/territories/yokuts",
    ),
    IndigenousTerritory(
        name="Chumash",
        people="Chumash",
        region="central",
        centroid_lat=34.60, centroid_lon=-119.80,
        bbox=(34.0, -120.8, 35.2, -118.8),
        description="Coastal and inland areas from Santa Barbara to Ventura and San Luis Obispo",
        url="https://native-land.ca/maps/territories/chumash",
    ),
    IndigenousTerritory(
        name="Salinan",
        people="Salinan / T'epot'aha'l",
        region="central",
        centroid_lat=35.90, centroid_lon=-121.00,
        bbox=(35.3, -121.5, 36.5, -120.5),
        description="Salinas Valley and coastal ranges near Parkfield segment",
        url="https://native-land.ca/maps/territories/salinan",
    ),
    IndigenousTerritory(
        name="Esselen",
        people="Esselen Nation",
        region="central",
        centroid_lat=36.30, centroid_lon=-121.60,
        bbox=(36.0, -122.0, 36.6, -121.2),
        description="Big Sur coast and Carmel Valley",
        url="https://native-land.ca/maps/territories/esselen",
    ),
    # Northern section (Santa Cruz → San Francisco → Ukiah → Eureka)
    IndigenousTerritory(
        name="Ohlone",
        people="Ohlone / Costanoan",
        region="northern",
        centroid_lat=37.40, centroid_lon=-122.00,
        bbox=(36.8, -122.5, 38.0, -121.3),
        description="San Francisco Bay Area — major urban indigenous territory",
        url="https://native-land.ca/maps/territories/ohlone",
    ),
    IndigenousTerritory(
        name="Miwok",
        people="Coast Miwok / Bay Miwok",
        region="northern",
        centroid_lat=38.00, centroid_lon=-122.50,
        bbox=(37.5, -123.0, 38.5, -122.0),
        description="Marin County and North Bay — adjacent to San Andreas at Point Reyes",
        url="https://native-land.ca/maps/territories/coast-miwok",
    ),
    IndigenousTerritory(
        name="Pomo",
        people="Pomo",
        region="northern",
        centroid_lat=39.00, centroid_lon=-123.00,
        bbox=(38.3, -123.5, 39.7, -122.5),
        description="Sonoma, Mendocino, and Lake counties — northern SAF corridor",
        url="https://native-land.ca/maps/territories/pomo",
    ),
    IndigenousTerritory(
        name="Wiyot",
        people="Wiyot",
        region="northern",
        centroid_lat=40.75, centroid_lon=-124.10,
        bbox=(40.4, -124.4, 41.1, -123.8),
        description="Humboldt Bay area — northern terminus of SAF corridor",
        url="https://native-land.ca/maps/territories/wiyot",
    ),
    IndigenousTerritory(
        name="Kashaya",
        people="Kashaya Pomo",
        region="northern",
        centroid_lat=38.55, centroid_lon=-123.20,
        bbox=(38.3, -123.5, 38.8, -122.9),
        description="Sonoma Coast — Fort Ross area, directly on SAF trace",
        url="https://native-land.ca/maps/territories/kashaya",
    ),
]


def get_saf_territories() -> List[IndigenousTerritory]:
    """Return indigenous territories along the San Andreas Fault corridor."""
    log.info("Native Land: %d territories loaded (static data)", len(_TERRITORIES))
    return list(_TERRITORIES)


def find_territories_for_point(
    lat: float, lon: float
) -> List[IndigenousTerritory]:
    """Find which territories contain a given lat/lon point (bbox check)."""
    matches = []
    for t in _TERRITORIES:
        min_lat, min_lon, max_lat, max_lon = t.bbox
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            matches.append(t)
    return matches
