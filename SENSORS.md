# S.A.R — Sensor Reference

Complete documentation of all sensor data sources integrated in the
Seismic / Atmospheric Radio system.

---

## Sensor Summary

| # | Sensor | Source | Measures | Interval | Weight |
|---|--------|--------|----------|----------|--------|
| 1 | **Seismic (USGS)** | USGS GeoJSON API | Earthquakes: location, magnitude, depth, time | 2 min | 30% |
| 2 | **RF Anomaly (ML)** | Local RTL-SDR | EM anomaly score from 8 spectral features | ~167 ms | 25% |
| 3 | **Geomagnetic (Kp/Dst)** | NOAA SWPC | Planetary K-index, Dst ring current | 5 min | 15% |
| 4 | **Ionospheric TEC** | Kp-based model | Total Electron Content, deviation from median | 5 min | 15% |
| 5 | **GNSS Deformation** | UNR Nevada Geodetic Lab | Crustal displacement rate (mm/day) | 30 min | 10% |
| 6 | **Weather** | NOAA NWS API | Temperature, pressure, wind, humidity | 10 min | 5% |

**Composite score** = Σ (sensor_score × weight) — drives tile inversion on map.

---

## 1. USGS Earthquake Catalog (Seismic)

### What it measures
Real-time earthquake events along the San Andreas Fault corridor:
- **Location** (latitude, longitude)
- **Magnitude** (ml, mw, mb)
- **Depth** (km)
- **Origin time** (UTC)
- **Place** (human-readable description)
- **Status** (automatic / reviewed)

### Data source
**USGS Earthquake Hazards Program**

| Endpoint | URL |
|----------|-----|
| Real-time feed | `https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/{period}.geojson` |
| FDSN query | `https://earthquake.usgs.gov/fdsnws/event/1/query` |

Periods: `hour`, `day`, `week`, `month`

### Polling interval
**120 seconds** (2 minutes)

### Implementation
- **Client**: `sar/ingest/usgs_client.py`
- **Dataclass**: `Earthquake` (event_id, time_ms, lat, lon, depth_km, mag, mag_type, place, status, url)
- **Spatial filter**: Bounding box around the San Andreas Fault corridor (lat 32-42, lon -125 to -114)
- **Minimum magnitude**: 1.0

### Tile mapping
Each earthquake's lat/lon is projected (EPSG:4326 → EPSG:3310) and matched to the nearest tile using `TileMapper.map_earthquakes()`.

Per-tile fields updated:
- `SeismicState.events_24h` — count of quakes ≤ 24 hours old
- `SeismicState.events_1h` — count ≤ 1 hour old
- `SeismicState.max_mag` — maximum magnitude in tile
- `SeismicState.latest_mag` — most recent event's magnitude
- `SeismicState.latest_depth_km` — most recent depth
- `SeismicState.latest_time_utc` — most recent time

### Composite score formula
```
seismic_score = min(events_24h / 50.0, 1.0)
```
Weight: **0.30** (30%)

### Map visualization
- **Epicenter circles**: Cyan (<1h), blue (<24h), fading grey (older). Radius scales with magnitude (1.5–20 scene units).
- **Tile badges**: Top-right corner dot + earthquake count label per tile.
- **Tile inversion**: Tiles with composite score > 0.05 show inverted (negative) satellite imagery.
- **Tooltips**: Hover for magnitude, place, depth, time, age.

---

## 2. RF Anomaly Detection (SDR / ML)

### What it measures
Electromagnetic anomalies in the radio spectrum detected by machine learning:
- **8 spectral features** per frequency band per FFT frame:
  1. Band power (dB)
  2. Peak frequency (Hz)
  3. Spectral entropy
  4. Band kurtosis
  5. Spectral flatness
  6. Spectral centroid
  7. Band variance
  8. Peak-to-mean ratio
- **Composite anomaly score** (0–1)

### Data source
**Local RTL-SDR receiver** (pyrtlsdr)

| Parameter | Value |
|-----------|-------|
| Sample rate | 2.4 MHz |
| Hardware | RTL-SDR USB dongle |
| Antenna | Loop (VLF–MF) / Discone (VHF+) |

### Frequency bands monitored

| Band | Range | Antenna | Seismo-EM Interest |
|------|-------|---------|-------------------|
| VLF | 300 Hz – 30 kHz | loop | Pre-seismic ULF/VLF emissions |
| LF | 30 – 300 kHz | loop | Atmospheric EM anomalies |
| MF | 300 kHz – 1.7 MHz | loop | Ground wave propagation changes |
| HF | 2 – 30 MHz | loop | Ionospheric reflection anomalies |
| VHF | 30 – 300 MHz | discone | Sporadic E, tropospheric ducting |
| FM | 88 – 108 MHz | fm_broadcast | Signal propagation monitoring |
| 2m | 144 – 148 MHz | discone | Amateur radio propagation |
| 70cm | 430 – 450 MHz | discone | Higher frequency anomalies |

### Polling interval
**Continuous** (~6 Hz, every FFT frame from SpectrumWorker)

### Implementation
- **Monitor**: `python_app/ml/monitor.py` (SeismoMonitor)
- **Detector**: `python_app/ml/detector.py` (AnomalyDetector)
- **Bands**: `python_app/ml/bands.py`
- **Features**: `python_app/ml/features.py`

### Detection algorithm
Two independent detectors, averaged:

**Z-Score detector** (EMA baseline):
- Alpha: 0.002 (slow adaptation)
- Warmup: 500 samples
- Threshold: 4.0 standard deviations
- Produces per-feature Z-scores, composite = max|Z| normalized to [0,1]

**Isolation Forest** (scikit-learn):
- Contamination: 0.05
- Retrained every 300 samples on rolling window
- Produces anomaly score [0,1]

**Composite**: `(z_composite + if_score) / 2.0`
- Alert threshold: **0.75**

### Tile mapping
Average score across all bands, applied with distance falloff from the SDR site location to nearby tiles.

### Composite score formula
```
rf_score = anomaly_score  # direct [0,1]
```
Weight: **0.25** (25%)

### Map visualization
- Contributes to tile composite score → drives tile inversion
- **ML Monitor tab**: Band status table with per-band scores
- **Anomaly timeline**: Real-time plot (last 120 seconds)
- **Spectrum markers**: Cyan dashed line at anomalous peak frequency
- **Auto-tuning**: VFO tunes to anomalous signal during scanning

---

## 3. Geomagnetic Indices (Kp / Dst)

### What it measures
Planetary geomagnetic field disturbance:
- **Kp index** (0–9): Global geomagnetic activity level
  - 0–2: Quiet
  - 3: Unsettled
  - 4: Minor storm
  - 5–6: Moderate storm
  - 7–9: Severe storm
- **Dst index** (nanoTesla): Ring current intensity
  - > -30 nT: Quiet
  - -30 to -50: Unsettled
  - -50 to -100: Minor storm
  - -100 to -200: Moderate storm
  - < -200: Severe storm

### Data source
**NOAA Space Weather Prediction Center (SWPC)**

| Endpoint | URL |
|----------|-----|
| Kp index | `https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json` |
| Dst index | `https://services.swpc.noaa.gov/products/kyoto-dst.json` |
| Solar wind Bz | `https://services.swpc.noaa.gov/products/solar-wind/mag-2-hour.json` |

### Polling interval
**300 seconds** (5 minutes)

### Implementation
- **Client**: `sar/ingest/noaa_client.py`
- **Classifier**: `sar/ingest/geomag_client.py`
- **Dataclasses**: `KpReading`, `DstReading`, `MagneticReading`

### Tile mapping
Global values — applied equally to **all on-fault tiles** (geomagnetic storms affect the entire corridor uniformly).

Per-tile fields updated:
- `MagneticState.kp_index`
- `MagneticState.dst_index`

### Composite score formula
```
magnetic_score = min(kp_index / 9.0, 1.0)
```
Weight: **0.15** (15%)

### Map visualization
- Contributes to tile composite score
- **Sensors tab**: Text display with color-coded storm level
- Storm levels: QUIET (grey) → UNSETTLED (green) → MINOR (blue) → MODERATE (cyan) → SEVERE (bright cyan)

---

## 4. Ionospheric TEC (Total Electron Content)

### What it measures
- **TEC** (TECU): Total electron content in the ionosphere above the fault corridor
- **TEC delta** (σ): Deviation from the 27-day rolling median

Pre-seismic ionospheric anomalies have been reported in literature as TEC
enhancements or depletions in the days before major earthquakes.

### Data source
**Kp-based empirical estimation model**

Currently uses an empirical relationship between Kp index and TEC rather
than direct TEC map ingestion. The model estimates TEC at the fault corridor
centroid (lat 36.0°, lon -120.0°) based on current Kp value.

Future: Direct ingestion from NASA JPL GDGPS TEC maps
(`https://gdgps.jpl.nasa.gov/products/tec-maps.html`)

### Polling interval
**300 seconds** (5 minutes, same timer as geomagnetic)

### Implementation
- **Client**: `sar/ingest/ionospheric_client.py`
- **Dataclass**: `TECReading` (tec_tecu, tec_median, tec_delta, latitude, longitude, time_tag)
- **Functions**: `estimate_tec_from_kp()`, `get_tec_for_location()`

### Tile mapping
Single estimate for the fault corridor centroid — applied to **all on-fault tiles**.

Per-tile fields updated:
- `IonosphericState.tec_tecu`
- `IonosphericState.tec_delta`

### Composite score formula
```
ionospheric_score = min(abs(tec_delta) / 20.0, 1.0)
```
Weight: **0.15** (15%)

### Map visualization
- Contributes to tile composite score
- **Sensors tab**: TEC value + delta with color highlighting (cyan if |delta| > 2σ)

---

## 5. GNSS Crustal Deformation

### What it measures
Daily GPS/GNSS position solutions measuring millimetre-scale crustal motion:
- **Displacement rate** (mm/day): Total 3D movement rate
- **North component** (mm/day): North-south motion
- **East component** (mm/day): East-west motion  
- **Vertical component** (mm/day): Up-down motion
- **Station count**: Number of reporting stations per tile

Anomalous displacement rates may indicate accelerating tectonic strain,
slow-slip events, or post-seismic deformation.

### Data source
**University of Nevada, Reno — Nevada Geodetic Laboratory (UNR/NGL)**

| Resource | URL |
|----------|-----|
| Station data | `https://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/{STATION}.tenv3` |
| Station map | `https://geodesy.unr.edu/NGLStationPages/gpsnetmap/GPSNetMap.html` |
| Format docs | `https://geodesy.unr.edu/gps_timeseries/readmes/` |

**Data format**: tenv3 (daily position time series in IGS14 reference frame)
- Columns: station, date, decimal_year, dN, dE, dU (offsets in metres), sigmas, correlations, lat, lon, height
- Updated daily with ~24-hour latency

### Monitored stations (30 along the San Andreas Fault, N→S)

| Station | Location | Region |
|---------|----------|--------|
| P198 | Cape Mendocino | Northern CA |
| P162 | Shelter Cove | Northern CA |
| P159 | Willits | Northern CA |
| LUTZ | Upper Lake | Northern CA |
| P196 | Cloverdale | Northern CA |
| SVIN | Santa Rosa | North Bay |
| P200 | Petaluma | North Bay |
| TIBB | Tiburon | Bay Area |
| P224 | San Francisco | Bay Area |
| P225 | Daly City | Bay Area |
| P229 | Hayward | Bay Area |
| P231 | San Jose | Bay Area |
| P247 | Hollister | Central CA |
| P250 | Pinnacles | Central CA |
| P254 | Coalinga | Central CA |
| P261 | Parkfield | Central CA |
| P263 | Cholame | Central CA |
| CARH | Carrizo Plain | Central CA |
| P504 | Maricopa | Transverse Ranges |
| VNDN | Vandenberg | Central Coast |
| P503 | Santa Barbara | South Coast |
| P502 | Ventura | South Coast |
| AZU1 | Azusa | LA Basin |
| P487 | San Bernardino | Inland Empire |
| P486 | Banning | Inland Empire |
| P497 | Palm Springs | Coachella Valley |
| P500 | Salton Sea | Salton Trough |
| P494 | Calipatria | Imperial Valley |
| P496 | El Centro | Imperial Valley |
| IID2 | Imperial | Imperial Valley |

### Polling interval
**1800 seconds** (30 minutes) — data is daily, so frequent polling is unnecessary.
Initial fetch delayed 5 seconds after startup to avoid network burst.

### Implementation
- **Client**: `sar/ingest/gnss_client.py`
- **Dataclass**: `GNSSReading` (station_id, lat, lon, description, displacement_mm_day, north_mm_day, east_mm_day, vertical_mm_day, days_of_data, last_epoch)
- **Functions**: `fetch_gnss_displacements()`, `fetch_station_displacement()`

### Processing
1. Downloads the tenv3 file for each station from UNR
2. Parses the most recent 7 days of data
3. Computes displacement rate: linear difference between first and last epoch
4. Total rate = √(dN² + dE² + dU²) in mm/day

### Tile mapping
Each station's lat/lon is matched to a tile via `TileMapper.find_tile()`.
- Tiles with stations: use the station's displacement rate directly
- Tiles without stations: use the corridor-wide average rate

Per-tile fields updated:
- `GNSSState.median_mm_day`
- `GNSSState.max_mm_day`
- `GNSSState.station_count`

### Composite score formula
```
gnss_score = min(max_mm_day / 5.0, 1.0)
```
5.0 mm/day = full score (very high displacement, likely associated with an event)

Weight: **0.10** (10%)

### Map visualization
- Contributes to tile composite score → tile inversion
- **Sensors tab**: Station count, average rate, maximum rate with station ID
- Color highlighting: cyan when max displacement > 0.3 mm/day

---

## 6. Weather (Atmospheric Context)

### What it measures
Current atmospheric conditions along the fault corridor:
- **Temperature** (°C)
- **Barometric pressure** (hPa)
- **Wind speed** (m/s) and direction (°)
- **Relative humidity** (%)
- **Visibility** (m)
- **Description** (text: "Partly Cloudy", "Rain", etc.)

Weather is primarily a **contextual** sensor — high winds and pressure
anomalies can affect RF propagation, instrument readings, and GNSS accuracy.

### Data source
**NOAA National Weather Service (NWS) API**

| Endpoint | URL |
|----------|-----|
| Station observations | `https://api.weather.gov/stations/{ID}/observations/latest` |
| Point metadata | `https://api.weather.gov/points/{lat},{lon}` |
| Documentation | `https://www.weather.gov/documentation/services-web-api` |

No API key required. Returns GeoJSON. User-Agent header required.

### Monitored stations (19 along the fault corridor, N→S)

| Station | Location | Region |
|---------|----------|--------|
| KEKA | Eureka/Arcata | Northern CA |
| KUKI | Ukiah | Northern CA |
| KSTS | Santa Rosa | North Bay |
| KSFO | San Francisco Intl | Bay Area |
| KSJC | San Jose | Bay Area |
| KHOL | Hollister | Central CA |
| KSNS | Salinas | Central CA |
| KPRB | Paso Robles | Central CA |
| KSBP | San Luis Obispo | Central Coast |
| KVBG | Vandenberg AFB | Central Coast |
| KSBA | Santa Barbara | South Coast |
| KBUR | Burbank | LA Basin |
| KLAX | Los Angeles Intl | LA Basin |
| KONT | Ontario | Inland Empire |
| KSBD | San Bernardino | Inland Empire |
| KPSP | Palm Springs | Coachella Valley |
| KTRM | Thermal | Coachella Valley |
| KIPL | Imperial | Imperial Valley |
| KSDM | San Diego | San Diego |

### Polling interval
**600 seconds** (10 minutes) — NWS observations update hourly.
Initial fetch delayed 8 seconds after startup.

### Implementation
- **Client**: `sar/ingest/weather_client.py`
- **Dataclass**: `WeatherReading` (station_id, station_name, region, lat, lon, temperature_c, pressure_hpa, wind_speed_ms, wind_direction_deg, humidity_pct, visibility_m, description, timestamp)
- **Functions**: `fetch_weather_observations()`, `compute_corridor_summary()`

### Tile mapping
Each weather station's lat/lon is matched to a tile via `TileMapper.find_tile()`.
- Tiles with a station: use that station's observations
- Tiles without a station: use corridor-wide averages

Per-tile fields updated:
- `WeatherState.temperature_c`
- `WeatherState.pressure_hpa`
- `WeatherState.wind_speed_ms`
- `WeatherState.humidity_pct`

### Composite score formula
```
weather_score = min(wind_speed_ms / 25.0, 1.0)  # high wind = higher score
```
Only wind speed contributes (threshold: >5 m/s starts contributing).

Weight: **0.05** (5%) — context only, not a primary anomaly indicator.

### Map visualization
- Small contribution to tile composite score
- **Sensors tab**: Station count, average temperature, wind speed, pressure, humidity
- Color highlighting: cyan when average wind > 10 m/s

---

## Data Flow Architecture

```
                 ┌─────────────────────────────────────┐
                 │       SensorScheduler                │
                 │       (QTimer + threading)           │
                 └──────────────┬───────────────────────┘
                                │
         ┌──────────┬───────────┼───────────┬──────────────┐
         │          │           │           │              │
         ▼          ▼           ▼           ▼              ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
    │  USGS   │ │  NOAA   │ │   TEC   │ │  GNSS   │ │ Weather │
    │ (2 min) │ │ (5 min) │ │ (5 min) │ │(30 min) │ │(10 min) │
    └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
         │          │           │           │            │
         ▼          ▼           ▼           ▼            ▼
    ┌─────────────────────────────────────────────────────────┐
    │                    TileMapper                           │
    │           (lat/lon → tile spatial join)                 │
    └──────────────────────┬──────────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────────────┐
    │                    TileState                            │
    │  ┌──────────┬──────────┬─────────┬────────┬─────────┐  │
    │  │ Seismic  │ Magnetic │  Iono   │  GNSS  │ Weather │  │
    │  │  (30%)   │  (15%)   │ (15%)   │ (10%)  │  (5%)   │  │
    │  └──────────┴──────────┴─────────┴────────┴─────────┘  │
    │              + RF Anomaly (25%)                         │
    │                    ↓                                    │
    │            composite_score                              │
    └──────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │   Map    │ │  Fusion  │ │   OSC    │
        │  Widget  │ │  Engine  │ │  Bridge  │
        │ (visual) │ │ (alerts) │ │ (sound)  │
        └──────────┘ └──────────┘ └──────────┘
```

---

## Composite Score Calculation

The per-tile composite score aggregates all sensor anomaly indicators:

```python
composite = (
    0.30 × min(seismic.events_24h / 50, 1.0)        # Earthquakes
  + 0.25 × rf.anomaly_score                           # RF/ML
  + 0.15 × min(magnetic.kp_index / 9, 1.0)           # Geomagnetic
  + 0.15 × min(|ionospheric.tec_delta| / 20, 1.0)    # Ionospheric
  + 0.10 × min(gnss.max_mm_day / 5, 1.0)             # Crustal deformation
  + 0.05 × min(weather.wind_speed_ms / 25, 1.0)      # Weather context
)
```

**Score interpretation:**
- 0.00 – 0.05: Quiet — normal satellite imagery
- 0.05 – 0.30: Low activity — tile shows inverted (negative) imagery
- 0.30 – 0.60: Moderate — scanner visits with standard dwell
- 0.60 – 1.00: High — scanner visits with extended dwell

---

## Network Requirements

| Sensor | Bandwidth | Requests/hour |
|--------|-----------|---------------|
| USGS | ~50 KB/request | 30 |
| NOAA Kp/Dst | ~10 KB/request | 12 |
| TEC | 0 (computed locally) | 0 |
| GNSS | ~100 KB/station × 30 stations | 2 |
| Weather | ~5 KB/station × 19 stations | 6 |
| **Total** | ~4 MB/hour | ~50 |

All data sources are free, open access, and require no API keys.

---

*Last updated: 2026-02-14*
