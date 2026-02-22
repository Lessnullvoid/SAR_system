# S.A.R — Seismic / Atmospheric Radio

**Multi-sensor research and sonic instrument for the San Andreas Fault**

![S.A.R System](img/sar1.png)

---

## Overview

**S.A.R** is a hybrid scientific-artistic system that monitors electromagnetic, seismic, geodetic, atmospheric, and satellite signals along the **San Andreas Fault corridor**. It operates as a **local, autonomous instrument**: ingesting open sensor data, detecting anomalies across multiple modalities, mapping them to a geographic tile grid, and translating the resulting state into both visual and sonic output.

The system combines an **RTL-SDR radio receiver** with **machine learning anomaly detection**, **real-time geophysical sensor fusion**, an **interactive fault corridor map**, and a **SuperCollider drone** that evolves in response to the Earth's activity.

S.A.R does **not** claim to predict earthquakes. It explores correlations and patterns across sensing modalities and temporal scales.

---

## Core Principles

- Multi-sensor correlation, not prediction
- Transparent use of open scientific datasets
- Local processing only (no cloud, no web streaming)
- Sound as an analytical and perceptual tool
- Respect for place, people, and data ethics

---

## Architecture

```
SAR_system/
├── python_app/                    # SDR application + ML pipeline
│   ├── main.py                    # Entry point
│   ├── gui_main.py                # Main GUI (3-column layout, PyQt5)
│   ├── dsp_core.py                # FFT, filtering, decimation
│   ├── demod.py                   # FM/AM/SSB demodulators with AGC
│   ├── audio_output.py            # Ring-buffer audio thread (sounddevice)
│   ├── rtl_device.py              # RTL-SDR hardware interface (pyrtlsdr)
│   ├── config/
│   │   └── antennas.json          # Antenna configurations
│   └── ml/
│       ├── bands.py               # Frequency band definitions (VLF → 70cm)
│       ├── features.py            # 8 spectral feature extractors
│       ├── detector.py            # Z-score + Isolation Forest anomaly detector
│       ├── monitor.py             # ML pipeline orchestrator
│       ├── scanner.py             # Automated radio band scanner
│       └── store.py               # SQLite feature/anomaly storage
│
├── sar/                           # Geospatial + sensor fusion
│   ├── geo/
│   │   ├── fault_trace.py         # San Andreas Fault trace + corridor
│   │   ├── tile_grid.py           # 10 km tile grid (EPSG:3310)
│   │   ├── tile_state.py          # Per-tile sensor state dataclasses
│   │   ├── sat_tiles.py           # ESRI satellite imagery downloader + cache
│   │   └── vector_data.py         # Roads, cities, faults, borders, labels
│   ├── gui/
│   │   └── map_widget.py          # FaultMapWidget + TileItem + MapScanner
│   ├── ingest/
│   │   ├── usgs_client.py         # USGS earthquake API
│   │   ├── noaa_client.py         # NOAA SWPC Kp/Dst
│   │   ├── ionospheric_client.py  # TEC estimation (Kp-based empirical)
│   │   ├── geomag_client.py       # Geomagnetic state classification
│   │   ├── gnss_client.py         # GNSS crustal deformation (UNR NGL)
│   │   ├── weather_client.py      # NOAA NWS weather
│   │   ├── ice_client.py          # GDELT news feed
│   │   ├── census_client.py       # U.S. Census / ACS demographics
│   │   ├── native_land_client.py  # Native Land Digital territories
│   │   ├── scedc_client.py        # Southern CA Earthquake Data Center
│   │   ├── tile_mapper.py         # Spatial assignment of events to tiles
│   │   └── sensor_scheduler.py    # QTimer + threading poller
│   ├── fusion/
│   │   └── engine.py              # Multi-sensor composite scoring + alerts
│   ├── osc/
│   │   └── bridge.py              # OSC bridge + SuperCollider process manager
│   └── storage/
│       └── tile_store.py          # SQLite tile state persistence
│
├── supercollider/
│   ├── sar_drone.scd              # Geological drone SynthDef
│   └── start_sc.sh               # Manual SC startup script (Pi)
│
├── data/
│   ├── sat_cache/                 # Satellite imagery cache (PNG)
│   ├── news_cache/                # Downloaded news images (512x512 PNG)
│   └── seismo_em.db               # ML feature/anomaly SQLite database
│
├── python_sdr/                    # Standalone SDR app (pre-SAR snapshot)
│
├── SENSORS.md                     # Detailed sensor documentation
└── requirements.txt               # Python dependencies
```

---

## Sensor Data Sources

All data sources are free, open access, and require no API keys.

| # | Sensor | Source | Interval | Fusion Weight |
|---|--------|--------|----------|---------------|
| 1 | **Seismic** | USGS Earthquake Catalog (GeoJSON) | 2 min | 30% |
| 2 | **RF Anomaly** | Local RTL-SDR + ML pipeline | Continuous | 25% |
| 3 | **Geomagnetic** | NOAA SWPC (Kp / Dst indices) | 5 min | 15% |
| 4 | **Ionospheric TEC** | Kp-based empirical model | 5 min | 15% |
| 5 | **GNSS Deformation** | UNR Nevada Geodetic Lab (30 stations) | 30 min | 10% |
| 6 | **Weather** | NOAA NWS API (19 stations) | 10 min | 5% |

Additional contextual feeds:
- **GDELT Project** — News articles (ICE/immigration focus) geocoded to fault tiles, displayed as map overlays (15 min)
- **U.S. Census / ACS** — Demographic data for the Social tab (on startup)
- **Native Land Digital** — Indigenous territory data (on startup)

See [SENSORS.md](SENSORS.md) for comprehensive documentation of all sensors, endpoints, processing, and scoring formulas.

---

## Spatial Model — San Andreas Tile System

The fault is modeled as a **north-south corridor** subdivided into ~800 on-fault tiles (10 km x 10 km, EPSG:3310 projection). Each tile aggregates sensor data into a composite anomaly score that drives both the map visualization and the sonic output.

```json
{
  "tile_id": "SAF_023",
  "time_utc": "2026-02-22T10:00:00Z",
  "seismic": {"events_24h": 11, "max_mag": 3.2},
  "gnss": {"median_mm_day": 0.18},
  "weather": {"wind": 5.3},
  "magnetics": {"kp": 3},
  "rf": {"anomaly_score": 0.71}
}
```

---

## Map & Scanner

The fault corridor map renders satellite imagery (ESRI World Imagery, grayscale with histogram stretch) with vector overlays (coastline, highways, secondary faults, cities) and real-time earthquake markers.

**Tile rendering:**
- Active tiles (score > 0.05): fully inverted (negative) satellite imagery
- Quiet on-fault tiles: original full-contrast imagery
- Off-fault tiles: darkened overlay

**Scanner navigation** uses a cluster-based system:
1. Identifies active clusters (tiles with seismic activity, anomaly scores, or news)
2. Expands clusters to include surrounding tiles (~15 km radius)
3. Visits each cluster with regional zoom-out, then close-up per tile
4. After all active clusters: free navigation covers remaining sectors
5. Zooms to full corridor between clusters for spatial context

---

## RF / ML Anomaly Detection

The SDR subsystem captures IQ data from an **RTL-SDR** receiver and processes it through a multi-stage DSP pipeline (decimation, channel filtering, FM/AM/SSB demodulation with two-stage AGC).

**ML scanner** automatically tunes through seismo-EM frequency bands (VLF through 70cm, excluding FM broadcast which suppresses relevant signal characteristics). At each band, 8 spectral features are extracted per FFT frame:

1. Band power (dB)
2. Peak frequency (Hz)
3. Spectral entropy
4. Band kurtosis
5. Spectral flatness
6. Spectral centroid
7. Band variance
8. Peak-to-mean ratio

**Detection**: Z-score (EMA baseline) + Isolation Forest, averaged. Alert threshold: 0.75.

---

## Audio Pipeline

```
RTL-SDR USB → Reader Thread → SpectrumWorker DSP
  → VFO shift (float64 phase) → Multi-stage decimation
  → Channel filter → Demodulation (FM/AM/SSB)
  → AGC (per-mode targets) → Output smoothing (15 kHz LPF)
  → Ring buffer → Audio playback thread → sounddevice
```

All demodulators normalize to consistent output levels via slow AGC. The volume slider acts as an absolute ceiling — set it to 40% and no band ever exceeds 40%.

---

## SuperCollider Drone

A continuous **geological drone** evolves through 7 minor keys (Am → Dm → Gm → Cm → Fm → Bbm → Ebm), driven by sensor data via OSC on port 57120.

**Design philosophy:**
- **Glacial tempo** — Full chord cycle takes 15-40 minutes. Individual chords linger for 5+ minutes. 20-second portamento makes transitions feel tectonic.
- **Compositional sensor mapping** — Activity controls *which* voices are present through thresholds (quiet: root + sub only; awakening: third emerges; active: fifth + grains; tense: full voicing with 7th). Kp shapes timbre. Events drive harmonic rhythm.
- **Near-constant volume** — Interest comes from timbral and harmonic evolution, not loudness changes. Amplitude range is deliberately tight.
- **Slow breathing** — Overlapping 1-4 minute modulation cycles create swells felt more than heard.

| Sensor | Sound Parameter |
|--------|----------------|
| `activity` (0-1) | Voice emergence thresholds, harmonic lean |
| `events` (count) | Chord progression speed, grain density |
| `kp` (0-9) | Filter brightness, 7th voice gate |
| `dst` (nT) | Microtonal detuning (storm = dissonance) |
| `alert` (0-1) | Dissonant minor 9th tension tone |

Python auto-launches `sclang` on startup and kills it on exit.

---

## GUI Layout

```
┌──────────────┬───────────────┬──────────────────────┐
│  Column 1    │  Column 2     │  Column 3            │
│  RF Display  │  Tabs:        │  Fault Map           │
│              │  - ML Monitor │  (always visible)    │
│  Spectrum    │  - Sensors    │                      │
│  (pyqtgraph) │  - Social    │  Satellite tiles +   │
│              │  - Audio     │  vector overlay +    │
│  Waterfall   │               │  earthquake markers  │
│  (pyqtgraph) │  News Feed   │  + scanner nav       │
│              │  Anomaly Plot │                      │
│              │  Anomaly Log  │                      │
├──────────────┴───────────────┴──────────────────────┤
│  Footer: SDR status, Volume, PPM, Sample rate       │
└─────────────────────────────────────────────────────┘
```

Tabs auto-rotate (ML → Sensors → Social → Audio), context-driven by scanner activity.

---

## Thread Model

| Thread | Role |
|--------|------|
| SDR Reader | USB I/O → IQ queue (decoupled from DSP) |
| SpectrumWorker | FFT + DSP + demodulation (QThread) |
| Audio Playback | Ring buffer → sounddevice (blocking write) |
| Qt Main | GUI, map, scanner, ML monitor, tabs |
| Sensor Pollers | Background threads for each API (USGS, NOAA, GNSS, etc.) |
| SuperCollider | `sclang` subprocess, receives OSC |

---

## Running

```bash
cd SAR_system
pip install -r requirements.txt
python -m python_app.gui_main
```

Requires:
- **RTL-SDR** USB dongle (pyrtlsdr)
- **SuperCollider** installed (optional — system works without it)
- **Python 3.10+**
- Internet access for sensor API polling

### Raspberry Pi 5 (8 GB recommended)

The system runs on Raspberry Pi 5 with 8 GB RAM. Pi 4 with less than 4 GB is not supported (the full tile map + SuperCollider + SDR pipeline exceeds ~650 MB).

**1. System packages** (heavy libraries that can't compile on Pi):

```bash
sudo apt update
sudo apt install python3-pyqt5 python3-pyqtgraph python3-numpy \
    python3-scipy python3-pil python3-matplotlib \
    supercollider sc3-plugins jackd2 git
```

Say **Yes** when asked about real-time priority for JACK.

**2. Clone and set up virtual environment:**

```bash
git clone https://github.com/Lessnullvoid/SAR.git
cd SAR
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install pyrtlsdr sounddevice scikit-learn shapely pyproj requests python-osc
```

The `--system-site-packages` flag lets pip-installed packages coexist with the apt-installed PyQt5/numpy/scipy.

**3. Increase swap** (safety net for peak memory during startup):

```bash
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

**4. Audio group** (required for JACK real-time priority):

```bash
sudo usermod -a -G audio $USER
```

Log out and back in for the group change to take effect.

**5. Run:**

```bash
cd ~/SAR
source .venv/bin/activate
python -m python_app.gui_main
```

On first run, click the **Satellite** button to download map tiles (~50 MB at Pi resolution). After that, tiles are cached locally.

**Automatic Pi optimizations:**
- Satellite tiles downloaded at 128px (vs 512px desktop), displayed at 64x64
- Reduced animation FPS (30 fps vs 60 fps)
- Batched tile loading to avoid GIL starvation of audio pipeline
- SDR audio routed to BCM2835 headphone jack; SuperCollider via USB sound card
- SuperCollider launch deferred until after map and SDR are stable
- Staggered sensor polling to spread network and CPU load

---

## Dependencies

```
numpy>=1.24        scipy>=1.10        PyQt5>=5.15
pyqtgraph>=0.13    pyrtlsdr>=0.3.0    sounddevice>=0.4
scikit-learn>=1.3   shapely>=2.0       pyproj>=3.6
requests>=2.28     Pillow>=9.0        python-osc>=1.8
```

---

## Ethics Statement

- S.A.R does **not** provide warnings, predictions, or safety guidance.
- Detected anomalies are exploratory signals, not causal claims.
- Environmental, geomagnetic, and instrumental confounders are always considered.
- Data sources are credited and used according to their licenses.
- The project avoids sensationalism and public alarm.
- Indigenous territories and local communities are acknowledged.

---

## Documentation

| File | Contents |
|------|----------|
| [SENSORS.md](SENSORS.md) | All sensor sources, endpoints, processing, scoring formulas, station lists |


---

## License

Code: MIT (proposed)
Concept, sound, and visual outputs: Artistic license (TBD)

---

S.A.R — Seismic / Atmospheric Radio
Hybrid research instrument bridging geophysics, electromagnetism, and sound.
