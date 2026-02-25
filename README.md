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
│   ├── gui_main.py                # Entry point + Main GUI (PyQt5)
│   ├── __init__.py                # Package docstring
│   ├── dsp_core.py                # FFT, filtering, decimation
│   ├── demod.py                   # FM/AM/SSB demodulators with AGC
│   ├── audio_output.py            # Ring-buffer audio thread (sounddevice)
│   ├── rtl_device.py              # RTL-SDR hardware interface (pyrtlsdr)
│   ├── config/
│   │   └── antennas.json          # Antenna profiles (loop, FM whip, discone)
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
│   │   ├── map_widget.py          # FaultMapWidget + TileItem + MapScanner
│   │   └── narrative_engine.py    # Cycle-based narrative map navigation engine
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
│   └── sar_main.scd               # Drone + Resonator SynthDefs
│
├── scripts/
│   ├── sar_autostart.sh           # Pi auto-start script
│   └── sar.desktop                # Desktop autostart entry
│
├── data/
│   ├── sat_cache/                 # Satellite imagery cache (PNG)
│   ├── news_cache/                # Curated news images + articles.json
│   ├── history_cache/             # Curated historical event images
│   └── seismo_em.db               # ML feature/anomaly SQLite database
│
├── python_sdr/                    # Standalone SDR app (pre-SAR snapshot)
│
├── SENSORS.md                     # Detailed sensor documentation
├── RASPBERRY_PI_SETUP.md          # Pi 5 installation guide
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
- **News imagery** — Curated collection of GDELT news images geocoded to fault tiles, displayed as map overlays (no automatic downloads; manually curated `data/news_cache/`)
- **Historical imagery** — Curated historical event photographs along the fault corridor (manually curated `data/history_cache/`)
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

## Map & Narrative Scanner

The fault corridor map renders satellite imagery (ESRI World Imagery, grayscale with histogram stretch) with vector overlays (coastline, highways, secondary faults, cities) and real-time earthquake markers.

**Tile rendering:**
- Scanner-focused tile: color-inverted (negative) for immediate visibility
- Active tiles (score > 0.05): highlighted with cyan border
- Quiet on-fault tiles: original full-contrast imagery
- Off-fault tiles: darkened overlay

**Narrative engine** (`sar/gui/narrative_engine.py`) drives the map as a semi-generative navigation system organised into **cycles**. Each cycle builds a manifest of every notable tile and visits them using a different traversal strategy, so the presentation never repeats identically.

**Cycle structure:**
1. **Build manifest** — categorise all notable tiles: ALERT (seismic), ECHO (historical images), DISPATCH (news images), SURVEY (corridor fill). Tiles with both image types randomly alternate between ECHO and DISPATCH across cycles. Recently-visited tiles are deprioritised.
2. **Pick strategy** — six traversal strategies rotate without repetition: North-South, South-North, Section Walk, Hotspot First, Interleaved, Random Walk. Each includes internal shuffling for organic movement.
3. **Generate playlist** — ordered sequence of chapters with periodic BREATH (full corridor overview) inserts for spatial context.
4. **Execute** — each chapter composes shots (APPROACH, FOCUS, CONTEXT) that control camera animation, tile highlighting, and image overlays.
5. **Next cycle** — when the playlist is exhausted, a new cycle begins with a fresh manifest and different strategy.

**Image distribution (~40% of content):**
- ECHO and DISPATCH chapters show images at their source tiles (multi-tile span, aspect-ratio preserved, randomly selected from the tile's image pool)
- ALERT and SURVEY chapters can borrow images from the nearest image tile, dispersing visual content across the entire corridor
- Image overlays span 2-6 tiles with centered placement and border indicators

**ML anomaly interrupts:** When the ML detector fires a significant anomaly (score >= 0.75), up to 3 high-scoring on-fault tiles are queued as priority ALERT chapters. A 120-second per-tile cooldown and queue cap of 6 prevent flood monopolisation.

**Overlay indicators:**
- Strategy label at top of map shows current cycle, strategy name, and progress (e.g. `C2 SOUTH_NORTH 15/108`)
- All overlay text uses black background for readability across any map region

---

## RF / ML Anomaly Detection

The SDR subsystem captures IQ data from an **RTL-SDR** receiver and processes it through a multi-stage DSP pipeline (decimation, channel filtering, FM/AM/SSB demodulation with two-stage AGC).

**ML scanner** automatically tunes through seismo-EM frequency bands (VLF through HF 10m). FM broadcast (88-108 MHz), 2m ham (146 MHz), and 70cm (440 MHz) are excluded — FM suppresses amplitude/phase characteristics relevant to precursor detection, and ham FM bands produce dangerously loud audio spikes from repeaters. At each band, 8 spectral features are extracted per FFT frame:

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

All demodulators normalize to consistent output levels via slow AGC (FM target 0.03, AM/SSB target 0.12 to equalize perceived loudness). The volume slider acts as an absolute ceiling — set it to 50% and no band ever exceeds 50%. Default volume: 90%.

---

## SuperCollider Audio Engine

Two synths run simultaneously, both driven by the same sensor data via OSC on port 57120. Use the `--synth` flag to select which ones to launch.

### Geological Drone (`sar_drone`)

A continuous drone that evolves through 7 minor keys (Am → Dm → Gm → Cm → Fm → Bbm → Ebm).

**Design philosophy:**
- **Glacial tempo** — Full chord cycle takes 15-40 minutes. Individual chords linger for 5+ minutes. 20-second portamento makes transitions feel tectonic.
- **Compositional sensor mapping** — Activity controls *which* voices are present through thresholds (quiet: root + sub only; awakening: third emerges; active: fifth + grains; tense: full voicing with 7th). Kp shapes timbre. Events drive harmonic rhythm.
- **Silence timer** — A random timer triggers a silence episode every 5, 9, or 13 minutes (randomly chosen). Each episode: sinusoidal fade-out (25s) → silence (80s) → sinusoidal fade-in (35s). Independent of sensor data.
- **Slow breathing** — Overlapping 1-4 minute modulation cycles create deeper swells (down to 0.82x). The reverb tail swells as the drone retreats, leaving a lingering spatial residue in the silence.

### Sympathetic String Resonator (`sar_resonator`)

A 6-string harmonic bank tuned to E2 A2 D3 G3 B3 E4 (open guitar tuning). Each string is modeled as a set of partials with independent wobble and detuning. Strings activate progressively as seismic activity rises — quiet conditions sustain only the two lowest strings, while higher activity opens the upper register.

- **Physical excitation** — Impulse-driven with rate controlled by event count, simulating ground-coupled vibration exciting the strings.
- **Sympathetic resonance** — Partials interact through shared detuning (driven by Dst), so geomagnetic storms make the strings beat and shimmer.
- **Silence timer** — Triggers every 3, 7, or 11 minutes (randomly chosen). Shorter episodes than the drone: fade-out (15s) → silence (50s) → fade-in (20s). As it retreats, the stereo field widens — the last harmonics scatter before fading.
- **Alert response** — High anomaly scores inject dissonant energy into the string bank.

### OSC Sensor Mapping (shared by both synths)

| Sensor | Drone | Resonator |
|--------|-------|-----------|
| `activity` (0-1) | Voice emergence thresholds, harmonic lean | String gate thresholds, partial depth |
| `events` (count) | Chord progression speed, grain density | Impulse excitation rate |
| `kp` (0-9) | Filter brightness, 7th voice gate | Filter brightness |
| `dst` (nT) | Microtonal detuning (storm = dissonance) | Sympathetic detuning across all strings |
| `alert` (0-1) | Dissonant minor 9th tension tone | Dissonant energy injection |

Python auto-launches `sclang` on startup (via `pw-jack` on PipeWire systems) and kills it on exit. On Raspberry Pi 5, both SDR and SuperCollider audio are mixed through PipeWire to a shared USB sound card.

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
| Qt Main | GUI, map, narrative engine, scanner, ML monitor, tabs |
| Sensor Pollers | Background threads for each API (USGS, NOAA, GNSS, etc.) |
| SuperCollider | `sclang` subprocess, receives OSC |

---

## Running

```bash
cd SAR_system
pip install -r requirements.txt
python -m python_app.gui_main
```

The application starts in **fullscreen**. Press **F** to toggle fullscreen mode.

### Startup flags

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--antenna` | `loop_antenna`, `fm_broadcast`, `discone` | `loop_antenna` | Which antenna is physically connected to the RTL-SDR |
| `--synth` | `both`, `drone`, `resonator` | `both` | Which SuperCollider synth(s) to launch |

### Antenna selection

The `--antenna` flag tells the system which antenna is physically connected.
It sets the correct RTL-SDR hardware mode, default frequency, gain, and
demodulation mode at startup:

| Antenna | RTL-SDR mode | Default freq | Mode | Gain | Use case |
|---------|-------------|-------------|------|------|----------|
| `loop_antenna` | Direct Sampling Q | 1.0 MHz | AM | 20 dB | HF/VLF seismo-EM monitoring (primary) |
| `fm_broadcast` | Quadrature | 98.0 MHz | FM | 30 dB | VHF whip, FM broadcast reference |
| `discone` | Quadrature | 144.0 MHz | FM | 25 dB | Broadband VHF/UHF scanning |

The ML scanner plan currently uses only `loop_antenna` positions (VLF through HF 10m).

```bash
python -m python_app.gui_main --antenna loop_antenna     # HF loop (default)
python -m python_app.gui_main --antenna fm_broadcast     # VHF whip
python -m python_app.gui_main --antenna discone          # broadband discone
```

### SuperCollider synth selection

By default both synths (drone + sympathetic string resonator) launch together.
Use the `--synth` flag to select which one:

```bash
python -m python_app.gui_main --synth both        # default — drone + resonator
python -m python_app.gui_main --synth drone        # drone only
python -m python_app.gui_main --synth resonator    # resonator only
```

### Combined flags

```bash
python -m python_app.gui_main --antenna discone --synth drone
```

### Environment variables (Pi autostart)

On the Raspberry Pi autostart script (`scripts/sar_autostart.sh`), both flags
can be overridden via environment variables set before boot (e.g. in
`~/.bashrc` or directly in the script):

| Variable | Default | Equivalent flag |
|----------|---------|-----------------|
| `SAR_ANTENNA` | `loop_antenna` | `--antenna` |
| `SAR_SYNTH` | `both` | `--synth` |

Requires:
- **RTL-SDR** USB dongle (pyrtlsdr) — system runs without it (map + sensors still active)
- **SuperCollider** installed (optional — drone disabled if not found)
- **Python 3.10+**
- Internet access for sensor API polling

### Raspberry Pi 5 Setup (8 GB recommended)

The system runs on Raspberry Pi 5 with 8 GB RAM. Pi 4 with less than 4 GB is not supported (the full tile map + SuperCollider + SDR pipeline uses ~1.8 GB).

> **Full setup guide:** [RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md) — step-by-step instructions covering system packages, PipeWire audio, USB sound card, Python venv, troubleshooting, and monitoring.

#### 1. System packages

Heavy libraries that can't compile on Pi must be installed via apt:

```bash
sudo apt update
sudo apt install -y \
    python3-pyqt5 python3-pyqtgraph python3-numpy \
    python3-scipy python3-pil python3-matplotlib \
    supercollider sc3-plugins jackd2 git
```

Say **Yes** when asked about real-time priority for JACK.

#### 2. Audio setup (PipeWire + USB sound card)

Raspberry Pi 5 has **no 3.5mm audio jack**. Both SDR and SuperCollider share a USB sound card through PipeWire (the default audio server on Raspberry Pi OS Bookworm).

Install PipeWire's ALSA and JACK compatibility layers:

```bash
sudo apt install -y pipewire-alsa pipewire-jack
```

Restart PipeWire:

```bash
systemctl --user restart pipewire wireplumber
```

Verify the USB sound card appears as a sink:

```bash
wpctl status
```

Look for your USB card under **Audio → Sinks** (e.g. `Sound Blaster Play! 3 Analog Stereo`). Set it as the default output:

```bash
wpctl set-default <SINK_ID>
wpctl set-volume <SINK_ID> 1.0
```

Replace `<SINK_ID>` with the number next to your USB card in the `wpctl status` output.

Verify audio works:

```bash
speaker-test -c 2 -t wav
```

> **Note (Pi 4):** If your Pi has a 3.5mm jack, SDR audio routes there automatically and SuperCollider uses the USB card via JACK — no PipeWire configuration needed.

#### 3. Clone and set up virtual environment

```bash
git clone https://github.com/Lessnullvoid/SAR_system.git
cd SAR_system
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install pyrtlsdr sounddevice scikit-learn shapely pyproj requests python-osc
```

The `--system-site-packages` flag lets pip-installed packages coexist with the apt-installed PyQt5/numpy/scipy.

#### 4. Increase swap (safety net for peak memory)

```bash
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

#### 5. Audio group (real-time priority for audio threads)

```bash
sudo usermod -a -G audio $USER
```

Log out and back in (or reboot) for the group change to take effect.

#### 6. Run

```bash
cd ~/SAR_system
source .venv/bin/activate
python -m python_app.gui_main                                   # defaults (loop_antenna, both synths)
python -m python_app.gui_main --antenna loop_antenna             # explicit HF loop
python -m python_app.gui_main --antenna discone --synth drone    # discone + drone only
```

To change the defaults for auto-start on boot, edit the environment variables
in `scripts/sar_autostart.sh`:

```bash
SAR_ANTENNA="loop_antenna"   # or fm_broadcast, discone
SAR_SYNTH="both"             # or drone, resonator
```

The application starts in fullscreen. Press **F** to toggle fullscreen mode.

On first run, click the **Satellite** button in the map panel to download imagery tiles (~100 MB at Pi resolution). Tiles are cached in `data/sat_cache/` for subsequent runs.

#### 7. Updating

```bash
cd ~/SAR_system
git pull
```

If satellite tiles were re-downloaded at a different resolution, clear the cache:

```bash
rm ~/SAR_system/data/sat_cache/SAF_*.png
```

Then restart the app and click **Satellite** again.

#### Hardware requirements

| Component | Required | Notes |
|-----------|----------|-------|
| Raspberry Pi 5 | 8 GB RAM | Pi 4 with 4+ GB may work but is untested |
| USB sound card | Yes | e.g. Creative Sound Blaster Play! 3 |
| RTL-SDR dongle | For radio | e.g. RTL-SDR Blog V3/V4 (direct-sampling capable for HF) |
| Antenna | For radio | Active HF loop (`loop_antenna`), VHF whip (`fm_broadcast`), or broadband discone (`discone`) — set via `--antenna` flag |
| Display | HDMI | GUI requires a display (not headless) |
| Internet | Yes | Sensor APIs require network access |
| SD card | 32 GB+ | ~200 MB for app + tiles + database |

#### Monitoring resource usage

Check the app's memory and CPU in real time:

```bash
# Quick snapshot
ps aux | grep python

# Live monitoring
top -p $(pgrep -f "python_app.gui_main")

# Detailed memory breakdown
pgrep -f "python_app.gui_main" | xargs -I{} cat /proc/{}/status | grep -E "VmRSS|VmSwap"

# System-wide overview
free -h
```

Typical usage: ~1.8 GB RAM, ~70% of one CPU core (out of 4).

#### Automatic Pi optimizations

- Satellite tiles downloaded at 256px (vs 512px desktop) with smooth rendering
- Global dark stylesheet (overrides Pi desktop theme for pure black GUI)
- Batched lazy loading for news and history images to avoid GIL starvation of audio pipeline
- Narrative engine manifest generation runs once per cycle (O(N log N) then O(1) per chapter)
- Audio routed through PipeWire (Pi 5) or BCM2835 jack (Pi 4)
- SuperCollider launched via `pw-jack` for PipeWire JACK compatibility
- SuperCollider launch deferred 10s after map + SDR are stable
- Staggered sensor polling to spread network and CPU load
- News/history images loaded from local cache only (no network downloads at runtime)
- Auto-start on boot via desktop autostart entry (see [RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md))

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
| [RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md) | Complete Raspberry Pi 5 installation and configuration guide |

![S.A.R Installation](img/installation.png)

---

## License

Code: MIT (proposed)
Concept, sound, and visual outputs: Artistic license (TBD)

---

S.A.R — Seismic / Atmospheric Radio
Hybrid research instrument bridging geophysics, electromagnetism, and sound.
