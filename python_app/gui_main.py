"""
Seismo-EM SDR Console — lightweight SDR receiver GUI.

Inspired by uSDR / SDR++. Built with PyQt5 + pyqtgraph.

Architecture:
    ┌─────────────────────────────────────────────────┐
    │  SpectrumWorker (QThread)                       │
    │                                                 │
    │  RTL-SDR → raw IQ (2.4 MHz)                    │
    │    ├── FFT → spectrum_ready signal → GUI plots  │
    │    │         └──▶ SeismoMonitor (ML anomaly det)│
    │    └── channel_filter → demod → audio_output    │
    └─────────────────────────────────────────────────┘
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

# Force pyqtgraph to use PyQt5 (not PySide6 which may also be installed)
os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"

# Pi-specific Qt hints (must be set before QApplication is created)
if platform.system() == "Linux" and platform.machine().startswith(("aarch64", "arm")):
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from .audio_output import AudioOutput, find_sdr_audio_device
from .dsp_core import (
    FftConfig,
    compute_power_spectrum,
)
from .ml.monitor import SeismoMonitor
from .ml.bands import BANDS
from .ml.scanner import ScanOrchestrator, ScanPosition
from .rtl_device import RtlDevice, RtlDeviceConfig

# Channel bandwidths per mode — inlined here to avoid importing scipy
# at module level (demod.py imports scipy.signal which is slow on Pi).
MODE_CHANNEL_BW = {
    "fm": 200_000,
    "am": 10_000,
    "lsb": 3_000,
    "usb": 3_000,
}

# S.A.R geospatial map (lazy import — only fails if shapely/pyproj missing)
try:
    from sar.geo.tile_grid import build_tile_grid
    from sar.gui.map_widget import FaultMapWidget
    from sar.ingest.sensor_scheduler import SensorScheduler
    from sar.osc.bridge import OSCBridge, SuperColliderProcess, get_scd_path
    _HAS_SAR_GEO = True
except ImportError:
    _HAS_SAR_GEO = False

log = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "python_app" / "config"

# ── Platform detection ─────────────────────────────────────────────────
_IS_PI = (
    platform.system() == "Linux"
    and platform.machine().startswith(("aarch64", "arm"))
)


def load_antenna_config(antenna_key: str) -> dict:
    cfg_path = CONFIG_DIR / "antennas.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if antenna_key not in cfg:
        raise KeyError(f"Antenna '{antenna_key}' not found in antennas.json")
    return cfg[antenna_key]


# ---------------------------------------------------------------------------
# Spectrum + audio worker thread
# ---------------------------------------------------------------------------

class SpectrumWorker(QtCore.QThread):
    """
    Background thread that reads IQ from the RTL-SDR and:
      1. Computes an FFT for the spectrum/waterfall display.
      2. Channel-filters, demodulates, and plays audio.
    """

    spectrum_ready = QtCore.pyqtSignal(object, object)   # freqs_hz, power_db
    signal_level = QtCore.pyqtSignal(float)              # peak dBFS
    audio_samples = QtCore.pyqtSignal(object)            # float32 audio block
    audio_debug = QtCore.pyqtSignal(str)                 # audio stats for debug
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        antenna_key: str,
        center_freq_hz: float,
        demod_mode: str,
        gain_db: float,
        squelch_db: float,
        vfo_offset_hz: float = 0.0,
        ppm_correction: int = 0,
        audio_device: Optional[int] = None,
    ):
        super().__init__()
        self._antenna_key = antenna_key
        self._center_freq_hz = center_freq_hz
        self._demod_mode = demod_mode
        self._gain_db = gain_db
        self._squelch_db = squelch_db
        self._vfo_offset_hz = vfo_offset_hz   # VFO offset from LO center
        self._ppm_correction = ppm_correction
        self._audio_device = audio_device
        self._running = False
        self._device: Optional[RtlDevice] = None
        self._shift_phase: float = 0.0  # continuous oscillator phase (radians)
        # Pending device commands queued from the GUI thread.
        # The reader thread applies them between USB reads (thread-safe).
        self._pending_freq: Optional[float] = None
        self._pending_gain: Optional[float] = None

    # --- Thread-safe setters (called from GUI thread) ---
    # These do NOT touch the USB device directly — they queue
    # the change for the reader thread to apply between reads.

    def set_center_freq(self, hz: float) -> None:
        self._center_freq_hz = hz
        self._pending_freq = hz
        self._shift_phase = 0.0

    def set_vfo_offset(self, offset_hz: float) -> None:
        self._vfo_offset_hz = offset_hz

    def set_gain(self, db: float) -> None:
        self._gain_db = db
        self._pending_gain = db

    def set_squelch(self, db: float) -> None:
        self._squelch_db = db

    def stop(self) -> None:
        self._running = False

    # --- Main loop ---

    def run(self) -> None:
        from time import sleep
        from scipy import signal as sig
        from .demod import create_demodulator

        # Open device
        antenna_cfg = load_antenna_config(self._antenna_key)
        dev_cfg = RtlDeviceConfig(
            device_index=int(antenna_cfg["device_index"]),
            center_freq_hz=self._center_freq_hz,
            sample_rate_hz=float(antenna_cfg["default_sample_rate"]),
            gain_db=self._gain_db,
            direct_sampling_q=bool(antenna_cfg.get("use_direct_sampling_q", False)),
            ppm_correction=self._ppm_correction,
        )
        try:
            dev = RtlDevice(dev_cfg)
            dev.open()
            self._device = dev
        except Exception as exc:
            self.error.emit(str(exc))
            return

        fs = float(dev.config.sample_rate_hz)
        fft_cfg = FftConfig(fft_size=4096, window="hann")
        audio_out = AudioOutput(sample_rate=48000.0, device=self._audio_device)
        self._audio_out = audio_out  # expose for volume control

        # ── Compute decimation for EXACT 48 kHz audio ──
        # fs must be an exact multiple of 48000 (2.4 MHz / 48000 = 50).
        AUDIO_RATE = 48000.0
        total_decim = int(round(fs / AUDIO_RATE))  # 50 for 2.4 MHz
        ch_bw = float(MODE_CHANNEL_BW.get(self._demod_mode, 200_000))

        # ── Multi-stage decimation for narrow-band modes ──
        # For narrow modes (AM/SSB), the channel BW is tiny relative to
        # fs (e.g. 10kHz / 2.4MHz = wn=0.008). IIR filters at such low
        # normalised cutoffs are numerically unstable in float32.
        #
        # Solution (matching gqrx's PREF_QUAD_RATE approach):
        #   Stage 1: coarse decimate by pre_decim (e.g. 10x) with a
        #            wideband anti-alias filter at healthy wn values
        #   Stage 2: narrow channel filter at the reduced rate where
        #            the normalised cutoff is much more reasonable
        #
        # For FM (200kHz BW), single-stage is fine (wn=0.083).
        is_narrow = ch_bw < 50_000  # AM, LSB, USB

        if is_narrow:
            # Pre-decimation: decimate by 10 first (2.4MHz → 240kHz)
            # This puts the narrow filter at wn = ch_bw / (240k/2)
            # e.g. 10kHz / 120kHz = 0.083 (healthy) instead of 0.008
            pre_decim = 10
            while total_decim % pre_decim != 0 and pre_decim > 1:
                pre_decim -= 1
            intermediate_rate = fs / pre_decim

            # Remaining decimation after pre-decimation
            remaining_decim = total_decim // pre_decim

            # Channel filter at intermediate rate (normalised cutoff is healthy)
            ch_decim = 1
            max_ch = min(int(intermediate_rate / ch_bw), remaining_decim)
            for d in range(max_ch, 0, -1):
                if remaining_decim % d == 0:
                    ch_decim = d
                    break
            audio_decim = remaining_decim // ch_decim
            ch_rate = intermediate_rate / ch_decim
            actual_audio_rate = ch_rate / audio_decim

            # Pre-decimation anti-alias filter (wideband, float32 is fine)
            pre_wn = min(0.95, 0.9 / pre_decim)
            pre_sos = sig.butter(
                5, pre_wn, btype='low', output='sos'
            ).astype(np.float32)
            pre_sos_zi = sig.sosfilt_zi(pre_sos).astype(np.complex64) * 0.0
            pre_decim_phase = 0

            # Narrow channel filter at intermediate rate.
            # SOS form with wn ≈ 0.083 is numerically stable in float32.
            ch_wn = ch_bw / (intermediate_rate / 2)
            ch_sos = sig.butter(
                5, ch_wn, btype='low', output='sos'
            ).astype(np.float32)
            ch_sos_zi = sig.sosfilt_zi(ch_sos).astype(np.complex64) * 0.0
        else:
            # FM: single-stage decimation (wn is already healthy)
            pre_decim = 0  # flag: no pre-decimation
            max_ch_decim = min(int(fs / ch_bw), total_decim)
            ch_decim = 1
            for d in range(max_ch_decim, 0, -1):
                if total_decim % d == 0:
                    ch_decim = d
                    break
            audio_decim = total_decim // ch_decim
            ch_rate = fs / ch_decim
            actual_audio_rate = ch_rate / audio_decim

            ch_sos = sig.butter(
                5, ch_bw / (fs / 2), btype='low', output='sos'
            ).astype(np.float32)
            ch_sos_zi = sig.sosfilt_zi(ch_sos).astype(np.complex64) * 0.0

        decim = ch_decim

        # ── DC blocking filter (continuous across blocks, float32) ──
        # For narrow modes, DC removal happens AFTER pre-decimation
        # (at intermediate_rate) so it processes 10x fewer samples.
        # For FM, it stays at full rate (single-stage).
        dc_rate = intermediate_rate if is_narrow else fs
        dc_R = np.float32(1.0 - (np.pi * 5.0 / dc_rate))
        dc_b = np.array([1.0, -1.0], dtype=np.float32)
        dc_a = np.array([1.0, -dc_R], dtype=np.float32)
        dc_zi = sig.lfilter_zi(dc_b, dc_a).astype(np.complex64) * 0.0

        # Decimation phase tracking across blocks.
        ch_decim_phase = 0

        # ── Audio output smoothing (gentle 15 kHz LPF, removes aliasing residue) ──
        out_wn = min(0.95, 15000.0 / (AUDIO_RATE / 2))  # 15 kHz / 24 kHz
        out_sos = sig.butter(4, out_wn, output='sos').astype(np.float32)
        out_sos_zi = sig.sosfilt_zi(out_sos).astype(np.float32) * 0.0

        # Create demodulator — the demod will compute its own _audio_decim
        # from ch_rate / actual_audio_rate, which is our audio_decim (exact).
        demod = create_demodulator(self._demod_mode, ch_rate, actual_audio_rate)

        # Display throttle: only update GUI every N blocks.
        # At 2.4 MHz / 131072 = ~18.3 blocks/sec, updating every 3rd
        # block gives ~6 Hz display rate — smooth enough for eyes, but
        # frees the GIL so the audio callback can run without starvation.
        display_counter = 0
        # Pi: less frequent GUI updates = more CPU for audio pipeline
        DISPLAY_EVERY = 5 if _IS_PI else 3

        # ── Separate reader thread for USB → IQ queue ──────────────
        # read_samples() blocks for ~55 ms per 131k block. If the DSP
        # and read run sequentially, even 4 ms of DSP overhead slowly
        # drains the audio ring buffer. By reading in a dedicated
        # thread, the next USB transfer starts immediately while the
        # DSP thread processes the current block in parallel.
        import queue as _queue, threading as _thr

        iq_q: _queue.Queue = _queue.Queue(maxsize=8)
        reader_stop = _thr.Event()

        def _reader_loop() -> None:
            while not reader_stop.is_set():
                # Apply pending device commands between reads (same thread
                # as read_samples, so no concurrent USB access).
                pf = self._pending_freq
                if pf is not None:
                    self._pending_freq = None
                    try:
                        dev.set_center_frequency(pf)
                    except Exception:
                        pass
                pg = self._pending_gain
                if pg is not None:
                    self._pending_gain = None
                    try:
                        if dev._sdr is not None:
                            dev._sdr.gain = pg
                    except Exception:
                        pass
                try:
                    blk = dev.read_samples()
                    iq_q.put(blk, timeout=0.2)
                except Exception:
                    if not reader_stop.is_set():
                        continue
                    break

        reader_thread = _thr.Thread(
            target=_reader_loop, daemon=True, name="sdr-reader"
        )

        self._running = True
        reader_thread.start()
        try:
            while self._running:
                try:
                    # Pull the next IQ block from the reader thread.
                    # The reader runs concurrently, so no audio-production
                    # deficit from sequential read + DSP.
                    try:
                        samples = iq_q.get(timeout=0.2)
                    except _queue.Empty:
                        continue

                    # ━━━ AUDIO FIRST (time-critical) ━━━━━━━━━━━━━━━━━━━━━
                    # Process and deliver audio BEFORE FFT/display to minimize
                    # latency between data arrival and audio buffer write.

                    # Stay in complex64 throughout — RTL-SDR gives 8-bit I/Q,
                    # so complex64 (23-bit mantissa) is more than enough precision.
                    # This halves memory bandwidth vs complex128 → ~2x faster.
                    iq = samples  # already complex64 from rtl_device

                    #    Frequency-shift to VFO offset
                    #    CRITICAL: use float64 for phase computation!
                    #    With float32 and 131072 samples, phase accumulates
                    #    to ~17000 rad, losing ~0.03 rad precision at the
                    #    block end → periodic 18 Hz artifact.  Float64 gives
                    #    15 digits of precision → zero discontinuity.
                    vfo_off = self._vfo_offset_hz
                    if abs(vfo_off) > 0.5:
                        n = len(iq)
                        dphi = -2.0 * np.pi * vfo_off / fs  # float64
                        phases = self._shift_phase + dphi * np.arange(n, dtype=np.float64)
                        osc = np.exp(1j * phases).astype(np.complex64)
                        iq = iq * osc
                        self._shift_phase = float((phases[-1] + dphi) % (2.0 * np.pi))

                    #    Channel filter + decimation
                    if is_narrow and pre_decim > 1:
                        # ── Narrow-mode optimised path ──
                        # Order: VFO shift (done above) → pre-decimate → DC
                        # block → channel filter.  DC removal and channel
                        # filter run at intermediate_rate (10x fewer samples)
                        # saving ~10 ms per block vs full-rate DC removal.

                        # Stage 1: wideband anti-alias + coarse decimation
                        pre_filt, pre_sos_zi = sig.sosfilt(
                            pre_sos, iq, zi=pre_sos_zi
                        )
                        pre_out = pre_filt[pre_decim_phase::pre_decim]
                        n_pre = len(pre_out)
                        next_pre = pre_decim_phase + n_pre * pre_decim
                        pre_decim_phase = next_pre - len(pre_filt)

                        # DC removal at intermediate rate (13k vs 131k samples)
                        pre_out, dc_zi = sig.lfilter(
                            dc_b, dc_a, pre_out, zi=dc_zi
                        )

                        # Stage 2: narrow channel filter (complex64, fast)
                        filtered, ch_sos_zi = sig.sosfilt(
                            ch_sos, pre_out, zi=ch_sos_zi
                        )
                    else:
                        # ── FM wideband path (single-stage) ──
                        # DC removal at full rate (ok for FM — wn is healthy)
                        iq_clean, dc_zi = sig.lfilter(
                            dc_b, dc_a, iq, zi=dc_zi
                        )
                        filtered, ch_sos_zi = sig.sosfilt(
                            ch_sos, iq_clean, zi=ch_sos_zi
                        )

                    #    Decimate WITH phase tracking (continuous across blocks)
                    ch_iq = filtered[ch_decim_phase::decim]
                    n_picked = len(ch_iq)
                    next_pick = ch_decim_phase + n_picked * decim
                    ch_decim_phase = next_pick - len(filtered)

                    #    Demodulate → audio at exactly 48 kHz
                    audio = demod.demodulate(ch_iq)

                    # Output smoothing (15 kHz LPF, removes decimation aliases)
                    if len(audio) > 0:
                        audio, out_sos_zi = sig.sosfilt(
                            out_sos, audio, zi=out_sos_zi
                        )

                    # Squelch: mute if signal too weak
                    if len(audio) > 0:
                        rms_db = 20.0 * np.log10(
                            float(np.sqrt(np.mean(audio ** 2))) + 1e-12
                        )
                        if rms_db < self._squelch_db:
                            audio = np.zeros_like(audio)

                    # Write to audio ring buffer IMMEDIATELY (lowest latency)
                    audio_out.write(audio)

                    # ━━━ DISPLAY (non-time-critical, throttled) ━━━━━━━━━━
                    display_counter += 1
                    if display_counter >= DISPLAY_EVERY:
                        display_counter = 0

                        # FFT for spectrum/waterfall
                        base_freqs, power_db = compute_power_spectrum(
                            samples, sample_rate_hz=fs, cfg=fft_cfg
                        )
                        freqs = base_freqs + float(dev.config.center_freq_hz)
                        self.spectrum_ready.emit(freqs, power_db)
                        self.signal_level.emit(float(np.max(power_db)))

                        # Audio debug display
                        if len(audio) > 0:
                            self.audio_samples.emit(audio.copy())
                            a_rms = float(np.sqrt(np.mean(audio ** 2)))
                            a_peak = float(np.max(np.abs(audio)))
                            self.audio_debug.emit(
                                f"ch_rate={ch_rate:.0f} audio_decim={audio_decim} "
                                f"out={len(audio)}samp@48k | "
                                f"rms={a_rms:.4f} peak={a_peak:.4f} | "
                                f"vfo={self._vfo_offset_hz:.0f}Hz"
                            )

                except Exception as exc:
                    # Log the FULL traceback so we can diagnose hidden errors.
                    # NEVER sleep here — even 0.5s causes ~24000 samples of
                    # audio gap, which sounds like a massive dropout.
                    logging.error("DSP loop error: %s\n%s", exc, traceback.format_exc())
                    self.error.emit(str(exc))
        finally:
            reader_stop.set()
            reader_thread.join(timeout=2.0)
            audio_out.stop()
            dev.close()
            self._device = None


# ---------------------------------------------------------------------------
# Custom spectrum widget with click-to-tune and scroll
# ---------------------------------------------------------------------------

class SpectrumPlotWidget(pg.PlotWidget):
    frequency_clicked = QtCore.pyqtSignal(float)
    wheel_step = QtCore.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseEnabled(x=False, y=False)

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            vb = self.getViewBox()
            if vb is not None:
                pt = vb.mapSceneToView(ev.pos())
                self.frequency_clicked.emit(float(pt.x()))
        super().mousePressEvent(ev)

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y()
        if delta != 0:
            self.wheel_step.emit(1 if delta > 0 else -1)
            ev.accept()
        else:
            ev.ignore()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, synth_mode: str = "both"):
        super().__init__()
        self._synth_mode = synth_mode
        self.setWindowTitle("Seismo-EM SDR Console")
        self.resize(1500, 900)

        self.setStyleSheet("""
            QMainWindow, QWidget, QFrame, QSplitter {
                background-color: #000000;
                color: #c0d0e0;
            }
            QScrollBar:vertical, QScrollBar:horizontal {
                background: #060a10;
                border: none;
                width: 8px; height: 8px;
            }
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                background: #1a2a40;
                border-radius: 4px;
                min-height: 20px; min-width: 20px;
            }
            QScrollBar::add-line, QScrollBar::sub-line,
            QScrollBar::add-page, QScrollBar::sub-page {
                background: none; border: none;
            }
            QSplitter::handle { background: #0c1a2e; }
            QComboBox {
                background: #0a1a30; color: #a0b8d0;
                border: 1px solid #1a2a40; padding: 2px 4px;
                font-family: 'Helvetica Neue'; font-size: 10px;
            }
            QComboBox QAbstractItemView {
                background: #0a1a30; color: #a0b8d0;
                selection-background-color: #102a40;
            }
            QLineEdit {
                background: #060a10; color: #c0d0e0;
                border: 1px solid #1a2a40; padding: 2px 4px;
            }
            QSlider::groove:horizontal {
                background: #0c1a2e; height: 4px; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #1a3a60; width: 12px; margin: -4px 0;
                border-radius: 6px;
            }
            QLabel { background: transparent; }
        """)

        # State — LO (center) vs VFO (tuning) like SDR# / CubicSDR
        self._antenna_key = "loop_antenna"
        self._center_freq_hz = 1.0e6   # LO: RTL-SDR center frequency
        self._vfo_freq_hz = 1.0e6      # VFO: demodulator frequency (moves on click)
        self._demod_mode = "am"
        self._gain_db = 30.0
        self._squelch_db = -100.0       # disabled
        self._audio_bw_hz = 10_000.0
        self._sample_rate_hz = 2_048_000.0  # updated when worker starts
        self._worker: Optional[SpectrumWorker] = None

        # ML anomaly monitor (z_threshold=3.5 reduces false positives;
        # alert_threshold=0.6 requires a stronger signal before alerting)
        self._monitor = SeismoMonitor(alert_threshold=0.6, z_threshold=3.5)

        # Scan orchestrator (automated band scanning)
        self._scanner = ScanOrchestrator()
        self._scan_active = False
        self._scan_position_label = ""

        self._sdr_connected = False  # track SDR state

        # Auto-detect audio output devices:
        # Pi 4: SDR → bcm2835 headphones (3.5mm jack)
        # Pi 5: SDR → USB sound card (no built-in jack)
        # Desktop: SDR → system default (None)
        self._sdr_audio_device = find_sdr_audio_device()

        self._build_ui()

        # SDR starts AFTER all tiles are loaded so the heavy pixmap
        # decoding doesn't steal GIL time from the audio pipeline.
        # If there's no map (missing shapely/pyproj), start immediately.
        if self._fault_map is not None:
            self._fault_map.tiles_loaded.connect(self._on_tiles_loaded_start_sdr)
            self.signal_label.setText("Loading map tiles — SDR will start when ready…")
            self.signal_label.setStyleSheet(
                "color: #4488aa; font-size: 13px; font-family: 'Helvetica Neue';"
            )
        else:
            self._try_start_worker()
            if self._sdr_connected:
                QtCore.QTimer.singleShot(5000, self._auto_start_scanner)
            QtCore.QTimer.singleShot(10000, self._start_supercollider)

    # --- Keyboard shortcuts ---

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_F:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        else:
            super().keyPressEvent(event)

    # --- UI construction ---

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        # Top-level: horizontal splitter  [SDR panel | Map]
        # The map gets the full window height; SDR controls stay on the left.
        self._top_hsplit = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        top_layout = QtWidgets.QVBoxLayout(central)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)
        top_layout.addWidget(self._top_hsplit, 1)

        # ── Left panel: SDR controls, RF, ML/Audio, footer ──
        sdr_panel = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(sdr_panel)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self._top_hsplit.addWidget(sdr_panel)

        # ── Row 1: big frequency readout ──
        self.freq_label = QtWidgets.QLabel()
        self.freq_label.setFont(QtGui.QFont("Helvetica Neue Mono", 24))
        self.freq_label.setStyleSheet(
            "color: #00ccff; background: #000000; padding: 4px 8px;"
        )
        self.freq_label.setFixedHeight(56)
        root.addWidget(self.freq_label)

        # ── Row 2: signal meter ──
        self.signal_label = QtWidgets.QLabel("Signal: ---")
        self.signal_label.setStyleSheet("color: #a0b8d0; font-size: 13px; font-family: 'Helvetica Neue';")
        root.addWidget(self.signal_label)

        # ── Row 3: controls ──
        ctrl = QtWidgets.QHBoxLayout()
        root.addLayout(ctrl)

        # Antenna
        self.antenna_cb = QtWidgets.QComboBox()
        self.antenna_cb.addItems(["loop_antenna", "fm_broadcast", "discone"])
        self.antenna_cb.setCurrentText(self._antenna_key)

        # Frequency
        self.freq_spin = QtWidgets.QDoubleSpinBox()
        self.freq_spin.setDecimals(6)
        self.freq_spin.setRange(0.01, 2000.0)
        self.freq_spin.setSingleStep(0.1)
        self.freq_spin.setValue(self._center_freq_hz / 1e6)

        # Mode
        self.mode_cb = QtWidgets.QComboBox()
        self.mode_cb.addItems(["am", "fm", "lsb", "usb"])
        self.mode_cb.setCurrentText(self._demod_mode)

        # Gain
        self.gain_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gain_slider.setRange(0, 50)
        self.gain_slider.setValue(int(self._gain_db))
        self.gain_slider.setFixedWidth(120)
        self.gain_lbl = QtWidgets.QLabel(f"Gain: {int(self._gain_db)}dB")

        # Volume (software audio gain, 0–150%)
        self._audio_volume = 0.9
        self.vol_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.vol_slider.setRange(0, 150)
        self.vol_slider.setValue(90)
        self.vol_slider.setFixedWidth(120)
        self.vol_lbl = QtWidgets.QLabel("Vol: 90%")

        # PPM crystal correction (-100 to +100 ppm)
        self._ppm_correction = 0
        self.ppm_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.ppm_slider.setRange(-100, 100)
        self.ppm_slider.setValue(0)
        self.ppm_slider.setFixedWidth(100)
        self.ppm_lbl = QtWidgets.QLabel("PPM: 0")

        # Squelch
        self.squelch_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.squelch_slider.setRange(-120, 0)
        self.squelch_slider.setValue(int(self._squelch_db))
        self.squelch_slider.setFixedWidth(120)
        self.squelch_lbl = QtWidgets.QLabel("Squelch: Off")

        # Step
        self.step_cb = QtWidgets.QComboBox()
        for hz in [100, 1000, 5000, 10_000, 25_000, 50_000, 100_000]:
            self.step_cb.addItem(f"{hz} Hz", hz)
        self.step_cb.setCurrentIndex(1)  # 1000 Hz

        for label, widget in [
            ("Antenna:", self.antenna_cb),
            ("Freq (MHz):", self.freq_spin),
            ("Mode:", self.mode_cb),
        ]:
            ctrl.addWidget(QtWidgets.QLabel(label))
            ctrl.addWidget(widget)
            ctrl.addSpacing(12)

        ctrl.addWidget(self.gain_lbl)
        ctrl.addWidget(self.gain_slider)
        ctrl.addSpacing(8)
        ctrl.addWidget(self.vol_lbl)
        ctrl.addWidget(self.vol_slider)
        ctrl.addSpacing(8)
        ctrl.addWidget(self.ppm_lbl)
        ctrl.addWidget(self.ppm_slider)
        ctrl.addSpacing(8)
        ctrl.addWidget(self.squelch_lbl)
        ctrl.addWidget(self.squelch_slider)
        ctrl.addSpacing(8)
        ctrl.addWidget(QtWidgets.QLabel("Step:"))
        ctrl.addWidget(self.step_cb)
        ctrl.addStretch(1)

        # ── Spectrum + waterfall + audio debug ──
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)  # used by RF column

        # Spectrum plot
        self.spec_plot = SpectrumPlotWidget()
        self.spec_plot.setLabel("bottom", "Frequency (MHz)")
        self.spec_plot.setLabel("left", "dB")
        self.spec_plot.showGrid(x=True, y=True, alpha=0.15)

        # Spectrum trace line + baseline (must exist before FillBetweenItem)
        self.spec_baseline = self.spec_plot.plot(pen=None)
        self.spec_curve = self.spec_plot.plot(
            pen=pg.mkPen(color=(200, 200, 200), width=1)
        )

        # Filled spectrum (like SDR#): shaded area under the trace
        self.spec_fill = pg.FillBetweenItem(
            self.spec_baseline, self.spec_curve,
            brush=pg.mkBrush(15, 30, 60, 100),
        )
        self.spec_plot.addItem(self.spec_fill)

        # Tuning overlays
        self.tune_line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#00ccff", width=2)
        )
        self.spec_plot.addItem(self.tune_line)

        # ML anomaly marker — shows where the ML found the strongest signal
        # in an anomalous band. Dashed red line on spectrum + waterfall.
        self.anomaly_marker = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen("#00ffff", width=1, style=QtCore.Qt.DashLine),
        )
        self.anomaly_marker.setVisible(False)
        self.spec_plot.addItem(self.anomaly_marker)

        self.passband = pg.LinearRegionItem(
            orientation=pg.LinearRegionItem.Vertical,
            brush=pg.mkBrush(0, 160, 255, 20),
            movable=False,
        )
        self.passband.setZValue(-10)
        self.spec_plot.addItem(self.passband)

        # Spectrum averaging buffer
        self._spec_avg: Optional[np.ndarray] = None
        self._spec_alpha = 0.3  # EMA smoothing (0=frozen, 1=instant)

        # Waterfall
        self.wf_widget = pg.GraphicsLayoutWidget()
        self.wf_plot = self.wf_widget.addPlot()
        self.wf_img = pg.ImageItem()
        self.wf_plot.addItem(self.wf_img)
        self.wf_plot.setLabel("bottom", "Frequency (MHz)")
        self.wf_plot.setLabel("left", "Time")
        self.wf_plot.invertY(True)
        self.wf_data: Optional[np.ndarray] = None
        self.wf_tune = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen("#00ccff", width=1, style=QtCore.Qt.DashLine),
        )
        self.wf_plot.addItem(self.wf_tune)

        # ML anomaly marker on waterfall
        self.wf_anomaly_marker = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen("#00ffff", width=1, style=QtCore.Qt.DotLine),
        )
        self.wf_anomaly_marker.setVisible(False)
        self.wf_plot.addItem(self.wf_anomaly_marker)

        # SDR#-style colormap: black → deep blue → cyan → green → yellow → red → white
        # Noise floor = black, emissions = bright. Makes it easy to spot signals.
        colors = [
            (0, 0, 0),         # black        (noise floor)
            (0, 0, 60),        # very dark blue
            (0, 0, 180),       # blue         (weak signals)
            (0, 160, 220),     # cyan
            (0, 200, 0),       # green        (medium signals)
            (255, 255, 0),     # yellow
            (255, 80, 0),      # orange       (strong signals)
            (255, 0, 0),       # red
            (255, 255, 255),   # white        (very strong)
        ]
        positions = [0.0, 0.08, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0]
        cmap = pg.ColorMap(positions, colors)
        self.wf_lut = cmap.getLookupTable(nPts=256)

        # Audio debug panel (right side): waveform + spectrogram
        audio_panel = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        # Audio waveform
        self.audio_wave_plot = pg.PlotWidget(title="Audio Waveform")
        self.audio_wave_plot.setLabel("bottom", "Sample")
        self.audio_wave_plot.setLabel("left", "Amplitude")
        self.audio_wave_plot.setYRange(-1.0, 1.0)
        self.audio_wave_plot.showGrid(x=True, y=True, alpha=0.15)
        self.audio_wave_curve = self.audio_wave_plot.plot(
            pen=pg.mkPen("#00ccff", width=1)
        )

        # Audio spectrogram (time-frequency of the demodulated audio)
        self.audio_sg_widget = pg.GraphicsLayoutWidget()
        self.audio_sg_plot = self.audio_sg_widget.addPlot(title="Audio Spectrogram")
        self.audio_sg_img = pg.ImageItem()
        self.audio_sg_plot.addItem(self.audio_sg_img)
        self.audio_sg_plot.setLabel("bottom", "Frequency (kHz)")
        self.audio_sg_plot.setLabel("left", "Time")
        self.audio_sg_plot.invertY(True)
        self.audio_sg_data: Optional[np.ndarray] = None

        # Reuse the SDR# colormap for audio spectrogram
        self.audio_sg_lut = self.wf_lut

        audio_panel.addWidget(self.audio_wave_plot)
        audio_panel.addWidget(self.audio_sg_widget)
        audio_panel.setSizes([200, 300])

        # ── ML Anomaly Monitor panel ──
        ml_panel = QtWidgets.QWidget()
        ml_layout = QtWidgets.QVBoxLayout(ml_panel)
        ml_layout.setContentsMargins(4, 4, 4, 4)

        # Band status table
        ml_label = QtWidgets.QLabel("Seismo-EM Band Monitor")
        ml_label.setStyleSheet(
            "color: #00ccff; font-size: 13px; font-family: 'Helvetica Neue'; padding: 2px;"
        )
        ml_layout.addWidget(ml_label)

        self.band_table = QtWidgets.QTableWidget()
        self.band_table.setColumnCount(6)
        self.band_table.setHorizontalHeaderLabels(
            ["Band", "Score", "Peak MHz", "Z-Score", "IF Score", "Status"]
        )
        self.band_table.horizontalHeader().setStretchLastSection(True)
        self.band_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )
        self.band_table.verticalHeader().setVisible(False)
        self.band_table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        self.band_table.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
        self.band_table.setStyleSheet("""
            QTableWidget {
                background-color: #060a10;
                color: #c0d0e0;
                gridline-color: #0c1a2e;
                font-size: 11px;
                font-family: 'Helvetica Neue Mono', monospace;
            }
            QHeaderView::section {
                background-color: #0c1624;
                color: #00ccff;
                padding: 3px;
                border: 1px solid #0c1a2e;
            }
        """)
        ml_layout.addWidget(self.band_table, 1)

        # Anomaly score timeline (mini plot)
        self.anomaly_plot = pg.PlotWidget(title="Anomaly Score Timeline")
        self.anomaly_plot.setLabel("bottom", "seconds ago")
        self.anomaly_plot.setLabel("left", "Score")
        self.anomaly_plot.setYRange(0, 1.0)
        self.anomaly_plot.setXRange(-120, 0, padding=0)
        self.anomaly_plot.showGrid(x=True, y=True, alpha=0.15)
        # Threshold line — scores above this trigger alerts
        self.anomaly_plot.addLine(y=0.75, pen=pg.mkPen("#4488ff", width=1,
                                                        style=QtCore.Qt.DashLine))
        self.anomaly_plot.addLegend(
            offset=(10, 10), labelTextSize="8pt",
            brush=pg.mkBrush(6, 10, 16, 180),
            pen=pg.mkPen("#0c1a2e"),
        )
        self._anomaly_curves: dict = {}  # band_name → PlotDataItem
        self._anomaly_history: dict = {}  # band_name → deque of (time, score)
        ml_layout.addWidget(self.anomaly_plot, 1)

        # Anomaly log
        log_label = QtWidgets.QLabel("Anomaly Log")
        log_label.setStyleSheet("color: #4499ff; font-size: 12px; font-family: 'Helvetica Neue';")
        ml_layout.addWidget(log_label)

        self.anomaly_log = QtWidgets.QTextEdit()
        self.anomaly_log.setReadOnly(True)
        self.anomaly_log.setMaximumHeight(120)
        self.anomaly_log.setStyleSheet("""
            QTextEdit {
                background-color: #060a10;
                color: #80b0d0;
                font-family: 'Helvetica Neue Mono', monospace;
                font-size: 10px;
                border: 1px solid #0c1a2e;
            }
        """)
        ml_layout.addWidget(self.anomaly_log)

        # Right side: tabs for Audio Debug vs ML Monitor
        mid_tabs = self._mid_tabs = QtWidgets.QTabWidget()
        mid_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #0c1a2e; }
            QTabBar::tab {
                background: #0c1624;
                color: #a0b8d0;
                padding: 6px 16px;
                border: 1px solid #0c1a2e;
                font-family: 'Helvetica Neue';
                font-size: 11px;
            }
            QTabBar::tab:selected {
                background: #060a10;
                color: #00ccff;
                border-bottom: 2px solid #00ccff;
            }
        """)
        mid_tabs.addTab(audio_panel, "Audio")
        mid_tabs.addTab(ml_panel, "ML Monitor")

        # ── Sensor Status tab ──
        sensor_panel = QtWidgets.QWidget()
        sensor_layout = QtWidgets.QVBoxLayout(sensor_panel)
        sensor_layout.setContentsMargins(4, 4, 4, 4)
        sensor_layout.setSpacing(4)

        # ── Sensor data section — matches ML Monitor typography ──
        sensor_title = QtWidgets.QLabel("SENSOR DATA")
        sensor_title.setStyleSheet(
            "color: #00ccff; font-size: 13px; font-family: 'Helvetica Neue'; padding: 2px;"
        )
        sensor_layout.addWidget(sensor_title)

        _sensor_label_ss = (
            "color: #c0d0e0; background: #060a10; padding: 4px;"
            " border-radius: 3px; font-size: 11px;"
            " font-family: 'Helvetica Neue Mono', monospace;"
        )

        # Geomag section
        self._geomag_label = QtWidgets.QLabel(
            "Kp: --  |  Dst: -- nT  |  Status: --"
        )
        self._geomag_label.setStyleSheet(_sensor_label_ss)
        self._geomag_label.setWordWrap(True)
        sensor_layout.addWidget(self._geomag_label)

        # TEC section
        self._tec_label = QtWidgets.QLabel(
            "TEC: -- TECU  |  Delta: -- σ"
        )
        self._tec_label.setStyleSheet(_sensor_label_ss)
        sensor_layout.addWidget(self._tec_label)

        # Seismic summary
        self._seismic_label = QtWidgets.QLabel(
            "Earthquakes: -- (week)  |  -- in 24h"
        )
        self._seismic_label.setStyleSheet(_sensor_label_ss)
        sensor_layout.addWidget(self._seismic_label)

        # GNSS crustal deformation
        self._gnss_label = QtWidgets.QLabel(
            "GNSS: -- stations  |  -- mm/day"
        )
        self._gnss_label.setStyleSheet(_sensor_label_ss)
        self._gnss_label.setWordWrap(True)
        sensor_layout.addWidget(self._gnss_label)

        # Weather
        self._weather_label = QtWidgets.QLabel(
            "Weather: -- stations  |  --°C  |  -- m/s"
        )
        self._weather_label.setStyleSheet(_sensor_label_ss)
        self._weather_label.setWordWrap(True)
        sensor_layout.addWidget(self._weather_label)

        # Top tiles list
        top_tiles_title = QtWidgets.QLabel("TOP ACTIVE TILES")
        top_tiles_title.setStyleSheet(
            "color: #4499ff; font-size: 12px; font-family: 'Helvetica Neue';"
        )
        sensor_layout.addWidget(top_tiles_title)

        self._top_tiles_list = QtWidgets.QTextEdit()
        self._top_tiles_list.setReadOnly(True)
        self._top_tiles_list.setStyleSheet(
            "color: #c0d0e0; background: #060a10; border: 1px solid #0c1a2e;"
            " padding: 4px; font-size: 10px;"
            " font-family: 'Helvetica Neue Mono', monospace;"
        )
        self._top_tiles_list.setMaximumHeight(200)
        sensor_layout.addWidget(self._top_tiles_list)

        # News + History Feed
        news_title = QtWidgets.QLabel("NEWS & HISTORY — SAF CORRIDOR")
        news_title.setStyleSheet(
            "color: #00ccff; font-size: 13px; font-family: 'Helvetica Neue'; padding: 2px;"
        )
        sensor_layout.addWidget(news_title)

        self._news_summary_label = QtWidgets.QLabel(
            "News: polling GDELT..."
        )
        self._news_summary_label.setStyleSheet(
            "color: #c0d0e0; background: #060a10; padding: 4px;"
            " border-radius: 3px; font-size: 11px;"
            " font-family: 'Helvetica Neue Mono', monospace;"
        )
        self._news_summary_label.setWordWrap(True)
        sensor_layout.addWidget(self._news_summary_label)

        self._news_feed_log = QtWidgets.QTextEdit()
        self._news_feed_log.setReadOnly(True)
        self._news_feed_log.setStyleSheet(
            "color: #80b0d0; background: #060a10; border: 1px solid #0c1a2e;"
            " padding: 6px; font-size: 11px;"
            " font-family: 'Helvetica Neue', 'Segoe UI', sans-serif;"
        )
        self._news_feed_log.setMouseTracking(True)
        self._news_feed_log.installEventFilter(self)
        sensor_layout.addWidget(self._news_feed_log)
        self._history_events: list = []

        # Auto-scroll timer for news feed
        self._news_scroll_paused = False
        self._news_scroll_timer = QtCore.QTimer(self)
        self._news_scroll_timer.setInterval(120)
        self._news_scroll_timer.timeout.connect(self._auto_scroll_news_tick)
        self._news_scroll_timer.start()

        mid_tabs.addTab(sensor_panel, "Sensors")

        # ── Tab 4: Social Data (Census demographics + Indigenous territories) ──
        social_panel = QtWidgets.QWidget()
        social_layout = QtWidgets.QVBoxLayout(social_panel)
        social_layout.setContentsMargins(4, 4, 4, 4)
        social_layout.setSpacing(4)

        # Demographics section
        demo_title = QtWidgets.QLabel("DEMOGRAPHICS — SAF CORRIDOR")
        demo_title.setStyleSheet(
            "color: #00ccff; font-size: 13px; font-family: 'Helvetica Neue'; padding: 2px;"
        )
        social_layout.addWidget(demo_title)

        self._demo_summary = QtWidgets.QLabel("Census: loading ACS data...")
        self._demo_summary.setStyleSheet(
            "color: #c0d0e0; background: #060a10; padding: 4px;"
            " border-radius: 3px; font-size: 11px;"
            " font-family: 'Helvetica Neue Mono', monospace;"
        )
        self._demo_summary.setWordWrap(True)
        social_layout.addWidget(self._demo_summary)

        self._demo_table = QtWidgets.QTableWidget()
        self._demo_table.setColumnCount(5)
        self._demo_table.setHorizontalHeaderLabels(
            ["County", "Population", "Migration %", "Hispanic %", "Income $"]
        )
        self._demo_table.horizontalHeader().setStretchLastSection(True)
        self._demo_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )
        self._demo_table.verticalHeader().setVisible(False)
        self._demo_table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        self._demo_table.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
        self._demo_table.setStyleSheet("""
            QTableWidget {
                background-color: #060a10;
                color: #c0d0e0;
                gridline-color: #0c1a2e;
                font-size: 11px;
                font-family: 'Helvetica Neue Mono', monospace;
            }
            QHeaderView::section {
                background-color: #0c1624;
                color: #00ccff;
                padding: 3px;
                border: 1px solid #0c1a2e;
            }
        """)
        social_layout.addWidget(self._demo_table, 1)

        # Indigenous territories section
        terr_title = QtWidgets.QLabel("INDIGENOUS TERRITORIES")
        terr_title.setStyleSheet(
            "color: #00ccff; font-size: 13px; font-family: 'Helvetica Neue'; padding: 2px;"
        )
        social_layout.addWidget(terr_title)

        terr_attrib = QtWidgets.QLabel(
            "Data: Native Land Digital — native-land.ca  (CC BY-NC-SA)"
        )
        terr_attrib.setStyleSheet(
            "color: #607080; font-size: 9px; font-family: 'Helvetica Neue'; padding: 1px;"
        )
        social_layout.addWidget(terr_attrib)

        self._territory_log = QtWidgets.QTextEdit()
        self._territory_log.setReadOnly(True)
        self._territory_log.setStyleSheet(
            "color: #80b0d0; background: #060a10; border: 1px solid #0c1a2e;"
            " padding: 6px; font-size: 11px;"
            " font-family: 'Helvetica Neue', 'Segoe UI', sans-serif;"
        )
        self._territory_log.setMouseTracking(True)
        self._territory_log.installEventFilter(self)
        social_layout.addWidget(self._territory_log, 1)

        # Auto-scroll timer for territory log
        self._terr_scroll_paused = False
        self._terr_scroll_timer = QtCore.QTimer(self)
        self._terr_scroll_timer.setInterval(120)
        self._terr_scroll_timer.timeout.connect(self._auto_scroll_terr_tick)
        self._terr_scroll_timer.start()

        # SCEDC seismic section
        scedc_title = QtWidgets.QLabel("SCEDC — SO. CALIFORNIA SEISMICITY")
        scedc_title.setStyleSheet(
            "color: #00ccff; font-size: 13px; font-family: 'Helvetica Neue'; padding: 2px;"
        )
        social_layout.addWidget(scedc_title)

        self._scedc_label = QtWidgets.QLabel("SCEDC: loading...")
        self._scedc_label.setStyleSheet(
            "color: #c0d0e0; background: #060a10; padding: 4px;"
            " border-radius: 3px; font-size: 11px;"
            " font-family: 'Helvetica Neue Mono', monospace;"
        )
        self._scedc_label.setWordWrap(True)
        social_layout.addWidget(self._scedc_label)

        mid_tabs.addTab(social_panel, "Social")

        # ── Fault Map (S.A.R geospatial) — always-visible right column ──
        if _HAS_SAR_GEO:
            self._sar_tiles = build_tile_grid(tile_km=10.0, buffer_km=25.0, full_bbox=True)
            self._fault_map = FaultMapWidget(
                self._sar_tiles,
                site_lonlat=(-117.35, 34.30),  # Cajon Pass area
            )
            log.info("S.A.R Fault Map loaded: %d tiles", len(self._sar_tiles))

            # ── Sensor scheduler (USGS earthquakes, etc.) ──
            self._sensor_scheduler = SensorScheduler(
                self._sar_tiles, usgs_interval_s=120.0
            )
            self._sensor_scheduler.earthquakes_updated.connect(
                self._fault_map.update_earthquakes
            )
            self._sensor_scheduler.tiles_updated.connect(
                self._on_sensor_tiles_updated
            )
            self._sensor_scheduler.geomag_updated.connect(
                self._on_geomag_updated
            )
            self._sensor_scheduler.status_message.connect(
                lambda msg: log.info("Sensor: %s", msg)
            )
            self._sensor_scheduler.earthquakes_updated.connect(
                self._on_earthquakes_display
            )
            self._sensor_scheduler.gnss_updated.connect(
                self._on_gnss_updated
            )
            self._sensor_scheduler.weather_updated.connect(
                self._on_weather_updated
            )
            self._sensor_scheduler.news_updated.connect(
                self._on_news_updated
            )
            self._sensor_scheduler.history_updated.connect(
                self._on_history_loaded
            )
            self._sensor_scheduler.social_updated.connect(
                self._on_social_updated
            )
            self._sensor_scheduler.start()

            # ── OSC Bridge → SuperCollider drone ──
            self._osc_bridge = OSCBridge(
                host="127.0.0.1", port=57120, enabled=True
            )
            self._drone_kp = 0.0
            self._drone_dst = 0.0
            self._sc_process = SuperColliderProcess(
                scd_path=get_scd_path(self._synth_mode)
            )

            # ── Map scanner → auto-switch tabs ──
            self._fault_map._scanner.scanning_tile.connect(
                self._on_map_scanning_tile
            )
        else:
            self._fault_map = None
            self._sensor_scheduler = None
            self._sar_tiles = []

        # ── 2-column SDR layout: RF | ML/Audio ──
        sdr_hsplit = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Column 1: RF display (spectrum + waterfall)
        rf_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        rf_splitter.addWidget(self.spec_plot)
        rf_splitter.addWidget(self.wf_widget)
        rf_splitter.setSizes([250, 550])

        # Column 2: ML data + audio (tabs)
        sdr_hsplit.addWidget(rf_splitter)
        sdr_hsplit.addWidget(mid_tabs)
        sdr_hsplit.setStretchFactor(0, 3)   # RF gets 3/5 of SDR panel
        sdr_hsplit.setStretchFactor(1, 2)   # ML/Audio gets 2/5
        root.addWidget(sdr_hsplit, 1)

        # ── Right panel: Fault map (full window height) ──
        if self._fault_map:
            self._top_hsplit.addWidget(self._fault_map)
            screen = QtWidgets.QApplication.primaryScreen()
            sw = screen.size().width() if screen else 1500
            map_w = int(sw * 0.48)
            sdr_w = sw - map_w
            sdr_panel.setMaximumWidth(sdr_w)
            self._fault_map.setMinimumWidth(map_w)
            self._top_hsplit.setStretchFactor(0, 0)  # SDR: fixed
            self._top_hsplit.setStretchFactor(1, 1)  # Map: takes remaining
            self._top_hsplit.setSizes([sdr_w, map_w])
        else:
            self._top_hsplit.setSizes([1000])

        # ── Row 5: footer — SDR controls + manual tune + scan controls ──
        footer = QtWidgets.QHBoxLayout()
        root.addLayout(footer)

        # SDR connect/disconnect
        _btn_style = (
            "QPushButton { background: #0a1a30; color: #a0b8d0; "
            "padding: 4px 10px; border: 1px solid #1a2a40; border-radius: 3px; "
            "font-family: 'Helvetica Neue'; font-size: 10px; }"
            "QPushButton:hover { background: #102a40; color: #00ccff; }"
            "QPushButton:disabled { color: #303848; border-color: #101820; }"
        )
        self.btn_sdr_stop = QtWidgets.QPushButton("Stop")
        self.btn_sdr_stop.setStyleSheet(_btn_style)
        self.btn_sdr_stop.setToolTip("Disconnect RTL-SDR")
        self.btn_sdr_stop.clicked.connect(self._stop_worker)

        self.btn_sdr_start = QtWidgets.QPushButton("iStart")
        self.btn_sdr_start.setStyleSheet(_btn_style)
        self.btn_sdr_start.setToolTip("Connect RTL-SDR")
        self.btn_sdr_start.clicked.connect(self._try_start_worker)

        footer.addWidget(self.btn_sdr_stop)
        footer.addWidget(self.btn_sdr_start)

        self.btn_down = QtWidgets.QPushButton("-Step")
        self.btn_up = QtWidgets.QPushButton("+Step")
        self.freq_entry = QtWidgets.QLineEdit()
        self.freq_entry.setPlaceholderText("Enter MHz")
        self.freq_entry.setFixedWidth(140)
        self.btn_go = QtWidgets.QPushButton("Tune")

        for w in [self.btn_down, self.btn_up, self.freq_entry, self.btn_go]:
            footer.addWidget(w)

        # Separator
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.VLine)
        sep.setStyleSheet("color: #0c1a2e;")
        footer.addWidget(sep)

        # Scan controls
        self.btn_scan = QtWidgets.QPushButton("Start Scan")
        self.btn_scan.setStyleSheet("""
            QPushButton {
                background: #0a1a30; color: #00ccff;
                padding: 4px 16px;
                border: 1px solid #00ccff; border-radius: 3px;
                font-family: 'Helvetica Neue'; font-size: 11px;
            }
            QPushButton:hover { background: #102a40; }
            QPushButton:checked {
                background: #0a2040; color: #4499ff;
                border-color: #4499ff;
            }
        """)
        self.btn_scan.setCheckable(True)
        footer.addWidget(self.btn_scan)

        self.scan_status_label = QtWidgets.QLabel("Scan: Idle")
        self.scan_status_label.setStyleSheet(
            "color: #506880; font-size: 11px; font-family: 'Helvetica Neue'; padding: 0 8px;"
        )
        footer.addWidget(self.scan_status_label)

        self.scan_progress = QtWidgets.QProgressBar()
        self.scan_progress.setFixedWidth(120)
        self.scan_progress.setFixedHeight(18)
        self.scan_progress.setTextVisible(True)
        self.scan_progress.setStyleSheet("""
            QProgressBar {
                background: #060a10; border: 1px solid #0c1a2e;
                border-radius: 3px; color: #00ccff;
                font-size: 10px; text-align: center;
                font-family: 'Helvetica Neue Mono';
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #001a40, stop:1 #00ccff);
                border-radius: 2px;
            }
        """)
        self.scan_progress.setRange(0, 100)
        self.scan_progress.setValue(0)
        footer.addWidget(self.scan_progress)

        self.scan_dwell_label = QtWidgets.QLabel("")
        self.scan_dwell_label.setStyleSheet(
            "color: #708898; font-size: 10px; font-family: 'Helvetica Neue Mono', monospace;"
        )
        footer.addWidget(self.scan_dwell_label)

        footer.addStretch(1)

        # ── Connections ──
        self.freq_spin.valueChanged.connect(self._on_freq_changed)
        self.mode_cb.currentTextChanged.connect(self._on_mode_changed)
        self.antenna_cb.currentTextChanged.connect(self._on_antenna_changed)
        self.gain_slider.valueChanged.connect(self._on_gain_changed)
        self.vol_slider.valueChanged.connect(self._on_volume_changed)
        self.ppm_slider.valueChanged.connect(self._on_ppm_changed)
        self.squelch_slider.valueChanged.connect(self._on_squelch_changed)
        self.spec_plot.frequency_clicked.connect(self._on_plot_click)
        self.spec_plot.wheel_step.connect(self._on_wheel)
        self.btn_up.clicked.connect(lambda: self._step(+1))
        self.btn_down.clicked.connect(lambda: self._step(-1))
        self.btn_go.clicked.connect(self._on_tune_entry)
        self.freq_entry.returnPressed.connect(self._on_tune_entry)

        # ML monitor connections
        self._monitor.anomaly_detected.connect(self._on_anomaly_detected)
        self._monitor.band_update.connect(self._on_band_update)
        self._monitor.status_message.connect(
            lambda msg: self.statusBar().showMessage(msg)
        )

        # Scan orchestrator connections
        self.btn_scan.toggled.connect(self._on_scan_toggled)
        self._scanner.tune_requested.connect(self._on_scan_tune)
        self._scanner.progress.connect(self._on_scan_progress)
        self._scanner.dwell_tick.connect(self._on_scan_dwell_tick)
        self._scanner.state_changed.connect(self._on_scan_state)
        self._scanner.cycle_complete.connect(
            lambda: self.scan_status_label.setText("Scan: cycle complete ✓")
        )
        # When ML detects anomaly during scan → extend dwell
        self._monitor.anomaly_detected.connect(self._on_anomaly_extend_dwell)

        self._refresh_labels()

        # ── Start auto-tab rotation ──
        self._init_tab_rotation()

    # --- Worker management ---

    def _on_tiles_loaded_start_sdr(self):
        """Called once all tile pixmaps are loaded — safe to start SDR now.

        Disconnects itself so tile reloads don't restart the SDR.
        After the SDR starts, the radio scanner auto-starts after a short
        delay to let the audio pipeline stabilize.
        """
        try:
            self._fault_map.tiles_loaded.disconnect(self._on_tiles_loaded_start_sdr)
        except TypeError:
            pass

        log.info("Tiles loaded — starting SDR audio pipeline")
        self._try_start_worker()

        if self._sdr_connected:
            QtCore.QTimer.singleShot(5000, self._auto_start_scanner)

        # Defer SuperCollider launch so JACK + sclang don't compete with
        # tile loading and SDR startup for memory and CPU.
        QtCore.QTimer.singleShot(10000, self._start_supercollider)

    def _start_supercollider(self):
        """Launch SuperCollider after tiles and SDR are up."""
        if hasattr(self, "_sc_process") and self._sc_process is not None:
            log.info("Launching SuperCollider (deferred)")
            self._sc_process.start()

    def _try_start_worker(self):
        """Attempt to start the SDR worker.  If the RTL-SDR is not
        connected the app continues running with map + sensors only."""
        try:
            self._start_worker()
        except Exception as exc:
            log.warning("SDR not available: %s — running in map-only mode", exc)
            self._sdr_connected = False
            self._update_sdr_status()

    def _start_worker(self):
        if self._worker is not None:
            self._worker.stop()
            if not self._worker.wait(5000):
                log.error("Old worker did not stop in 5 s — will be abandoned")
        # Look up sample rate for the current antenna
        try:
            ant_cfg = load_antenna_config(self._antenna_key)
            self._sample_rate_hz = float(ant_cfg["default_sample_rate"])
        except Exception:
            self._sample_rate_hz = 2_048_000.0

        vfo_offset = self._vfo_freq_hz - self._center_freq_hz
        self._worker = SpectrumWorker(
            antenna_key=self._antenna_key,
            center_freq_hz=self._center_freq_hz,
            demod_mode=self._demod_mode,
            gain_db=self._gain_db,
            squelch_db=self._squelch_db,
            vfo_offset_hz=vfo_offset,
            ppm_correction=self._ppm_correction,
            audio_device=self._sdr_audio_device,
        )
        self._worker.spectrum_ready.connect(self._update_spectrum)
        self._worker.signal_level.connect(self._update_signal)
        self._worker.audio_samples.connect(self._update_audio_display)
        self._worker.audio_debug.connect(self._update_audio_debug)
        self._worker.error.connect(self._on_sdr_error)
        self._worker.start()
        # Elevate DSP thread priority so audio processing beats GUI rendering
        self._worker.setPriority(QtCore.QThread.HighestPriority)
        # Push the current volume setting to the new AudioOutput instance
        QtCore.QTimer.singleShot(200, self._sync_volume_to_worker)
        self._sdr_connected = True
        self._update_sdr_status()

    def _stop_worker(self):
        """Stop the SDR worker gracefully (with timeout to prevent freeze)."""
        if self._worker is not None:
            self._worker.stop()
            if not self._worker.wait(5000):
                log.error("Worker did not stop in 5 s — will be abandoned")
            self._worker = None
        self._sdr_connected = False
        self._update_sdr_status()

    def _restart_worker(self):
        """Restart the SDR worker (e.g. after PPM change)."""
        if self._sdr_connected:
            self._stop_worker()
            self._try_start_worker()

    def _on_sdr_error(self, msg: str):
        """Handle SDR device error — app continues in map-only mode."""
        log.warning("SDR error: %s", msg)
        self._sdr_connected = False
        self._update_sdr_status()
        self._show_error(msg)

    def _update_sdr_status(self):
        """Update the SDR status indicators in the footer."""
        if self._sdr_connected:
            self.btn_sdr_stop.setEnabled(True)
            self.btn_sdr_start.setEnabled(False)
            self.signal_label.setText("Signal: SDR connected")
            self.signal_label.setStyleSheet(
                "color: #00ccaa; font-size: 13px; font-family: 'Helvetica Neue';"
            )
        else:
            self.btn_sdr_stop.setEnabled(False)
            self.btn_sdr_start.setEnabled(True)
            self.signal_label.setText(
                "SDR offline — map + sensors active"
            )
            self.signal_label.setStyleSheet(
                "color: #506880; font-size: 13px; font-family: 'Helvetica Neue';"
            )

    # --- Slot handlers ---

    def _set_vfo(self, freq_hz: float):
        """
        Move the VFO (demodulator) to freq_hz.
        Only re-center the LO when the VFO goes fully outside the visible
        bandwidth (e.g. from spinbox, step, or wheel). Clicking on the
        spectrum can never exceed the visible range, so it won't recenter.
        """
        self._vfo_freq_hz = freq_hz

        # Only recenter when VFO is truly outside the visible bandwidth
        half_bw = self._sample_rate_hz / 2.0
        offset = self._vfo_freq_hz - self._center_freq_hz

        if abs(offset) > half_bw * 0.95:
            # Shift LO so VFO sits at 25% from center (keeps context visible)
            shift_dir = 1.0 if offset > 0 else -1.0
            self._center_freq_hz = self._vfo_freq_hz - shift_dir * half_bw * 0.25
            if self._worker is not None:
                self._worker.set_center_freq(self._center_freq_hz)
            offset = self._vfo_freq_hz - self._center_freq_hz

        # Tell the worker where the VFO is relative to LO
        if self._worker is not None:
            self._worker.set_vfo_offset(offset)

        # Update the spinbox without retriggering (block signals)
        self.freq_spin.blockSignals(True)
        self.freq_spin.setValue(self._vfo_freq_hz / 1e6)
        self.freq_spin.blockSignals(False)

        self._refresh_labels()

    def _on_freq_changed(self, mhz: float):
        """Spinbox changed → move VFO to that frequency."""
        self._set_vfo(mhz * 1e6)

    def _on_mode_changed(self, mode: str):
        self._demod_mode = mode
        bw = MODE_CHANNEL_BW.get(mode, 10_000)
        self._audio_bw_hz = float(bw)

        # Auto-switch antenna for mode
        if mode == "fm" and self._antenna_key == "loop_antenna":
            self.antenna_cb.blockSignals(True)
            self.antenna_cb.setCurrentText("fm_broadcast")
            self.antenna_cb.blockSignals(False)
            self._antenna_key = "fm_broadcast"
            self._center_freq_hz = 100.0e6
            self._vfo_freq_hz = 100.0e6
            self.freq_spin.blockSignals(True)
            self.freq_spin.setValue(100.0)
            self.freq_spin.blockSignals(False)
        elif mode == "am" and self._antenna_key in ("fm_broadcast", "discone"):
            self.antenna_cb.blockSignals(True)
            self.antenna_cb.setCurrentText("loop_antenna")
            self.antenna_cb.blockSignals(False)
            self._antenna_key = "loop_antenna"
            self._center_freq_hz = 1.0e6
            self._vfo_freq_hz = 1.0e6
            self.freq_spin.blockSignals(True)
            self.freq_spin.setValue(1.0)
            self.freq_spin.blockSignals(False)

        self._refresh_labels()
        self._try_start_worker()  # restart with new demod + channel filter

    def _on_antenna_changed(self, key: str):
        self._antenna_key = key
        self._try_start_worker()

    def _on_gain_changed(self, val: int):
        self._gain_db = float(val)
        self.gain_lbl.setText(f"Gain: {val}dB")
        if self._worker is not None:
            self._worker.set_gain(self._gain_db)

    def _sync_volume_to_worker(self):
        """Push the stored volume level to the running AudioOutput."""
        if self._worker is not None and hasattr(self._worker, "_audio_out"):
            ao = self._worker._audio_out
            if ao is not None:
                ao.volume = self._audio_volume

    def _on_volume_changed(self, val: int):
        vol = val / 100.0
        self._audio_volume = vol
        self.vol_lbl.setText(f"Vol: {val}%")
        self._sync_volume_to_worker()

    def _on_ppm_changed(self, val: int):
        self._ppm_correction = val
        self.ppm_lbl.setText(f"PPM: {val:+d}")
        # Debounce: restart worker 500ms after the last slider change.
        # Dragging from 0→50 no longer triggers 50 restarts.
        if not hasattr(self, "_ppm_debounce_timer"):
            self._ppm_debounce_timer = QtCore.QTimer(self)
            self._ppm_debounce_timer.setSingleShot(True)
            self._ppm_debounce_timer.timeout.connect(self._ppm_apply)
        self._ppm_debounce_timer.start(500)

    def _ppm_apply(self):
        if self._worker is not None:
            self._restart_worker()

    def _on_squelch_changed(self, val: int):
        self._squelch_db = float(val)
        if val <= -100:
            self.squelch_lbl.setText("Squelch: Off")
        else:
            self.squelch_lbl.setText(f"Squelch: {val}dB")
        if self._worker is not None:
            self._worker.set_squelch(self._squelch_db)

    def _on_plot_click(self, freq_mhz: float):
        """Click on spectrum → move VFO to clicked frequency."""
        self._set_vfo(freq_mhz * 1e6)

    def _on_wheel(self, direction: int):
        step_hz = float(self.step_cb.currentData() or 1000)
        self._set_vfo(self._vfo_freq_hz + direction * step_hz)

    def _step(self, direction: int):
        step_hz = float(self.step_cb.currentData() or 1000)
        self._set_vfo(self._vfo_freq_hz + direction * step_hz)

    def _on_tune_entry(self):
        text = self.freq_entry.text().strip()
        if not text:
            return
        try:
            self._set_vfo(float(text) * 1e6)
        except ValueError:
            pass

    # --- Scan orchestrator handlers ---

    def _auto_start_scanner(self):
        """Auto-start the radio scanner after the SDR pipeline is stable."""
        if self._scan_active:
            return  # already running (user started it manually)
        if not self._sdr_connected:
            return  # SDR failed to start — don't scan
        log.info("Auto-starting radio scanner")
        self.btn_scan.setChecked(True)  # triggers _on_scan_toggled

    def _on_anomaly_extend_dwell(self, result):
        """When ML fires during a scan, extend the dwell at current position."""
        if self._scan_active:
            self._scanner.anomaly_extend(result.band_name)

    def _on_scan_toggled(self, checked: bool):
        """Start / stop the automated scan."""
        if checked:
            self.btn_scan.setText("Stop Scan")
            self._scan_active = True
            self._scanner.start()
        else:
            self.btn_scan.setText("Start Scan")
            self._scan_active = False
            self._scan_position_label = ""
            self._scanner.stop()
            self.scan_status_label.setText("Scan: Idle")
            self.scan_progress.setValue(0)
            self.scan_dwell_label.setText("")
            self._refresh_labels()

    @QtCore.pyqtSlot(object)
    def _on_scan_tune(self, pos: ScanPosition):
        """Scanner requests tuning to a new position.

        This handles antenna switching (requires worker restart) and
        simple frequency changes (fast, no restart).  In both cases the
        display is reset so the waterfall and spectrum show the NEW band
        cleanly, and the audio demodulates the new signal immediately.
        """
        needs_restart = False

        # Antenna change?
        if pos.antenna != self._antenna_key:
            self._antenna_key = pos.antenna
            self.antenna_cb.blockSignals(True)
            self.antenna_cb.setCurrentText(pos.antenna)
            self.antenna_cb.blockSignals(False)
            needs_restart = True

        # Mode change?
        if pos.mode != self._demod_mode:
            self._demod_mode = pos.mode
            bw = MODE_CHANNEL_BW.get(pos.mode, 10_000)
            self._audio_bw_hz = float(bw)
            self.mode_cb.blockSignals(True)
            self.mode_cb.setCurrentText(pos.mode)
            self.mode_cb.blockSignals(False)
            needs_restart = True

        # Frequency
        self._center_freq_hz = pos.center_hz
        self._vfo_freq_hz = pos.center_hz
        self.freq_spin.blockSignals(True)
        self.freq_spin.setValue(pos.center_hz / 1e6)
        self.freq_spin.blockSignals(False)

        if needs_restart:
            self._try_start_worker()
        else:
            if self._worker is not None:
                self._worker.set_center_freq(pos.center_hz)
                self._worker.set_vfo_offset(0.0)

        # ── Reset all displays so new band starts clean ──
        # Waterfall: clear old rows (they were at a different frequency)
        self.wf_data = None
        # Spectrum averaging: reset so the first frame is instant, not blended
        self._spec_avg = None
        # Audio spectrogram: clear old audio data
        self.audio_sg_data = None
        # Hide anomaly markers (they pointed at the old frequency)
        self.anomaly_marker.setVisible(False)
        self.wf_anomaly_marker.setVisible(False)

        # Store scan label for the frequency readout
        self._scan_position_label = pos.label

        self._refresh_labels()
        self.scan_status_label.setText(f"Scan: {pos.label}")

    @QtCore.pyqtSlot(int, int)
    def _on_scan_progress(self, idx: int, total: int):
        """Update the scan progress bar."""
        pct = int((idx / max(total, 1)) * 100)
        self.scan_progress.setValue(pct)
        self.scan_progress.setFormat(f"{idx + 1}/{total}")

    @QtCore.pyqtSlot(float, float)
    def _on_scan_dwell_tick(self, elapsed: float, total: float):
        """Update the dwell countdown display."""
        remaining = max(0, total - elapsed)
        self.scan_dwell_label.setText(
            f"{int(elapsed)}s / {int(total)}s  ({int(remaining)}s left)"
        )

    @QtCore.pyqtSlot(str)
    def _on_scan_state(self, state: str):
        """Update scan status label based on orchestrator state."""
        pos = self._scanner.current_position
        label = pos.label if pos else "—"
        _scan_font = "font-size: 11px; font-family: 'Helvetica Neue'; padding: 0 8px;"
        if state == "scanning":
            self.scan_status_label.setText(f"Scan: {label}")
            self.scan_status_label.setStyleSheet(
                f"color: #00ccff; {_scan_font}"
            )
        elif state == "dwelling":
            self.scan_status_label.setText(f"Dwelling: {label}")
            self.scan_status_label.setStyleSheet(
                f"color: #00ccff; {_scan_font}"
            )
        elif state == "extended":
            self.scan_status_label.setText(f"EXTENDED: {label}")
            self.scan_status_label.setStyleSheet(
                f"color: #4499ff; {_scan_font}"
            )
        elif state == "paused":
            self.scan_status_label.setText(f"Scan: PAUSED ({label})")
            self.scan_status_label.setStyleSheet(
                f"color: #00cc99; {_scan_font}"
            )
        elif state == "stopped":
            self.scan_status_label.setText("Scan: Idle")
            self.scan_status_label.setStyleSheet(
                f"color: #506880; {_scan_font}"
            )

    # --- Display updates ---

    def _refresh_labels(self):
        vfo_mhz = self._vfo_freq_hz / 1e6
        mode = self._demod_mode.upper()
        bw_khz = self._audio_bw_hz / 1e3
        # When scanning, show the scan position label so the user
        # knows which band they're hearing
        scan_info = ""
        if self._scan_active and hasattr(self, "_scan_position_label"):
            scan_info = f"  [{self._scan_position_label}]"
        self.freq_label.setText(
            f" {vfo_mhz:.6f} MHz  {mode}  BW {bw_khz:.0f} kHz{scan_info}"
        )

    @QtCore.pyqtSlot(float)
    def _update_signal(self, peak_dbfs: float):
        if peak_dbfs > -40:
            color = "#00ccff"
            strength = "Strong"
        elif peak_dbfs > -70:
            color = "#80b8e0"
            strength = "Medium"
        else:
            color = "#304858"
            strength = "Weak"
        self.signal_label.setText(
            f"Signal: {peak_dbfs:+.1f} dBFS ({strength})"
        )
        self.signal_label.setStyleSheet(
            f"color: {color}; font-size: 13px; font-family: 'Helvetica Neue';"
        )

    @QtCore.pyqtSlot(object, object)
    def _update_spectrum(self, freqs_hz: np.ndarray, power_db: np.ndarray):
        if len(freqs_hz) < 2:
            return

        # Guard: skip if monitor already closed (queued signals during shutdown)
        if not hasattr(self, "_monitor") or self._monitor is None:
            return

        # Feed spectrum to ML monitor for anomaly detection
        try:
            self._monitor.on_spectrum(
                freqs_hz, power_db,
                center_hz=self._center_freq_hz,
                sample_rate_hz=self._sample_rate_hz,
            )
        except (AttributeError, RuntimeError, Exception):
            return  # monitor closed during shutdown (catches sqlite3.ProgrammingError too)

        # Convert to MHz for display
        freqs_mhz = freqs_hz / 1e6
        f_min, f_max = float(freqs_mhz[0]), float(freqs_mhz[-1])

        # Spectrum averaging (EMA smoothing like SDR#)
        if self._spec_avg is None or len(self._spec_avg) != len(power_db):
            self._spec_avg = power_db.copy()
        else:
            a = self._spec_alpha
            self._spec_avg = a * power_db + (1.0 - a) * self._spec_avg

        # Auto-range Y axis based on actual signal levels
        noise_floor = float(np.percentile(self._spec_avg, 10))
        peak = float(np.max(self._spec_avg))
        y_min = max(-140.0, noise_floor - 10)
        y_max = min(20.0, peak + 15)

        self.spec_plot.setXRange(f_min, f_max, padding=0)
        self.spec_plot.setYRange(y_min, y_max, padding=0)

        # Update spectrum trace + filled area underneath (SDR# style)
        self.spec_curve.setData(freqs_mhz, self._spec_avg)
        baseline = np.full_like(self._spec_avg, y_min)
        self.spec_baseline.setData(freqs_mhz, baseline)
        self.spec_fill.setCurves(self.spec_baseline, self.spec_curve)

        # Tuning overlays at VFO position (not LO center)
        vfo_mhz = self._vfo_freq_hz / 1e6
        half_bw_mhz = self._audio_bw_hz / 2e6
        self.tune_line.setPos(vfo_mhz)
        self.passband.setRegion((vfo_mhz - half_bw_mhz, vfo_mhz + half_bw_mhz))
        self.wf_tune.setPos(vfo_mhz)

        # ── Waterfall ──
        max_rows = 256 if _IS_PI else 512
        row = self._spec_avg.astype(np.float32)[np.newaxis, :]
        if self.wf_data is None:
            self.wf_data = np.repeat(row, max_rows, axis=0)
        else:
            self.wf_data = np.roll(self.wf_data, 1, axis=0)
            self.wf_data[0, :] = row

        # SDR#-style leveling: noise floor → black, signals → bright colors.
        # Use a wider range biased toward the noise floor so that only
        # actual emissions get bright colors.  This makes signals "pop"
        # against a dark background (the key for identifying emissions).
        wf_min = noise_floor
        wf_range = max(peak - noise_floor, 1.0)
        # Stretch: map noise_floor to 0, noise_floor + range*1.2 to 1
        # (the 1.2 gives a bit of headroom so strong signals saturate to white)
        wf_max = wf_min + wf_range * 1.2

        # Normalize to [0, 255] for LUT indexing
        normalized = (self.wf_data - wf_min) / (wf_max - wf_min)
        normalized = np.clip(normalized, 0.0, 1.0)
        indices = (normalized * 255).astype(np.uint8)

        # Apply SDR# colormap LUT → RGB image
        rgb = self.wf_lut[indices]

        # Transpose so that:
        #   X axis (horizontal) = frequency bins
        #   Y axis (vertical)   = time rows (newest at top, scrolls down)
        # pyqtgraph ImageItem maps dim-0 → X, dim-1 → Y
        rgb_t = rgb.transpose(1, 0, 2)  # (freq, time, 3)

        self.wf_img.setImage(rgb_t)

        width = f_max - f_min if f_max > f_min else 1.0
        n_rows = float(self.wf_data.shape[0])
        self.wf_img.setRect(
            QtCore.QRectF(f_min, 0, width, n_rows)
        )

    @QtCore.pyqtSlot(object)
    def _update_audio_display(self, audio: np.ndarray):
        """Update audio waveform and spectrogram panels."""
        if len(audio) < 64:
            return

        # ── Waveform: show the latest block ──
        self.audio_wave_curve.setData(audio[:2048])
        self.audio_wave_plot.setYRange(
            max(-1.0, float(np.min(audio)) - 0.05),
            min(1.0, float(np.max(audio)) + 0.05),
        )

        # ── Audio spectrogram (short-time FFT) ──
        NFFT = 256
        if len(audio) >= NFFT:
            # Compute one FFT column from the center of the block
            seg = audio[len(audio) // 2 - NFFT // 2: len(audio) // 2 + NFFT // 2]
            w = np.hanning(NFFT)
            spec = np.fft.rfft(seg * w)
            power = 20.0 * np.log10(np.abs(spec) + 1e-12)

            max_sg_rows = 200
            n_bins = len(power)

            if self.audio_sg_data is None or self.audio_sg_data.shape[1] != n_bins:
                self.audio_sg_data = np.full((max_sg_rows, n_bins), -80.0, dtype=np.float32)

            # Scroll: new row at top
            self.audio_sg_data = np.roll(self.audio_sg_data, 1, axis=0)
            self.audio_sg_data[0, :] = power.astype(np.float32)

            # Fixed-floor normalization: noise always stays dark,
            # only real signal features pop out as bright colors.
            # Auto-level caused solid color bands when AGC-amplified
            # noise filled the buffer with uniform spectral power.
            SG_FLOOR_DB = -60.0
            SG_CEIL_MIN_DB = -10.0
            sg_ceil = max(float(np.max(self.audio_sg_data)), SG_CEIL_MIN_DB)
            dyn_range = sg_ceil - SG_FLOOR_DB
            norm = (self.audio_sg_data - SG_FLOOR_DB) / dyn_range
            norm = np.clip(norm, 0.0, 1.0)
            indices = (norm * 255).astype(np.uint8)
            rgb = self.audio_sg_lut[indices]

            # Transpose: freq on X, time on Y
            rgb_t = rgb.transpose(1, 0, 2)
            self.audio_sg_img.setImage(rgb_t)

            # X axis in kHz (0 to Nyquist = 24 kHz for 48k sample rate)
            f_max_khz = 24.0
            self.audio_sg_img.setRect(
                QtCore.QRectF(0, 0, f_max_khz, float(max_sg_rows))
            )

    @QtCore.pyqtSlot(str)
    def _update_audio_debug(self, info: str):
        self.statusBar().showMessage(info)

    # --- ML anomaly handlers ---

    @QtCore.pyqtSlot(str, object)
    def _on_band_update(self, band_name: str, summary: dict):
        """Update the band status table and anomaly timeline."""
        from collections import deque

        # Update anomaly history for timeline plot
        if band_name not in self._anomaly_history:
            self._anomaly_history[band_name] = deque(maxlen=300)
        self._anomaly_history[band_name].append(
            (summary["timestamp"], summary["composite"])
        )

        # Update band table row
        self._update_band_table_row(band_name, summary)

        # Update anomaly timeline plot (every 5th update to save CPU)
        if hasattr(self, "_ml_plot_counter"):
            self._ml_plot_counter += 1
        else:
            self._ml_plot_counter = 0
        if self._ml_plot_counter % 5 == 0:
            self._update_anomaly_plot()

        # Bridge RF anomaly scores to fault map tiles (every 10th update)
        if self._fault_map is not None and self._ml_plot_counter % 10 == 0:
            self._update_fault_map_rf()

    def _update_band_table_row(self, band_name: str, summary: dict):
        """Add or update a row in the band status table."""
        table = self.band_table
        # Find existing row or create new
        row = -1
        for r in range(table.rowCount()):
            item = table.item(r, 0)
            if item and item.text() == band_name:
                row = r
                break
        if row < 0:
            row = table.rowCount()
            table.insertRow(row)

        composite = summary["composite"]
        z_score = summary["z_composite"]
        if_score = summary["if_score"]
        is_anomaly = summary["is_anomaly"]
        triggered = summary["triggered"]
        peak_mhz = summary.get("peak_freq_hz", 0) / 1e6

        # Color based on anomaly score (cold palette only)
        if composite > 0.7:
            color = "#00ffff"
            status = "ALERT"
        elif composite > 0.5:
            color = "#4499ff"
            status = "WARNING"
        elif composite > 0.3:
            color = "#00cc99"
            status = "ELEVATED"
        else:
            color = "#506880"
            status = "NORMAL"

        items = [
            band_name,
            f"{composite:.2f}",
            f"{peak_mhz:.4f}",
            f"{z_score:.2f}",
            f"{if_score:.2f}",
            status,
        ]
        for col, text in enumerate(items):
            item = QtWidgets.QTableWidgetItem(text)
            item.setForeground(QtGui.QColor(color))
            if col == 0:
                item.setFont(QtGui.QFont("Helvetica Neue Mono", 10))
            table.setItem(row, col, item)

    def _update_anomaly_plot(self):
        """Redraw the anomaly score timeline.

        Shows the last 120 seconds of anomaly scores with a fixed X range
        so the plot stays readable instead of auto-scaling to span the
        entire session history.
        """
        colors = [
            "#00ccff", "#4499ff", "#00ff88", "#00cc99",
            "#88bbff", "#00ffff", "#3377cc", "#66ddaa",
        ]
        WINDOW_S = 120.0    # show last 2 minutes
        now = time.time()

        for i, (band_name, history) in enumerate(self._anomaly_history.items()):
            if len(history) < 2:
                continue
            times = np.array([h[0] for h in history])
            scores = np.array([h[1] for h in history])
            age = now - times  # seconds ago (positive)

            # Only keep data within the display window
            mask = age <= WINDOW_S
            if mask.sum() < 2:
                # Not enough recent data — clear the curve
                if band_name in self._anomaly_curves:
                    self._anomaly_curves[band_name].setData([], [])
                continue

            x = -age[mask]          # negative = past, 0 = now
            y = scores[mask]

            color = colors[i % len(colors)]
            if band_name not in self._anomaly_curves:
                curve = self.anomaly_plot.plot(
                    pen=pg.mkPen(color, width=1), name=band_name
                )
                self._anomaly_curves[band_name] = curve

            self._anomaly_curves[band_name].setData(x, y)

        # Fixed X range so the plot doesn't jump around
        self.anomaly_plot.setXRange(-WINDOW_S, 0, padding=0)

    def _update_fault_map_rf(self) -> None:
        """Bridge ML anomaly scores into the fault map tile colours.

        Until we have per-tile sensor assignment (Phase 3+), we compute
        an overall RF anomaly level from all tracked bands and apply it
        to the tiles nearest the antenna site.  Tiles further away get
        a diminished score (distance falloff).
        """
        if self._fault_map is None:
            return

        # Average the latest composite scores across all ML-tracked bands
        latest = self._monitor.get_all_latest()
        if not latest:
            return
        avg_score = sum(r.composite for r in latest.values()) / len(latest)

        # Apply to tiles with distance falloff from site
        # Site is at ~(−117.35, 34.30) in the default config
        site_lat = 34.30
        for tile in self._sar_tiles:
            # Simple latitude-based distance (approximate)
            dlat = abs(tile.centroid_lonlat[1] - site_lat)
            # Falloff: full score within 0.5 deg, fading to 0 at 3 deg
            falloff = max(0.0, 1.0 - dlat / 3.0)
            tile_score = avg_score * falloff * falloff  # quadratic falloff
            self._fault_map.update_tile(tile.tile_id, tile_score)

    @QtCore.pyqtSlot(object)
    def _on_sensor_tiles_updated(self, scores: dict) -> None:
        """Merge seismic sensor scores into the map.

        Uses update_all_tiles which also updates the map scanner queue.
        """
        if self._fault_map is None:
            return

        self._fault_map.update_all_tiles(scores)

        # Update top tiles in sensor panel
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        lines = []
        for tid, sc in top:
            if sc > 0.001:
                bar = "█" * int(sc * 20)
                lines.append(f"  {tid:>12s}  {sc:.3f}  {bar}")
        if lines:
            self._top_tiles_list.setPlainText("\n".join(lines))

        # ── Send synth state to SuperCollider (drone + resonator) ──
        if hasattr(self, "_osc_bridge") and self._osc_bridge.enabled:
            max_sc = max(scores.values()) if scores else 0.0
            self._drone_last_max_score = max_sc
            total_ev = getattr(self, "_drone_eq_count_24h", 0)
            if self._synth_mode in ("both", "drone"):
                self._osc_bridge.send_drone_state(
                    max_score=max_sc,
                    kp=self._drone_kp,
                    dst=self._drone_dst,
                    total_events=total_ev,
                )
            if self._synth_mode in ("both", "resonator"):
                self._osc_bridge.send_resonator_state(
                    max_score=max_sc,
                    kp=self._drone_kp,
                    dst=self._drone_dst,
                    total_events=total_ev,
                )

    @QtCore.pyqtSlot(object)
    def _on_geomag_updated(self, info: dict) -> None:
        """Update geomag/TEC labels in the Sensors tab."""
        kp = info.get("kp", 0)
        dst = info.get("dst", 0)
        storm = info.get("storm_level", "unknown")
        tec = info.get("tec", 0)
        tec_d = info.get("tec_delta", 0)

        # Cache for synth OSC and send immediately
        self._drone_kp = kp
        self._drone_dst = dst
        if hasattr(self, "_osc_bridge") and self._osc_bridge.enabled:
            _ms = getattr(self, "_drone_last_max_score", 0.0)
            _ev = getattr(self, "_drone_eq_count_24h", 0)
            if self._synth_mode in ("both", "drone"):
                self._osc_bridge.send_drone_state(
                    max_score=_ms, kp=kp, dst=dst, total_events=_ev,
                )
            if self._synth_mode in ("both", "resonator"):
                self._osc_bridge.send_resonator_state(
                    max_score=_ms, kp=kp, dst=dst, total_events=_ev,
                )

        # Color code by storm level
        storm_colors = {
            "quiet": "#506880",
            "unsettled": "#00cc99",
            "minor": "#4499ff",
            "moderate": "#00ccff",
            "severe": "#00ffff",
        }
        sc = storm_colors.get(storm, "#a0b8d0")

        self._geomag_label.setText(
            f"Kp: {kp:.1f}  |  Dst: {dst:.0f} nT  |  "
            f"<span style='color:{sc}'>{storm.upper()}</span>"
        )

        # TEC colour based on delta
        tc = "#00ffff" if abs(tec_d) > 2 else "#a0b8d0"
        self._tec_label.setText(
            f"TEC: {tec:.1f} TECU  |  "
            f"<span style='color:{tc}'>Delta: {tec_d:+.1f}σ</span>"
        )

    @QtCore.pyqtSlot(object)
    def _on_earthquakes_display(self, quakes) -> None:
        """Update seismic summary label in Sensors tab."""
        total = len(quakes) if quakes else 0
        recent_24h = sum(1 for q in quakes if q.age_hours <= 24.0) if quakes else 0
        recent_1h = sum(1 for q in quakes if q.age_hours <= 1.0) if quakes else 0
        max_mag = max((q.mag for q in quakes), default=0.0) if quakes else 0.0

        # Cache for drone sonification
        self._drone_eq_count_24h = recent_24h
        self._drone_eq_max_mag = max_mag

        self._seismic_label.setText(
            f"Earthquakes: {total} (week)  |  {recent_24h} in 24h  |  {recent_1h} in 1h\n"
            f"Max magnitude: M{max_mag:.1f}"
        )

    @QtCore.pyqtSlot(object)
    def _on_gnss_updated(self, summary: dict) -> None:
        """Update GNSS label in the Sensors tab."""
        sc = summary.get("station_count", 0)
        avg = summary.get("avg_mm_day", 0)
        mx = summary.get("max_mm_day", 0)
        mx_sta = summary.get("max_station", "—")

        # Color: cyan if displacement is above normal (~0.1 mm/day typical)
        col = "#00ffff" if mx > 0.3 else "#a0b8d0"
        self._gnss_label.setText(
            f"GNSS: {sc} stations  |  avg {avg:.3f} mm/day  |  "
            f"<span style='color:{col}'>max {mx:.3f} mm/day ({mx_sta})</span>"
        )

    @QtCore.pyqtSlot(object)
    def _on_weather_updated(self, summary: dict) -> None:
        """Update Weather label in the Sensors tab."""
        sc = summary.get("station_count", 0)
        t = summary.get("temperature", {})
        w = summary.get("wind", {})
        p = summary.get("pressure", {})
        h = summary.get("humidity", {})

        t_avg = t.get("avg", 0)
        w_avg = w.get("avg", 0)
        p_avg = p.get("avg", 0)
        h_avg = h.get("avg", 0)

        # Color: highlight high winds (>10 m/s) or extreme pressure
        wc = "#00ffff" if w_avg > 10 else "#a0b8d0"
        self._weather_label.setText(
            f"Weather: {sc} stations  |  {t_avg:.1f}°C  |  "
            f"<span style='color:{wc}'>{w_avg:.1f} m/s</span>  |  "
            f"{p_avg:.0f} hPa  |  {h_avg:.0f}% RH"
        )

    @QtCore.pyqtSlot(object)
    @QtCore.pyqtSlot(object)
    def _on_history_loaded(self, events: list) -> None:
        """Receive static historical events — store and push to map + feed."""
        self._history_events = list(events)
        if self._fault_map:
            self._fault_map.update_history_events(events)
        self._rebuild_news_feed()

    def _on_news_updated(self, news_by_tile: dict) -> None:
        """Update news feed display and push articles to the map."""
        if self._fault_map:
            self._fault_map.update_news(news_by_tile)

        self._latest_news_by_tile = news_by_tile
        self._rebuild_news_feed()

    @staticmethod
    def _normalize_title(title: str) -> str:
        import re
        return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()

    def _rebuild_news_feed(self) -> None:
        """Merge GDELT news + historical events into a single feed."""
        news_by_tile = getattr(self, "_latest_news_by_tile", {})

        raw_articles = []
        for articles in news_by_tile.values():
            raw_articles.extend(articles)

        # Deduplicate: first by URL, then by title prefix (first 50 chars)
        by_url = {}
        for a in raw_articles:
            if a.url not in by_url:
                by_url[a.url] = a
        seen_prefixes: set = set()
        all_articles = []
        for a in by_url.values():
            norm = self._normalize_title(a.title)[:50]
            if norm in seen_prefixes:
                continue
            seen_prefixes.add(norm)
            all_articles.append(a)

        total_news = len(all_articles)
        total_history = len(self._history_events)
        with_img = sum(1 for a in all_articles if a.local_image_path)
        hist_img = sum(1 for e in self._history_events if e.image_path)

        self._news_summary_label.setText(
            f"News: <span style='color:#00ffff'>{total_news}</span> articles  |  "
            f"History: <span style='color:#dcaa1e'>{total_history}</span> events  |  "
            f"{with_img + hist_img} images"
        )

        _box_news = (
            'margin:0 0 6px 0; padding:8px 10px; '
            'border-left:4px solid #00ccff; '
            'background:#0a1520; '
            'border-radius:3px;'
        )
        _box_hist = (
            'margin:0 0 6px 0; padding:8px 10px; '
            'border-left:4px solid #dcaa1e; '
            'background:#12100a; '
            'border-radius:3px;'
        )

        html_parts = []

        # GDELT news (newest first, up to 10)
        if all_articles:
            html_parts.append(
                '<div style="color:#00ccff; font-size:14px; font-weight:bold; '
                'margin:6px 0 4px 0; letter-spacing:1px;">RECENT NEWS</div>'
            )
            sorted_articles = sorted(
                all_articles, key=lambda a: a.date, reverse=True
            )
            for a in sorted_articles[:10]:
                city_tag = f"<b>{a.city}</b>" if a.city else ""
                img_tag = ' <span style="color:#00ffaa">[IMG]</span>' if a.local_image_path else ""
                date_short = a.date[:10] if len(a.date) >= 10 else a.date
                html_parts.append(
                    f'<div style="{_box_news}">'
                    f'<div style="font-size:13px; margin-bottom:3px;">'
                    f'<span style="color:#00ccff">{date_short}</span> '
                    f'<span style="color:#40a0d0">{city_tag}</span>'
                    f'{img_tag}</div>'
                    f'<div style="font-size:13px; color:#d0e0f0; line-height:1.3;">'
                    f'{a.title[:120]}</div>'
                    f'</div>'
                )

        # Historical events (sorted by date descending)
        if self._history_events:
            html_parts.append(
                '<div style="color:#dcaa1e; font-size:14px; font-weight:bold; '
                'margin:10px 0 4px 0; letter-spacing:1px;">SOCIAL HISTORY — SAF CORRIDOR</div>'
            )
            sorted_history = sorted(
                self._history_events, key=lambda e: e.date, reverse=True
            )
            for ev in sorted_history:
                img_tag = ' <span style="color:#dcaa1e">[IMG]</span>' if ev.image_path else ""
                themes = " ".join(
                    f'<span style="color:#b08820; font-size:11px;">[{t.strip()}]</span>'
                    for t in ev.theme.split(",") if t.strip()
                )
                html_parts.append(
                    f'<div style="{_box_hist}">'
                    f'<div style="font-size:13px; margin-bottom:2px;">'
                    f'<span style="color:#dcaa1e">{ev.date[:10]}</span> '
                    f'<span style="color:#b09040"><b>{ev.city}</b></span> '
                    f'<span style="color:#8a7030; font-size:11px;">{ev.period}</span>'
                    f'{img_tag}</div>'
                    f'<div style="font-size:13px; color:#d0e0f0; line-height:1.3; '
                    f'margin-bottom:3px;">{ev.title}</div>'
                    f'<div style="font-size:12px; color:#8a9aaa; line-height:1.3;">'
                    f'{ev.description[:200]}</div>'
                    f'<div style="margin-top:3px;">{themes}</div>'
                    f'</div>'
                )

        self._news_feed_log.setHtml("".join(html_parts))

    def _on_social_updated(self, payload: dict) -> None:
        """Populate the Social tab with Census, territory, and SCEDC data."""
        # ── Census demographics table ──
        census = payload.get("census", [])
        if census:
            total_pop = sum(c.population for c in census)
            avg_mig = sum(c.migration_pct for c in census) / len(census) if census else 0
            self._demo_summary.setText(
                f"<span style='color:#00ffff'>{len(census)}</span> counties  |  "
                f"Pop: <span style='color:#00ffff'>{total_pop:,}</span>  |  "
                f"Avg migration: <span style='color:#00ffff'>{avg_mig:.1f}%</span>"
            )

            self._demo_table.setRowCount(len(census))
            # Sort by population descending
            census_sorted = sorted(census, key=lambda c: c.population, reverse=True)
            for i, c in enumerate(census_sorted):
                hisp_pct = (c.hispanic_pop / c.population * 100) if c.population > 0 else 0
                self._demo_table.setItem(i, 0, QtWidgets.QTableWidgetItem(c.name))
                self._demo_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{c.population:,}"))
                self._demo_table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{c.migration_pct:.1f}%"))
                self._demo_table.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{hisp_pct:.1f}%"))
                self._demo_table.setItem(i, 4, QtWidgets.QTableWidgetItem(f"${c.median_income:,}"))

                # Color-code migration % (higher = warmer color)
                for col in range(5):
                    item = self._demo_table.item(i, col)
                    if item:
                        if c.migration_pct > 5.0:
                            item.setForeground(QtGui.QColor(0, 255, 200))
                        elif c.migration_pct > 3.0:
                            item.setForeground(QtGui.QColor(0, 200, 255))
                        else:
                            item.setForeground(QtGui.QColor(160, 200, 220))

        # ── Indigenous territories ──
        territories = payload.get("territories", [])
        if territories:
            _box_terr = (
                'margin:0 0 6px 0; padding:8px 10px; '
                'border-left:4px solid #00ffaa; '
                'background:#0a1510; '
                'border-radius:3px;'
            )
            html_parts = []
            regions = {"southern": [], "central": [], "northern": []}
            for t in territories:
                regions.get(t.region, []).append(t)

            for region_name in ["southern", "central", "northern"]:
                terr_list = regions.get(region_name, [])
                if not terr_list:
                    continue
                html_parts.append(
                    f'<div style="color:#00ccff; font-size:14px; font-weight:bold; '
                    f'margin:8px 0 4px 0; letter-spacing:1px;">'
                    f'{region_name.upper()} SECTION</div>'
                )
                for t in terr_list:
                    html_parts.append(
                        f'<div style="{_box_terr}">'
                        f'<div style="font-size:13px; margin-bottom:3px;">'
                        f'<span style="color:#00ffaa"><b>{t.name}</b></span> — '
                        f'<span style="color:#c0d0e0">{t.people}</span></div>'
                        f'<div style="font-size:12px; color:#8a9aaa; line-height:1.3;">'
                        f'{t.description}</div>'
                        f'</div>'
                    )
            self._territory_log.setHtml("".join(html_parts))

        # ── SCEDC seismic data ──
        scedc = payload.get("scedc", [])
        if scedc:
            max_mag = max((e.magnitude for e in scedc), default=0)
            avg_depth = sum(e.depth_km for e in scedc) / len(scedc) if scedc else 0
            self._scedc_label.setText(
                f"Events: <span style='color:#00ffff'>{len(scedc)}</span> (7 days, M>=2.0)  |  "
                f"Max: <span style='color:#00ffff'>M{max_mag:.1f}</span>  |  "
                f"Avg depth: <span style='color:#00ffff'>{avg_depth:.1f} km</span>"
            )
        else:
            self._scedc_label.setText("SCEDC: no events or service unavailable")

    # ── Auto-tab navigation (ML-driven) ─────────────────────────────
    #
    # Cycles through all 4 tabs so the user can read each one.
    # Default rotation: ML Monitor → Sensors → Social → Audio → repeat
    # Each tab is shown for a configurable duration (seconds).
    # Anomaly detection can interrupt and jump to ML tab immediately.
    # News/seismic tiles scroll the feed when Sensors tab is active.

    _TAB_AUDIO = 0
    _TAB_ML = 1
    _TAB_SENSORS = 2
    _TAB_SOCIAL = 3

    # Seconds each tab is displayed before rotating
    _TAB_DURATIONS = {
        0: 8,    # Audio: 8s  (compact, quick glance)
        1: 15,   # ML Monitor: 15s  (read band table + anomaly log)
        2: 25,   # Sensors: 25s  (news feed has many articles to scan)
        3: 20,   # Social: 20s  (demographics table + territories list)
    }

    def _init_tab_rotation(self) -> None:
        """Set up the automatic tab rotation timer."""
        self._tab_rotation_order = [
            self._TAB_ML, self._TAB_SENSORS, self._TAB_SOCIAL, self._TAB_AUDIO
        ]
        self._tab_rotation_idx = 0
        self._tab_rotation_timer = QtCore.QTimer(self)
        self._tab_rotation_timer.setSingleShot(True)
        self._tab_rotation_timer.timeout.connect(self._rotate_tab)
        # Start on ML Monitor
        self._mid_tabs.setCurrentIndex(self._TAB_ML)
        self._tab_rotation_timer.start(
            self._TAB_DURATIONS[self._TAB_ML] * 1000
        )

    def _rotate_tab(self) -> None:
        """Advance to the next tab in the rotation."""
        if not hasattr(self, '_mid_tabs'):
            return
        self._tab_rotation_idx = (
            (self._tab_rotation_idx + 1) % len(self._tab_rotation_order)
        )
        tab = self._tab_rotation_order[self._tab_rotation_idx]
        self._mid_tabs.setCurrentIndex(tab)

        # If showing Sensors, scroll news to the currently scanned tile
        if tab == self._TAB_SENSORS:
            self._scroll_news_to_current_tile()

        duration = self._TAB_DURATIONS.get(tab, 8)
        self._tab_rotation_timer.start(duration * 1000)

    def _scroll_news_to_current_tile(self) -> None:
        """Scroll the news feed to match the tile the scanner is visiting."""
        if not self._fault_map:
            return
        scanner = self._fault_map._scanner
        tile_id = scanner._current_tile_id
        if not tile_id or tile_id not in scanner._news_tile_ids:
            return
        item = self._fault_map._tile_items.get(tile_id)
        if item and item._news_articles:
            city = item._news_articles[0].city
            if city:
                cursor = self._news_feed_log.document().find(city)
                if not cursor.isNull():
                    self._news_feed_log.setTextCursor(cursor)
                    self._news_feed_log.ensureCursorVisible()

    def _auto_scroll_news_tick(self) -> None:
        """Advance the news feed scroll by 1 px; wrap to top at the bottom."""
        if self._news_scroll_paused:
            return
        sb = self._news_feed_log.verticalScrollBar()
        if sb.value() >= sb.maximum():
            self._news_scroll_paused = True
            QtCore.QTimer.singleShot(3000, self._news_scroll_resume_top)
            return
        sb.setValue(sb.value() + 1)

    def _news_scroll_resume_top(self) -> None:
        sb = self._news_feed_log.verticalScrollBar()
        sb.setValue(0)
        self._news_scroll_paused = False

    def _auto_scroll_terr_tick(self) -> None:
        """Advance the territory log scroll by 1 px; wrap to top at the bottom."""
        if self._terr_scroll_paused:
            return
        sb = self._territory_log.verticalScrollBar()
        if sb.value() >= sb.maximum():
            self._terr_scroll_paused = True
            QtCore.QTimer.singleShot(3000, self._terr_scroll_resume_top)
            return
        sb.setValue(sb.value() + 1)

    def _terr_scroll_resume_top(self) -> None:
        sb = self._territory_log.verticalScrollBar()
        sb.setValue(0)
        self._terr_scroll_paused = False

    def eventFilter(self, obj, event) -> bool:
        if obj is self._news_feed_log:
            if event.type() == QtCore.QEvent.Enter:
                self._news_scroll_paused = True
            elif event.type() == QtCore.QEvent.Leave:
                self._news_scroll_paused = False
        elif obj is self._territory_log:
            if event.type() == QtCore.QEvent.Enter:
                self._terr_scroll_paused = True
            elif event.type() == QtCore.QEvent.Leave:
                self._terr_scroll_paused = False
        return super().eventFilter(obj, event)

    @QtCore.pyqtSlot(str)
    def _on_map_scanning_tile(self, tile_id: str) -> None:
        """When the map visits a news tile, scroll the feed if Sensors is active."""
        if not hasattr(self, '_mid_tabs'):
            return
        if self._mid_tabs.currentIndex() == self._TAB_SENSORS:
            self._scroll_news_to_current_tile()

    def _switch_to_ml_tab(self) -> None:
        """Interrupt rotation and jump to ML tab for a significant anomaly."""
        if not hasattr(self, '_mid_tabs'):
            return
        # Jump to ML tab and reset rotation from there
        self._mid_tabs.setCurrentIndex(self._TAB_ML)
        # Find ML's position in the rotation so it continues naturally
        try:
            self._tab_rotation_idx = self._tab_rotation_order.index(self._TAB_ML)
        except ValueError:
            self._tab_rotation_idx = 0
        # Restart timer with ML duration
        self._tab_rotation_timer.start(
            self._TAB_DURATIONS[self._TAB_ML] * 1000
        )

    @QtCore.pyqtSlot(object)
    def _on_anomaly_detected(self, result):
        """Log anomaly and auto-tune VFO to the anomalous signal.

        When scanning, this moves the VFO to the peak frequency within
        the anomalous band so the user hears the signal and sees the
        tune line pointing at it on the spectrum/waterfall.
        """
        from datetime import datetime

        peak_mhz = result.peak_freq_hz / 1e6 if result.peak_freq_hz > 0 else 0

        # ── Auto-switch to ML tab on significant anomalies only ──
        if result.composite >= 0.6:
            self._switch_to_ml_tab()

        # ── Auto-tune VFO to the anomaly's peak frequency ──
        # Only when scanning AND the peak is within our current spectrum window
        if self._scan_active and result.peak_freq_hz > 0:
            half_bw = self._sample_rate_hz / 2.0
            lo = self._center_freq_hz - half_bw
            hi = self._center_freq_hz + half_bw
            if lo <= result.peak_freq_hz <= hi:
                # Move VFO offset — don't change center freq (scanner owns that)
                self._vfo_freq_hz = result.peak_freq_hz
                offset = result.peak_freq_hz - self._center_freq_hz
                if self._worker is not None:
                    self._worker.set_vfo_offset(offset)
                # Update display labels + tune line position
                self._refresh_labels()

        # ── Show anomaly marker on spectrum + waterfall ──
        if result.peak_freq_hz > 0:
            marker_mhz = result.peak_freq_hz / 1e6
            self.anomaly_marker.setPos(marker_mhz)
            self.anomaly_marker.setVisible(True)
            self.wf_anomaly_marker.setPos(marker_mhz)
            self.wf_anomaly_marker.setVisible(True)
        else:
            self.anomaly_marker.setVisible(False)
            self.wf_anomaly_marker.setVisible(False)

        # ── Log to anomaly panel ──
        ts = datetime.fromtimestamp(result.timestamp).strftime("%H:%M:%S")
        triggered = ", ".join(result.triggered_features) if result.triggered_features else "multivariate"
        freq_str = f" @ {peak_mhz:.4f} MHz" if peak_mhz > 0 else ""
        msg = (
            f"[{ts}] <b style='color:#00ffff'>{result.band_name}</b>"
            f"{freq_str} "
            f"score={result.composite:.2f} "
            f"(Z={result.z_composite:.2f} IF={result.if_score:.2f}) "
            f"→ {triggered}"
        )
        self.anomaly_log.append(msg)
        # Keep log from growing too large
        if self.anomaly_log.document().lineCount() > 200:
            cursor = self.anomaly_log.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.movePosition(cursor.Down, cursor.KeepAnchor, 50)
            cursor.removeSelectedText()

    @QtCore.pyqtSlot(str)
    def _show_error(self, msg: str):
        # Only show once (avoid dialog flood)
        self.signal_label.setText(f"Error: {msg}")
        self.signal_label.setStyleSheet("color: #4499ff; font-size: 13px; font-family: 'Helvetica Neue';")

    def closeEvent(self, ev):
        log.info("Shutting down S.A.R system...")

        # Stop timers first to prevent callbacks into destroyed widgets
        if hasattr(self, "_tab_rotation_timer"):
            self._tab_rotation_timer.stop()

        # Stop sensor scheduler (stops QTimers + prevents new bg threads)
        if hasattr(self, "_sensor_scheduler") and self._sensor_scheduler is not None:
            self._sensor_scheduler.stop()

        # Stop SDR worker and wait for it to finish
        if self._worker is not None:
            self._worker.stop()
            if not self._worker.wait(5000):
                log.error("Worker did not stop in 5 s — will be abandoned")
            self._worker = None

        # Close ML data store only after worker is fully stopped
        if hasattr(self, "_monitor") and self._monitor is not None:
            self._monitor.close()
            self._monitor = None

        # Close OSC bridge (releases UDP socket)
        if hasattr(self, "_osc_bridge") and self._osc_bridge is not None:
            self._osc_bridge.close()

        # Stop SuperCollider process
        if hasattr(self, "_sc_process") and self._sc_process is not None:
            self._sc_process.stop()

        log.info("S.A.R shutdown complete.")
        super().closeEvent(ev)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="S.A.R. — Seismic Activity & Radio")
    parser.add_argument(
        "--synth",
        choices=["both", "drone", "resonator"],
        default="both",
        help="Which SuperCollider synth(s) to launch: both, drone, or resonator (default: both)",
    )
    args, remaining = parser.parse_known_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    pg.setConfigOption("background", "#000000")    # black like SDR#
    pg.setConfigOption("foreground", "#a0b8d0")

    sys.argv = sys.argv[:1] + remaining
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette — cold blue-black tones matching the waterfall
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#060a10"))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#c0d0e0"))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#080c14"))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#060a10"))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor("#c0d0e0"))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor("#0c1624"))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#c0d0e0"))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#00ccff"))
    app.setPalette(palette)

    log.info("Synth mode: %s", args.synth)
    win = MainWindow(synth_mode=args.synth)
    win.showFullScreen()

    # ── Graceful Ctrl+C / SIGTERM shutdown ──
    # Qt's event loop blocks Python signal delivery, so we use a small
    # timer to let Python process pending signals periodically.
    def _sigint_handler(*_args):
        log.info("SIGINT received — shutting down gracefully...")
        win.close()

    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    # Timer fires every 200ms, giving Python a chance to run the handler
    _sig_timer = QtCore.QTimer()
    _sig_timer.timeout.connect(lambda: None)
    _sig_timer.start(200)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
