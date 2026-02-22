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

import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from .audio_output import AudioOutput
from .demod import MODE_CHANNEL_BW, create_demodulator, Demodulator
from .dsp_core import (
    FftConfig,
    compute_power_spectrum,
)
from .ml.monitor import SeismoMonitor
from .ml.bands import BANDS
from .ml.scanner import ScanOrchestrator, ScanPosition
from .rtl_device import RtlDevice, RtlDeviceConfig

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "python_app" / "config"


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
    ):
        super().__init__()
        self._antenna_key = antenna_key
        self._center_freq_hz = center_freq_hz
        self._demod_mode = demod_mode
        self._gain_db = gain_db
        self._squelch_db = squelch_db
        self._vfo_offset_hz = vfo_offset_hz   # VFO offset from LO center
        self._running = False
        self._device: Optional[RtlDevice] = None
        self._shift_phase: float = 0.0  # continuous oscillator phase (radians)

    # --- Thread-safe setters (called from GUI thread) ---

    def set_center_freq(self, hz: float) -> None:
        self._center_freq_hz = hz
        if self._device is not None:
            try:
                self._device.set_center_frequency(hz)
            except Exception:
                pass
        # Reset oscillator phase so VFO shift doesn't carry stale state
        self._shift_phase = 0.0

    def set_vfo_offset(self, offset_hz: float) -> None:
        """Set the VFO offset from center (for click-to-tune within bandwidth)."""
        self._vfo_offset_hz = offset_hz

    def set_gain(self, db: float) -> None:
        self._gain_db = db
        if self._device is not None and self._device._sdr is not None:
            try:
                self._device._sdr.gain = db
            except Exception:
                pass

    def set_squelch(self, db: float) -> None:
        self._squelch_db = db

    def stop(self) -> None:
        self._running = False

    # --- Main loop ---

    def run(self) -> None:
        from time import sleep
        from scipy import signal as sig

        # Open device
        antenna_cfg = load_antenna_config(self._antenna_key)
        dev_cfg = RtlDeviceConfig(
            device_index=int(antenna_cfg["device_index"]),
            center_freq_hz=self._center_freq_hz,
            sample_rate_hz=float(antenna_cfg["default_sample_rate"]),
            gain_db=self._gain_db,
            direct_sampling_q=bool(antenna_cfg.get("use_direct_sampling_q", False)),
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
        audio_out = AudioOutput(sample_rate=48000.0)

        # ── Compute decimation for EXACT 48 kHz audio ──
        # fs must be an exact multiple of 48000 (2.4 MHz / 48000 = 50).
        AUDIO_RATE = 48000.0
        total_decim = int(round(fs / AUDIO_RATE))  # 50 for 2.4 MHz
        ch_bw = float(MODE_CHANNEL_BW.get(self._demod_mode, 200_000))

        # Channel decimation: largest factor of total_decim that keeps
        # ch_rate >= ch_bw. This guarantees ch_decim * audio_decim = total_decim,
        # so the final audio rate is EXACTLY 48000 Hz — no resampling needed.
        max_ch_decim = min(int(fs / ch_bw), total_decim)
        ch_decim = 1
        for d in range(max_ch_decim, 0, -1):
            if total_decim % d == 0:
                ch_decim = d
                break
        audio_decim = total_decim // ch_decim
        ch_rate = fs / ch_decim
        actual_audio_rate = ch_rate / audio_decim  # exactly 48000.0

        decim = ch_decim

        # ── DC blocking filter (continuous across blocks, float32) ──
        # IIR DC blocker: y[n] = x[n] - x[n-1] + R·y[n-1], R ≈ 1
        dc_R = np.float32(1.0 - (np.pi * 5.0 / fs))
        dc_b = np.array([1.0, -1.0], dtype=np.float32)
        dc_a = np.array([1.0, -dc_R], dtype=np.float32)
        dc_zi = sig.lfilter_zi(dc_b, dc_a).astype(np.complex64) * 0.0

        # ── Channel filter: IIR Butterworth (SOS) ──
        # IIR is ~3x faster than 101-tap FIR (0.9ms vs 3.0ms on 131k samples).
        # 5th-order Butterworth is plenty for FM/AM channel selection.
        ch_sos = sig.butter(
            5, ch_bw / (fs / 2), btype='low', output='sos'
        ).astype(np.float32)
        ch_sos_zi = sig.sosfilt_zi(ch_sos).astype(np.complex64) * 0.0

        # Decimation phase tracking across blocks.
        ch_decim_phase = 0

        # Create demodulator — the demod will compute its own _audio_decim
        # from ch_rate / actual_audio_rate, which is our audio_decim (exact).
        demod = create_demodulator(self._demod_mode, ch_rate, actual_audio_rate)

        # Display throttle: only update GUI every N blocks.
        # At 2.4 MHz / 131072 = ~18.3 blocks/sec, updating every 3rd
        # block gives ~6 Hz display rate — smooth enough for eyes, but
        # frees the GIL so the audio callback can run without starvation.
        display_counter = 0
        DISPLAY_EVERY = 3  # update spectrum/waterfall every 3rd block

        self._running = True
        try:
            while self._running:
                try:
                    # read_samples() is BLOCKING — it naturally paces the loop
                    # at ~55 ms per block (131072 / 2.4 MHz). No sleep needed.
                    samples = dev.read_samples()

                    # ━━━ AUDIO FIRST (time-critical) ━━━━━━━━━━━━━━━━━━━━━
                    # Process and deliver audio BEFORE FFT/display to minimize
                    # latency between data arrival and audio buffer write.

                    # Stay in complex64 throughout — RTL-SDR gives 8-bit I/Q,
                    # so complex64 (23-bit mantissa) is more than enough precision.
                    # This halves memory bandwidth vs complex128 → ~2x faster.
                    iq = samples  # already complex64 from rtl_device

                    #    DC removal (continuous IIR, float32)
                    iq_clean, dc_zi = sig.lfilter(dc_b, dc_a, iq, zi=dc_zi)

                    #    Frequency-shift to VFO offset (cos+sin is faster than exp)
                    vfo_off = self._vfo_offset_hz
                    if abs(vfo_off) > 0.5:
                        n = len(iq_clean)
                        dphi = np.float32(-2.0 * np.pi * vfo_off / fs)
                        phases = np.float32(self._shift_phase) + dphi * np.arange(n, dtype=np.float32)
                        osc = (np.cos(phases) + 1j * np.sin(phases)).astype(np.complex64)
                        iq_clean = iq_clean * osc
                        self._shift_phase = float((phases[-1] + dphi) % (2.0 * np.pi))

                    #    IIR channel filter (Butterworth SOS, ~3x faster than FIR)
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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Seismo-EM SDR Console")
        self.resize(1500, 900)

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

        self._build_ui()
        self._start_worker()

    # --- UI construction ---

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # ── Row 1: big frequency readout ──
        self.freq_label = QtWidgets.QLabel()
        self.freq_label.setFont(QtGui.QFont("DejaVu Sans Mono", 24))
        self.freq_label.setStyleSheet(
            "color: #00ccff; background: #000000; padding: 4px 8px;"
        )
        self.freq_label.setFixedHeight(56)
        root.addWidget(self.freq_label)

        # ── Row 2: signal meter ──
        self.signal_label = QtWidgets.QLabel("Signal: ---")
        self.signal_label.setStyleSheet("color: #a0b8d0; font-size: 13px; font-family: 'DejaVu Sans';")
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
            "color: #00ccff; font-size: 13px; font-family: 'DejaVu Sans'; padding: 2px;"
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
                font-family: 'DejaVu Sans Mono', monospace;
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
        log_label.setStyleSheet("color: #4499ff; font-size: 12px; font-family: 'DejaVu Sans';")
        ml_layout.addWidget(log_label)

        self.anomaly_log = QtWidgets.QTextEdit()
        self.anomaly_log.setReadOnly(True)
        self.anomaly_log.setMaximumHeight(120)
        self.anomaly_log.setStyleSheet("""
            QTextEdit {
                background-color: #060a10;
                color: #80b0d0;
                font-family: 'DejaVu Sans Mono', monospace;
                font-size: 10px;
                border: 1px solid #0c1a2e;
            }
        """)
        ml_layout.addWidget(self.anomaly_log)

        # Right side: tabs for Audio Debug vs ML Monitor
        right_tabs = QtWidgets.QTabWidget()
        right_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #0c1a2e; }
            QTabBar::tab {
                background: #0c1624;
                color: #a0b8d0;
                padding: 6px 16px;
                border: 1px solid #0c1a2e;
                font-family: 'DejaVu Sans';
                font-size: 11px;
            }
            QTabBar::tab:selected {
                background: #060a10;
                color: #00ccff;
                border-bottom: 2px solid #00ccff;
            }
        """)
        right_tabs.addTab(audio_panel, "Audio Debug")
        right_tabs.addTab(ml_panel, "ML Monitor")

        # Main horizontal splitter: RF display (left) | Tabs (right)
        main_hsplit = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        # RF column (spectrum + waterfall)
        rf_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        rf_splitter.addWidget(self.spec_plot)
        rf_splitter.addWidget(self.wf_widget)
        rf_splitter.setSizes([250, 550])

        main_hsplit.addWidget(rf_splitter)
        main_hsplit.addWidget(right_tabs)
        main_hsplit.setSizes([750, 350])
        root.addWidget(main_hsplit, 1)

        # ── Row 5: footer — manual tune + scan controls ──
        footer = QtWidgets.QHBoxLayout()
        root.addLayout(footer)

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
                font-family: 'DejaVu Sans'; font-size: 11px;
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
            "color: #506880; font-size: 11px; font-family: 'DejaVu Sans'; padding: 0 8px;"
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
                font-family: 'DejaVu Sans Mono';
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
            "color: #708898; font-size: 10px; font-family: 'DejaVu Sans Mono', monospace;"
        )
        footer.addWidget(self.scan_dwell_label)

        footer.addStretch(1)

        # ── Connections ──
        self.freq_spin.valueChanged.connect(self._on_freq_changed)
        self.mode_cb.currentTextChanged.connect(self._on_mode_changed)
        self.antenna_cb.currentTextChanged.connect(self._on_antenna_changed)
        self.gain_slider.valueChanged.connect(self._on_gain_changed)
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

    # --- Worker management ---

    def _start_worker(self):
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait()
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
        )
        self._worker.spectrum_ready.connect(self._update_spectrum)
        self._worker.signal_level.connect(self._update_signal)
        self._worker.audio_samples.connect(self._update_audio_display)
        self._worker.audio_debug.connect(self._update_audio_debug)
        self._worker.error.connect(self._show_error)
        self._worker.start()

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
        self._start_worker()  # restart with new demod + channel filter

    def _on_antenna_changed(self, key: str):
        self._antenna_key = key
        self._start_worker()

    def _on_gain_changed(self, val: int):
        self._gain_db = float(val)
        self.gain_lbl.setText(f"Gain: {val}dB")
        if self._worker is not None:
            self._worker.set_gain(self._gain_db)

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
            self._start_worker()
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
        _scan_font = "font-size: 11px; font-family: 'DejaVu Sans'; padding: 0 8px;"
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
            f"color: {color}; font-size: 13px; font-family: 'DejaVu Sans';"
        )

    @QtCore.pyqtSlot(object, object)
    def _update_spectrum(self, freqs_hz: np.ndarray, power_db: np.ndarray):
        if len(freqs_hz) < 2:
            return

        # Feed spectrum to ML monitor for anomaly detection
        self._monitor.on_spectrum(
            freqs_hz, power_db,
            center_hz=self._center_freq_hz,
            sample_rate_hz=self._sample_rate_hz,
        )

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
        max_rows = 512
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

            # Normalize for colormap (auto-level)
            sg_min = float(np.percentile(self.audio_sg_data, 5))
            sg_max = float(np.max(self.audio_sg_data))
            if sg_max <= sg_min:
                sg_max = sg_min + 1.0
            norm = (self.audio_sg_data - sg_min) / (sg_max - sg_min)
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
                item.setFont(QtGui.QFont("DejaVu Sans Mono", 10))
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

    @QtCore.pyqtSlot(object)
    def _on_anomaly_detected(self, result):
        """Log anomaly and auto-tune VFO to the anomalous signal.

        When scanning, this moves the VFO to the peak frequency within
        the anomalous band so the user hears the signal and sees the
        tune line pointing at it on the spectrum/waterfall.
        """
        from datetime import datetime

        peak_mhz = result.peak_freq_hz / 1e6 if result.peak_freq_hz > 0 else 0

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
        self.signal_label.setStyleSheet("color: #4499ff; font-size: 13px; font-family: 'DejaVu Sans';")

    def closeEvent(self, ev):
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait()
        self._monitor.close()
        super().closeEvent(ev)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    pg.setConfigOption("background", "#000000")    # black like SDR#
    pg.setConfigOption("foreground", "#a0b8d0")

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

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
