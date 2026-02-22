"""
Automated radio scan orchestrator for seismo-EM monitoring.

The orchestrator automatically tunes the SDR through a list of frequency
positions, dwelling at each one long enough for the ML detector to collect
meaningful data.  When an anomaly is detected, the dwell time is extended
so the system gathers more evidence.

Scan behaviour
──────────────
  1. Group positions by antenna to minimise hardware reconfig (slow).
  2. At each position: tune → settle → dwell → analyse → next.
  3. If the ML detector fires during dwell, double the remaining time
     (up to ``max_dwell_s``) to capture more data.
  4. After completing one full cycle, start over.
  5. The user can pause / resume / stop at any time.

Data flow
─────────
  ScanOrchestrator ─(tune_requested)──▶ MainWindow._on_scan_tune()
       │                                   ├── set antenna/mode/freq
       │                                   └── restart worker if needed
       │
       ├─ QTimer (1 s tick)
       └─ anomaly_extend() called by ML monitor when score > threshold
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from PyQt5 import QtCore

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scan position
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScanPosition:
    """One stop in the scan plan."""

    label: str          # human-readable label, e.g. "HF 40m"
    center_hz: float    # SDR center frequency (Hz)
    antenna: str        # antenna key from antennas.json
    mode: str           # demodulation mode: am, fm, lsb, usb
    dwell_s: float      # base dwell time (seconds)
    band_name: str      # which Band from bands.py this covers
    priority: int = 1   # 1 = primary, 2 = secondary


# ---------------------------------------------------------------------------
# Default scan plan — covers all bands from the monitoring plan
# ---------------------------------------------------------------------------

DEFAULT_SCAN_PLAN: List[ScanPosition] = [
    # ── Loop antenna (direct-sampling Q) ─────────────────────────────
    # Group all loop-antenna positions together to avoid antenna switch.

    # VLF + LF: beacon frequencies (NWC 19.8 kHz, NAA 24 kHz, DCF77 77.5 kHz)
    ScanPosition(
        label="VLF/LF Beacons",
        center_hz=500_000,
        antenna="loop_antenna",
        mode="am",
        dwell_s=45,
        band_name="VLF",
        priority=1,
    ),

    # MF: AM broadcast (530–1700 kHz)
    ScanPosition(
        label="MF AM Lower",
        center_hz=1_000_000,
        antenna="loop_antenna",
        mode="am",
        dwell_s=30,
        band_name="MF",
        priority=1,
    ),
    ScanPosition(
        label="MF AM Upper",
        center_hz=1_500_000,
        antenna="loop_antenna",
        mode="am",
        dwell_s=30,
        band_name="MF",
        priority=2,
    ),

    # HF: Key amateur / broadcast bands for ionospheric monitoring
    ScanPosition(
        label="HF 80m",
        center_hz=3_500_000,
        antenna="loop_antenna",
        mode="lsb",
        dwell_s=30,
        band_name="HF",
        priority=1,
    ),
    ScanPosition(
        label="HF 40m",
        center_hz=7_100_000,
        antenna="loop_antenna",
        mode="lsb",
        dwell_s=30,
        band_name="HF",
        priority=1,
    ),
    ScanPosition(
        label="HF 30m / WWV",
        center_hz=10_000_000,
        antenna="loop_antenna",
        mode="am",
        dwell_s=45,
        band_name="HF",
        priority=1,
    ),
    ScanPosition(
        label="HF 20m",
        center_hz=14_100_000,
        antenna="loop_antenna",
        mode="usb",
        dwell_s=30,
        band_name="HF",
        priority=1,
    ),
    ScanPosition(
        label="HF 15m",
        center_hz=21_000_000,
        antenna="loop_antenna",
        mode="usb",
        dwell_s=30,
        band_name="HF",
        priority=2,
    ),
    ScanPosition(
        label="HF 10m",
        center_hz=28_000_000,
        antenna="loop_antenna",
        mode="usb",
        dwell_s=30,
        band_name="HF",
        priority=2,
    ),

    # ── FM broadcast antenna (quadrature) ────────────────────────────

    ScanPosition(
        label="FM Low",
        center_hz=92_000_000,
        antenna="fm_broadcast",
        mode="fm",
        dwell_s=30,
        band_name="FM",
        priority=1,
    ),
    ScanPosition(
        label="FM Mid",
        center_hz=98_000_000,
        antenna="fm_broadcast",
        mode="fm",
        dwell_s=30,
        band_name="FM",
        priority=1,
    ),
    ScanPosition(
        label="FM High",
        center_hz=104_000_000,
        antenna="fm_broadcast",
        mode="fm",
        dwell_s=30,
        band_name="FM",
        priority=1,
    ),

    # ── Discone antenna (quadrature) ─────────────────────────────────

    ScanPosition(
        label="2m Ham",
        center_hz=146_000_000,
        antenna="discone",
        mode="fm",
        dwell_s=30,
        band_name="2m",
        priority=1,
    ),
    ScanPosition(
        label="70cm",
        center_hz=440_000_000,
        antenna="discone",
        mode="fm",
        dwell_s=30,
        band_name="70cm",
        priority=2,
    ),
]


# ---------------------------------------------------------------------------
# Scan orchestrator
# ---------------------------------------------------------------------------

class ScanOrchestrator(QtCore.QObject):
    """Automated frequency-scanning controller.

    Signals
    -------
    tune_requested(ScanPosition)
        Emitted when the SDR must be tuned to a new position.
        The GUI connects this to its tuning logic.
    progress(int, int)
        (current_index, total_positions) — for progress bar.
    dwell_tick(float, float)
        (elapsed_s, total_s) — seconds into / total dwell at current pos.
    cycle_complete()
        Emitted after the last position in one full scan cycle.
    state_changed(str)
        "scanning", "paused", "stopped", "dwelling", "extended"
    """

    tune_requested = QtCore.pyqtSignal(object)       # ScanPosition
    progress = QtCore.pyqtSignal(int, int)            # idx, total
    dwell_tick = QtCore.pyqtSignal(float, float)      # elapsed, total
    cycle_complete = QtCore.pyqtSignal()
    state_changed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        plan: Optional[List[ScanPosition]] = None,
        max_dwell_s: float = 180.0,
        settle_s: float = 3.0,
        parent: Optional[QtCore.QObject] = None,
    ):
        """
        Parameters
        ----------
        plan : list[ScanPosition], optional
            Scan positions to cycle through.  Defaults to DEFAULT_SCAN_PLAN.
        max_dwell_s : float
            Maximum dwell time even with anomaly extensions (seconds).
        settle_s : float
            Seconds to wait after tuning before counting dwell time
            (lets filters / PLL settle).
        """
        super().__init__(parent)
        self._plan = list(plan or DEFAULT_SCAN_PLAN)
        self._max_dwell_s = max_dwell_s
        self._settle_s = settle_s

        self._current_idx = 0
        self._dwell_elapsed = 0.0
        self._dwell_target = 0.0
        self._settling = False
        self._settle_elapsed = 0.0
        self._running = False
        self._paused = False
        self._cycle_count = 0
        self._dwell_extended = False    # only allow ONE extension per position

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(1000)   # 1-second tick
        self._timer.timeout.connect(self._on_tick)

    # ── properties ────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def current_position(self) -> Optional[ScanPosition]:
        if not self._plan:
            return None
        return self._plan[self._current_idx % len(self._plan)]

    @property
    def plan(self) -> List[ScanPosition]:
        return list(self._plan)

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    # ── control ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start or resume scanning."""
        if not self._plan:
            log.warning("Cannot start scan — empty plan")
            return

        if self._paused:
            self._paused = False
            self._timer.start()
            self.state_changed.emit("scanning")
            log.info("Scan resumed at position %d/%d",
                     self._current_idx + 1, len(self._plan))
            return

        self._running = True
        self._paused = False
        self._current_idx = 0
        self._cycle_count = 0
        self._go_to_position(0)
        self._timer.start()
        self.state_changed.emit("scanning")
        log.info("Scan started: %d positions, cycle ≈ %.0f s",
                 len(self._plan),
                 sum(p.dwell_s + self._settle_s for p in self._plan))

    def pause(self) -> None:
        """Pause scanning (keeps position, stops timer)."""
        if self._running and not self._paused:
            self._paused = True
            self._timer.stop()
            self.state_changed.emit("paused")
            log.info("Scan paused at position %d/%d",
                     self._current_idx + 1, len(self._plan))

    def stop(self) -> None:
        """Stop scanning entirely."""
        self._running = False
        self._paused = False
        self._timer.stop()
        self.state_changed.emit("stopped")
        log.info("Scan stopped after %d cycles", self._cycle_count)

    def anomaly_extend(self, band_name: str) -> None:
        """Called by the ML monitor when an anomaly is detected.

        Adds a FIXED +30 s extension, but ONLY ONCE per scan position.
        Previous behaviour doubled remaining time on every single anomaly
        alert, causing the scanner to get stuck at one position forever
        (feedback loop: more dwell → more anomalies → more dwell).
        """
        if not self._running or self._paused or self._settling:
            return
        if self._dwell_extended:
            return   # already extended at this position — ignore

        pos = self.current_position
        if pos is None:
            return

        extension = min(30.0, self._max_dwell_s - self._dwell_target)
        if extension > 2.0:
            self._dwell_target = self._dwell_target + extension
            self._dwell_extended = True     # block further extensions
            self.state_changed.emit("extended")
            log.info(
                "Dwell extended at %s: +%.0f s (now %.0f/%.0f s) "
                "due to anomaly in %s",
                pos.label, extension, self._dwell_elapsed,
                self._dwell_target, band_name,
            )

    # ── internal ──────────────────────────────────────────────────────

    def _go_to_position(self, idx: int) -> None:
        """Tune to position *idx* and start the settle countdown."""
        self._current_idx = idx % len(self._plan)
        pos = self._plan[self._current_idx]

        self._settling = True
        self._settle_elapsed = 0.0
        self._dwell_elapsed = 0.0
        self._dwell_target = pos.dwell_s
        self._dwell_extended = False    # allow one extension at the new position

        self.tune_requested.emit(pos)
        self.progress.emit(self._current_idx, len(self._plan))
        self.dwell_tick.emit(0.0, pos.dwell_s)

        log.info(
            "Tuning to [%d/%d] %s  %.3f MHz  %s  %s  dwell=%ds",
            self._current_idx + 1, len(self._plan),
            pos.label, pos.center_hz / 1e6,
            pos.antenna, pos.mode, int(pos.dwell_s),
        )

    def _on_tick(self) -> None:
        """Called every second by the QTimer."""
        if not self._running or self._paused:
            return

        # Settle phase: wait for filters / PLL to stabilise
        if self._settling:
            self._settle_elapsed += 1.0
            if self._settle_elapsed >= self._settle_s:
                self._settling = False
                self.state_changed.emit("dwelling")
            return

        # Dwell phase: count down
        self._dwell_elapsed += 1.0
        self.dwell_tick.emit(self._dwell_elapsed, self._dwell_target)

        if self._dwell_elapsed >= self._dwell_target:
            self._advance()

    def _advance(self) -> None:
        """Move to the next position in the scan plan."""
        next_idx = self._current_idx + 1
        if next_idx >= len(self._plan):
            next_idx = 0
            self._cycle_count += 1
            self.cycle_complete.emit()
            log.info("Scan cycle %d complete", self._cycle_count)

        self._go_to_position(next_idx)
