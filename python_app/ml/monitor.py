"""
Seismo-EM Monitor — ties together feature extraction, anomaly detection,
and data storage.

Runs on the Qt event loop (receives signals from the SpectrumWorker)
and emits its own signals when anomalies are detected.

Data flow
─────────
  SpectrumWorker ─spectrum_ready─▶ Monitor.on_spectrum()
       │                               ├── extract features
       │                               ├── detector.update()
       │                               ├── store to SQLite
       │                               └── emit anomaly_detected if score > threshold
       │
       └── runs in QThread              Monitor lives on GUI thread
                                        (feature extraction is fast: <1 ms)
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np
from PyQt5 import QtCore

from .bands import Band, BANDS, bands_in_range
from .detector import AnomalyDetector, AnomalyResult
from .features import BandFeatures, extract_features, N_FEATURES, FEATURE_NAMES
from .store import DataStore

log = logging.getLogger(__name__)


class SeismoMonitor(QtCore.QObject):
    """Central ML monitor.  Lives on the GUI thread.

    Signals
    -------
    anomaly_detected(AnomalyResult)
        Emitted whenever a band's composite anomaly score exceeds the
        alert threshold.
    band_update(str, dict)
        Emitted for every band update with a summary dict:
        ``{"band", "timestamp", "composite", "z_composite", "if_score",
           "features": {name: value, ...}, "triggered": [...]}``
    status_message(str)
        Informational messages for the status bar.
    """

    anomaly_detected = QtCore.pyqtSignal(object)   # AnomalyResult
    band_update = QtCore.pyqtSignal(str, object)    # band_name, summary_dict
    status_message = QtCore.pyqtSignal(str)

    # How often to commit to SQLite (seconds)
    _FLUSH_INTERVAL = 10.0
    # How often to prune old data (seconds)
    _PRUNE_INTERVAL = 3600.0
    # Anomaly alert threshold — raised to 0.75 so BOTH the Z-score and
    # Isolation Forest must agree before triggering an alert.  With the
    # composite now being an average (not max), score=0.75 means both
    # detectors are strongly elevated.
    _ALERT_THRESHOLD = 0.75

    def __init__(
        self,
        alert_threshold: float = 0.75,
        z_threshold: float = 4.0,
        parent: Optional[QtCore.QObject] = None,
    ):
        super().__init__(parent)
        self._alert_threshold = alert_threshold
        self._detector = AnomalyDetector(
            z_threshold=z_threshold,
            z_alpha=0.002,      # slower EMA → ~500-sample window (~80 s)
            z_warmup=500,       # ~80 s learning before any Z-score alerts
            if_contamination=0.05,  # expect 5% anomalies (RF is noisy)
        )
        self._store = DataStore()
        self._last_flush = time.time()
        self._last_prune = time.time()
        self._update_count = 0

        # Latest results per band (for GUI display)
        self._latest: Dict[str, AnomalyResult] = {}

    # ── public API ────────────────────────────────────────────────────

    @QtCore.pyqtSlot(object, object)
    def on_spectrum(
        self,
        freqs_hz: np.ndarray,
        power_db: np.ndarray,
        center_hz: float = 0.0,
        sample_rate_hz: float = 2_400_000.0,
    ) -> None:
        """Called when a new spectrum arrives from the SpectrumWorker.

        This does feature extraction + anomaly detection + storage.
        Typically takes < 1 ms so it's fine on the GUI thread.
        """
        ts = time.time()

        # Extract features for all visible bands
        band_features = extract_features(
            freqs_hz, power_db, center_hz, sample_rate_hz, timestamp=ts
        )

        for bf in band_features:
            # Anomaly detection
            result = self._detector.update(bf)
            # Propagate peak frequency so the GUI can auto-tune to it
            result.peak_freq_hz = bf.peak_freq_hz
            self._latest[bf.band_name] = result

            # Store to SQLite
            self._store.store_features(bf)
            self._store.store_anomaly(result)

            # Emit update for GUI
            summary = {
                "band": bf.band_name,
                "timestamp": ts,
                "composite": result.composite,
                "z_composite": result.z_composite,
                "if_score": result.if_score,
                "features": bf.as_dict(),
                "triggered": result.triggered_features,
                "is_anomaly": result.is_anomaly,
                "peak_freq_hz": bf.peak_freq_hz,
            }
            self.band_update.emit(bf.band_name, summary)

            # Alert if anomalous — only emit signal and log WARNING for
            # strong detections (composite ≥ 0.75).  This requires BOTH the
            # rolling Z-score and Isolation Forest to be elevated, which
            # virtually eliminates false positives.
            if result.composite >= self._alert_threshold:
                self.anomaly_detected.emit(result)
                log.warning(
                    "ANOMALY [%s] score=%.2f z=%.2f if=%.2f peak=%.3f MHz triggered=%s",
                    bf.band_name, result.composite, result.z_composite,
                    result.if_score, bf.peak_freq_hz / 1e6,
                    result.triggered_features,
                )
            elif result.composite >= 0.5:
                # Moderate detection — log at DEBUG to avoid flooding the console
                log.debug(
                    "anomaly-low [%s] score=%.2f z=%.2f if=%.2f",
                    bf.band_name, result.composite,
                    result.z_composite, result.if_score,
                )

        self._update_count += 1

        # Periodic flush
        if ts - self._last_flush > self._FLUSH_INTERVAL:
            self._store.flush()
            self._last_flush = ts

        # Periodic prune
        if ts - self._last_prune > self._PRUNE_INTERVAL:
            self._store.prune_old()
            self._last_prune = ts

        # Periodic status
        if self._update_count % 30 == 0:
            n_bands = len(self._latest)
            n_anom = sum(1 for r in self._latest.values() if r.is_anomaly)
            self.status_message.emit(
                f"ML: {n_bands} bands monitored, "
                f"{n_anom} anomalies active, "
                f"{self._update_count} updates"
            )

    def get_latest(self, band_name: str) -> Optional[AnomalyResult]:
        """Get the most recent anomaly result for a band."""
        return self._latest.get(band_name)

    def get_all_latest(self) -> Dict[str, AnomalyResult]:
        """Get latest results for all tracked bands."""
        return dict(self._latest)

    def get_history(self, band: str, seconds: float = 3600.0) -> List[Dict]:
        """Get recent feature history from the database."""
        return self._store.get_recent_features(band, seconds)

    def get_recent_anomalies(self, seconds: float = 86400.0) -> List[Dict]:
        """Get recent anomaly records from the database."""
        return self._store.get_recent_anomalies(seconds)

    def close(self) -> None:
        """Flush and close the data store."""
        self._store.flush()
        self._store.close()
