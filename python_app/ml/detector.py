"""
Anomaly detection for seismo-EM monitoring.

Two complementary detectors run in parallel:

1. **Rolling Z-score** (real-time, per-feature)
   Maintains an exponential moving average (EMA) and variance for each
   feature in each band.  When a feature deviates more than ``z_threshold``
   standard deviations from the baseline, it is flagged.
   Fast, lightweight, pure-numpy.  Good for sudden shifts.

2. **Isolation Forest** (batch, multivariate)
   Periodically re-trained on the most recent feature history.
   Catches complex multivariate anomalies that individual Z-scores miss
   (e.g. noise floor drops while occupancy rises — a pattern potentially
   linked to ionospheric changes before seismic events).

Both produce a per-band anomaly score ∈ [0, 1].  The composite score
is ``max(z_score_norm, if_score)`` so that either detector can trigger
an alert.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .features import BandFeatures, N_FEATURES, FEATURE_NAMES

log = logging.getLogger(__name__)

# Try to import sklearn; if unavailable, fall back to Z-score only.
try:
    from sklearn.ensemble import IsolationForest

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    log.warning("scikit-learn not installed — Isolation Forest disabled")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AnomalyResult:
    """Result of anomaly detection for one band at one time step."""

    band_name: str
    timestamp: float
    z_scores: np.ndarray        # per-feature Z-scores, shape (N_FEATURES,)
    z_composite: float          # max |Z| across features, normalised to [0,1]
    if_score: float             # Isolation Forest anomaly score [0,1]  (0=normal)
    composite: float            # combined anomaly score [0,1]
    triggered_features: List[str]  # feature names that exceeded z_threshold
    peak_freq_hz: float = 0.0  # frequency of strongest signal in the band

    @property
    def is_anomaly(self) -> bool:
        return self.composite > 0.5


# ---------------------------------------------------------------------------
# Rolling Z-score detector
# ---------------------------------------------------------------------------

class _RollingBaseline:
    """Per-band EMA baseline for Z-score detection."""

    def __init__(self, alpha: float = 0.002, warmup: int = 500):
        """
        Parameters
        ----------
        alpha : float
            EMA smoothing factor.  0.002 → ~500-sample time constant.
            At ~6 updates/sec this is a ~80-second effective window.
        warmup : int
            Number of samples before the baseline is considered valid.
            500 samples at ~6 Hz = ~80 seconds of learning before alerting.
        """
        self.alpha = alpha
        self.warmup = warmup
        self._mean: Optional[np.ndarray] = None  # EMA of features
        self._var: Optional[np.ndarray] = None    # EMA of (feature − mean)²
        self._count = 0

    @property
    def ready(self) -> bool:
        return self._count >= self.warmup

    def update(self, values: np.ndarray) -> np.ndarray:
        """Update baseline and return Z-scores for the new observation.

        Returns
        -------
        z_scores : ndarray, shape (N_FEATURES,)
            Z-scores (signed).  NaN during warmup.
        """
        v = values.astype(np.float64)
        if self._mean is None:
            self._mean = v.copy()
            self._var = np.zeros_like(v)
            self._count = 1
            return np.full(len(v), np.nan)

        self._count += 1
        a = self.alpha
        delta = v - self._mean
        self._mean += a * delta
        self._var = (1 - a) * (self._var + a * delta ** 2)

        if not self.ready:
            return np.full(len(v), np.nan)

        std = np.sqrt(np.maximum(self._var, 1e-12))
        return (v - self._mean) / std


# ---------------------------------------------------------------------------
# Isolation Forest wrapper
# ---------------------------------------------------------------------------

class _IFModel:
    """Threadsafe Isolation Forest wrapper with periodic retraining."""

    MIN_SAMPLES = 200       # minimum observations before first training
    RETRAIN_EVERY = 300     # retrain every N new observations
    MAX_HISTORY = 3000      # keep most recent N observations for training

    def __init__(self, contamination: float = 0.02):
        self._history: List[np.ndarray] = []
        self._model: Optional[object] = None
        self._since_train = 0
        self._contamination = contamination
        self._lock = threading.Lock()

    def add(self, values: np.ndarray) -> float:
        """Add an observation and return anomaly score ∈ [0, 1].

        Returns 0.0 until the model has been trained.
        """
        self._history.append(values.copy())
        if len(self._history) > self.MAX_HISTORY:
            self._history = self._history[-self.MAX_HISTORY:]
        self._since_train += 1

        # Check if retraining is needed
        if self._since_train >= self.RETRAIN_EVERY and len(self._history) >= self.MIN_SAMPLES:
            self._retrain()

        return self._score(values)

    def _retrain(self) -> None:
        if not _HAS_SKLEARN:
            return
        X = np.array(self._history, dtype=np.float64)
        # Replace NaN/inf with 0 for robustness
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            model = IsolationForest(
                n_estimators=100,
                contamination=self._contamination,
                random_state=42,
                n_jobs=1,  # don't parallelise — we're already in a thread
            )
            model.fit(X)
            with self._lock:
                self._model = model
            self._since_train = 0
            log.info("Isolation Forest retrained on %d samples", len(X))
        except Exception as exc:
            log.warning("IF retrain failed: %s", exc)

    def _score(self, values: np.ndarray) -> float:
        with self._lock:
            if self._model is None:
                return 0.0
        v = np.nan_to_num(values.astype(np.float64).reshape(1, -1))
        try:
            with self._lock:
                # decision_function: large negative = anomaly
                raw = float(self._model.decision_function(v)[0])
            # Map to [0, 1]: raw ∈ roughly [-0.5, 0.5]
            # Negative = anomaly, positive = normal
            score = np.clip(0.5 - raw, 0.0, 1.0)
            return float(score)
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """Combined Z-score + Isolation Forest anomaly detector.

    Usage::

        detector = AnomalyDetector()

        # Called every time new features arrive (e.g. ~6 Hz)
        for band_features in extract_features(...):
            result = detector.update(band_features)
            if result.is_anomaly:
                alert(result)
    """

    def __init__(
        self,
        z_threshold: float = 4.0,
        z_alpha: float = 0.002,
        z_warmup: int = 500,
        if_contamination: float = 0.05,
    ):
        """
        Parameters
        ----------
        z_threshold : float
            Z-score threshold for flagging a feature as anomalous.
            4.0 = ~0.006% false-positive rate per feature (very conservative).
        z_alpha : float
            EMA smoothing factor for the rolling baseline.
            0.002 → ~500-sample effective window (~80 s at 6 Hz).
        z_warmup : int
            Observations before Z-score baseline is valid.
            500 at ~6 Hz = ~80 s of learning before alerting.
        if_contamination : float
            Expected fraction of anomalies for Isolation Forest.
            0.05 (5%) is more realistic for noisy RF environments.
        """
        self._z_threshold = z_threshold
        self._z_alpha = z_alpha
        self._z_warmup = z_warmup
        self._if_contamination = if_contamination

        # Per-band state
        self._baselines: Dict[str, _RollingBaseline] = {}
        self._if_models: Dict[str, _IFModel] = {}

    def update(self, bf: BandFeatures) -> AnomalyResult:
        """Process a new feature vector and return the anomaly assessment."""
        name = bf.band_name

        # Lazy-init per-band state
        if name not in self._baselines:
            self._baselines[name] = _RollingBaseline(
                alpha=self._z_alpha, warmup=self._z_warmup
            )
            self._if_models[name] = _IFModel(
                contamination=self._if_contamination
            )

        # Z-score
        z_scores = self._baselines[name].update(bf.values)
        if np.any(np.isnan(z_scores)):
            z_composite = 0.0
            triggered: List[str] = []
        else:
            abs_z = np.abs(z_scores)
            # Normalise max |Z| to [0, 1]:  threshold → 0.5,  2×threshold → 1.0
            z_composite = float(np.clip(
                np.max(abs_z) / (2.0 * self._z_threshold), 0.0, 1.0
            ))
            triggered = [
                FEATURE_NAMES[i]
                for i in range(len(z_scores))
                if abs_z[i] > self._z_threshold
            ]

        # Isolation Forest
        if_score = self._if_models[name].add(bf.values)

        # Composite: require BOTH detectors to agree (average, not max).
        # Using max() caused too many false positives because IF scores
        # naturally hover around 0.4-0.6 for noisy RF data.
        composite = (z_composite + if_score) / 2.0

        return AnomalyResult(
            band_name=name,
            timestamp=bf.timestamp,
            z_scores=z_scores,
            z_composite=z_composite,
            if_score=if_score,
            composite=composite,
            triggered_features=triggered,
        )

    def reset(self, band_name: Optional[str] = None) -> None:
        """Reset detector state for one or all bands."""
        if band_name:
            self._baselines.pop(band_name, None)
            self._if_models.pop(band_name, None)
        else:
            self._baselines.clear()
            self._if_models.clear()

    @property
    def band_names(self) -> List[str]:
        """Bands currently being tracked."""
        return list(self._baselines.keys())
