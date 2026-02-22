from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class AnomalyThresholds:
    power_dbm_anomaly_threshold: float


class AnomalyEngine:
    """
    Very simple anomaly detector based on deviation from rolling median.
    """

    def __init__(self, threshold_db: float = 10.0, history_size: int = 120) -> None:
        self.threshold_db = threshold_db
        self.history_size = history_size
        self._history: Dict[str, List[np.ndarray]] = {}

    def update_and_detect(
        self, band_name: str, freqs_hz: np.ndarray, power_db: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Returns a list of (freq_hz, delta_db) where anomalies are detected.
        """
        if band_name not in self._history:
            self._history[band_name] = []

        history = self._history[band_name]
        history.append(power_db.astype(np.float32))
        if len(history) > self.history_size:
            history.pop(0)

        if len(history) < 10:
            return []

        baseline = np.median(np.stack(history, axis=0), axis=0)
        delta = power_db - baseline
        mask = delta > self.threshold_db
        if not np.any(mask):
            return []

        return [(float(freqs_hz[i]), float(delta[i])) for i in np.where(mask)[0]]


