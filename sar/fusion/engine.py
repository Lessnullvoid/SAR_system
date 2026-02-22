"""
Sensor fusion engine — combines multi-source data into per-tile scores.

The fusion engine sits between the ingest layer (sensor_scheduler) and
the presentation layer (GUI, OSC, storage).  It:

1. Maintains a rolling history of TileState snapshots.
2. Computes composite scores using weighted sensor contributions.
3. Detects multi-sensor correlation (seismic + EM is more significant
   than either alone).
4. Emits alerts when composite scores cross thresholds.

Scientific basis
────────────────
Pre-seismic EM signals are more credible when they correlate with:
- Microseismic swarm activity
- Crustal deformation (GNSS)
- Ionospheric TEC perturbations
- Geomagnetic field anomalies

The fusion engine quantifies this multi-parameter correlation.

Usage
-----
    from sar.fusion.engine import FusionEngine
    engine = FusionEngine(tile_states)
    engine.update()
    alerts = engine.get_alerts()
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..geo.tile_state import TileState

log = logging.getLogger(__name__)


@dataclass
class FusionAlert:
    """A multi-sensor correlated alert."""
    tile_id: str
    composite_score: float
    contributing_sensors: List[str]
    description: str
    timestamp: float = field(default_factory=time.time)
    level: str = "info"  # "info", "watch", "warning", "critical"


class FusionEngine:
    """Multi-sensor fusion for per-tile anomaly assessment.

    Maintains a rolling window of tile state snapshots and computes
    trend-aware composite scores.
    """

    # Alert thresholds
    WATCH_THRESHOLD = 0.15
    WARNING_THRESHOLD = 0.30
    CRITICAL_THRESHOLD = 0.60

    def __init__(
        self,
        tile_states: Dict[str, TileState],
        history_window: int = 60,  # number of snapshots to keep
    ):
        self._states = tile_states
        self._history: Dict[str, deque] = {
            tid: deque(maxlen=history_window)
            for tid in tile_states
        }
        self._alerts: List[FusionAlert] = []
        self._alert_history: deque = deque(maxlen=500)

    def update(self) -> List[FusionAlert]:
        """Process current tile states and generate any new alerts.

        Returns new alerts generated in this update cycle.
        """
        new_alerts: List[FusionAlert] = []

        for tid, state in self._states.items():
            # Record snapshot
            self._history[tid].append({
                "time": state.timestamp,
                "composite": state.composite_score,
                "seismic": state.seismic.events_24h,
                "max_mag": state.seismic.max_mag,
                "kp": state.magnetic.kp_index,
                "tec_delta": state.ionospheric.tec_delta,
                "rf": state.rf.anomaly_score,
            })

            score = state.composite_score
            if score < self.WATCH_THRESHOLD:
                continue

            # Determine which sensors are contributing
            contributors = []
            if state.seismic.events_24h > 5 or state.seismic.max_mag > 3.0:
                contributors.append("seismic")
            if state.rf.anomaly_score > 0.3:
                contributors.append("rf_anomaly")
            if state.magnetic.kp_index > 3.0:
                contributors.append("geomagnetic")
            if abs(state.ionospheric.tec_delta) > 2.0:
                contributors.append("ionospheric")
            if state.gnss.max_mm_day > 2.0:
                contributors.append("gnss")

            if not contributors:
                continue

            # Multi-sensor correlation bonus
            n_sources = len(contributors)
            correlation_factor = 1.0 + (n_sources - 1) * 0.15
            adjusted_score = min(score * correlation_factor, 1.0)

            # Determine alert level
            if adjusted_score >= self.CRITICAL_THRESHOLD:
                level = "critical"
            elif adjusted_score >= self.WARNING_THRESHOLD:
                level = "warning"
            else:
                level = "watch"

            # Trend analysis — is the score rising?
            hist = self._history[tid]
            trend = ""
            if len(hist) >= 3:
                recent = [h["composite"] for h in list(hist)[-3:]]
                if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
                    trend = " (rising)"

            desc_parts = []
            if "seismic" in contributors:
                desc_parts.append(
                    f"seismic: {state.seismic.events_24h} events, "
                    f"M{state.seismic.max_mag:.1f} max"
                )
            if "rf_anomaly" in contributors:
                desc_parts.append(
                    f"RF anomaly: {state.rf.anomaly_score:.2f}"
                )
            if "geomagnetic" in contributors:
                desc_parts.append(
                    f"Kp={state.magnetic.kp_index:.1f}"
                )
            if "ionospheric" in contributors:
                desc_parts.append(
                    f"TEC delta: {state.ionospheric.tec_delta:.1f}σ"
                )
            if "gnss" in contributors:
                desc_parts.append(
                    f"GNSS: {state.gnss.max_mm_day:.1f} mm/day"
                )

            alert = FusionAlert(
                tile_id=tid,
                composite_score=adjusted_score,
                contributing_sensors=contributors,
                description="; ".join(desc_parts) + trend,
                level=level,
            )
            new_alerts.append(alert)
            self._alert_history.append(alert)

        self._alerts = new_alerts

        # Log significant alerts
        for a in new_alerts:
            if a.level in ("warning", "critical"):
                log.warning(
                    "FUSION %s — tile %s: %.2f [%s] %s",
                    a.level.upper(), a.tile_id, a.composite_score,
                    "+".join(a.contributing_sensors), a.description,
                )

        return new_alerts

    def get_alerts(self, level: Optional[str] = None) -> List[FusionAlert]:
        """Get current alerts, optionally filtered by level."""
        if level:
            return [a for a in self._alerts if a.level == level]
        return list(self._alerts)

    def get_history(self, tile_id: str) -> List[dict]:
        """Get the score history for one tile."""
        return list(self._history.get(tile_id, []))

    def get_top_tiles(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get the N tiles with highest composite scores."""
        scored = [
            (tid, state.composite_score)
            for tid, state in self._states.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]
