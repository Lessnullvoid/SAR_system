"""
SQLite time-series storage for seismo-EM feature data.

Stores feature vectors and anomaly scores in a local database so that
post-hoc analysis can correlate RF anomalies with seismic catalogues.

Schema
──────
  features:   (id, band, timestamp, f0..f7, center_hz, sample_rate_hz)
  anomalies:  (id, band, timestamp, z_composite, if_score, composite,
               triggered_features)

The database file lives at ``data/seismo_em.db`` relative to the
project root.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .features import BandFeatures, N_FEATURES, FEATURE_NAMES
from .detector import AnomalyResult

log = logging.getLogger(__name__)

# Default DB path (relative to project root)
_DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent.parent / "data"


class DataStore:
    """SQLite-backed time-series storage for features and anomalies."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            _DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
            db_path = _DEFAULT_DB_DIR / "seismo_em.db"
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._open()

    # ── lifecycle ──────────────────────────────────────────────────────

    def _open(self) -> None:
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,  # accessed from monitor thread
        )
        self._conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads
        self._conn.execute("PRAGMA synchronous=NORMAL")  # faster writes
        self._create_tables()
        log.info("DataStore opened: %s", self._db_path)

    def _create_tables(self) -> None:
        c = self._conn
        # Feature vectors
        cols = ", ".join(f"f{i} REAL" for i in range(N_FEATURES))
        c.execute(f"""
            CREATE TABLE IF NOT EXISTS features (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                band        TEXT    NOT NULL,
                timestamp   REAL    NOT NULL,
                {cols},
                center_hz   REAL,
                sample_rate_hz REAL
            )
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_band_ts
            ON features (band, timestamp)
        """)

        # Anomaly results
        c.execute("""
            CREATE TABLE IF NOT EXISTS anomalies (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                band        TEXT    NOT NULL,
                timestamp   REAL    NOT NULL,
                z_composite REAL,
                if_score    REAL,
                composite   REAL,
                triggered   TEXT
            )
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_anomalies_band_ts
            ON anomalies (band, timestamp)
        """)

        c.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── write ──────────────────────────────────────────────────────────

    def store_features(self, bf: BandFeatures) -> None:
        """Insert a feature vector."""
        placeholders = ", ".join(["?"] * N_FEATURES)
        cols = ", ".join(f"f{i}" for i in range(N_FEATURES))
        vals = [bf.band_name, bf.timestamp] + bf.values.tolist() + [
            bf.center_hz, bf.sample_rate_hz
        ]
        self._conn.execute(
            f"INSERT INTO features (band, timestamp, {cols}, center_hz, sample_rate_hz)"
            f" VALUES (?, ?, {placeholders}, ?, ?)",
            vals,
        )

    def store_anomaly(self, ar: AnomalyResult) -> None:
        """Insert an anomaly result (only if score > 0.3 to save space)."""
        if ar.composite < 0.3:
            return
        self._conn.execute(
            "INSERT INTO anomalies (band, timestamp, z_composite, if_score,"
            " composite, triggered) VALUES (?, ?, ?, ?, ?, ?)",
            (
                ar.band_name,
                ar.timestamp,
                ar.z_composite,
                ar.if_score,
                ar.composite,
                json.dumps(ar.triggered_features),
            ),
        )

    def flush(self) -> None:
        """Commit pending writes."""
        if self._conn:
            self._conn.commit()

    # ── read ───────────────────────────────────────────────────────────

    def get_recent_features(
        self, band: str, seconds: float = 3600.0
    ) -> List[Dict]:
        """Retrieve recent feature rows for a band.

        Returns list of dicts with keys: timestamp, f0..f7.
        """
        cutoff = time.time() - seconds
        cols = ", ".join(f"f{i}" for i in range(N_FEATURES))
        rows = self._conn.execute(
            f"SELECT timestamp, {cols} FROM features"
            " WHERE band = ? AND timestamp > ? ORDER BY timestamp",
            (band, cutoff),
        ).fetchall()
        result = []
        for row in rows:
            d = {"timestamp": row[0]}
            for i in range(N_FEATURES):
                d[FEATURE_NAMES[i]] = row[1 + i]
            result.append(d)
        return result

    def get_recent_anomalies(
        self, seconds: float = 86400.0
    ) -> List[Dict]:
        """Retrieve recent anomaly records (all bands)."""
        cutoff = time.time() - seconds
        rows = self._conn.execute(
            "SELECT band, timestamp, z_composite, if_score, composite, triggered"
            " FROM anomalies WHERE timestamp > ? ORDER BY timestamp DESC",
            (cutoff,),
        ).fetchall()
        return [
            {
                "band": r[0],
                "timestamp": r[1],
                "z_composite": r[2],
                "if_score": r[3],
                "composite": r[4],
                "triggered": json.loads(r[5]) if r[5] else [],
            }
            for r in rows
        ]

    def get_feature_stats(self, band: str, seconds: float = 3600.0) -> Optional[Dict]:
        """Compute mean/std of each feature over the last *seconds*."""
        cutoff = time.time() - seconds
        cols_avg = ", ".join(f"AVG(f{i})" for i in range(N_FEATURES))
        cols_std = ", ".join(
            f"AVG(f{i}*f{i}) - AVG(f{i})*AVG(f{i})" for i in range(N_FEATURES)
        )
        row = self._conn.execute(
            f"SELECT COUNT(*), {cols_avg}, {cols_std} FROM features"
            " WHERE band = ? AND timestamp > ?",
            (band, cutoff),
        ).fetchone()
        if not row or row[0] < 10:
            return None
        count = row[0]
        means = [row[1 + i] for i in range(N_FEATURES)]
        variances = [max(0.0, row[1 + N_FEATURES + i]) for i in range(N_FEATURES)]
        stds = [v ** 0.5 for v in variances]
        return {
            "count": count,
            "means": dict(zip(FEATURE_NAMES, means)),
            "stds": dict(zip(FEATURE_NAMES, stds)),
        }

    def prune_old(self, max_age_seconds: float = 7 * 86400.0) -> int:
        """Delete records older than *max_age_seconds*.  Returns rows deleted."""
        cutoff = time.time() - max_age_seconds
        c1 = self._conn.execute(
            "DELETE FROM features WHERE timestamp < ?", (cutoff,)
        ).rowcount
        c2 = self._conn.execute(
            "DELETE FROM anomalies WHERE timestamp < ?", (cutoff,)
        ).rowcount
        self._conn.commit()
        total = c1 + c2
        if total > 0:
            log.info("Pruned %d old records (features=%d, anomalies=%d)", total, c1, c2)
        return total
