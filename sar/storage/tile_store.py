"""
SQLite-backed storage for tile states.

Stores time-series snapshots of each tile's sensor state so we can
track changes over time and replay history.

Usage
-----
    store = TileStore()
    store.save_state(tile_state)
    recent = store.get_recent("SAF_023", hours=24)
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..geo.tile_state import TileState

log = logging.getLogger(__name__)

_DEFAULT_DB = Path(__file__).resolve().parent.parent.parent / "data" / "sar_tiles.db"


class TileStore:
    """Persistent storage for tile state snapshots.

    Thread-safe: uses check_same_thread=False and serialises writes
    through a lock so background sensor threads can call save_*().
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._path = db_path or _DEFAULT_DB
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._path), check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._write_lock = threading.Lock()
        self._pending_writes = 0
        self._FLUSH_EVERY = 20
        self._create_tables()
        log.info("TileStore opened: %s", self._path)

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS tile_snapshots (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                tile_id     TEXT    NOT NULL,
                timestamp   REAL    NOT NULL,
                composite   REAL    NOT NULL,
                data_json   TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tile_ts
                ON tile_snapshots(tile_id, timestamp);

            CREATE TABLE IF NOT EXISTS tile_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                tile_id     TEXT    NOT NULL,
                timestamp   REAL    NOT NULL,
                event_type  TEXT    NOT NULL,
                magnitude   REAL    DEFAULT 0,
                detail_json TEXT    DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_event_ts
                ON tile_events(tile_id, timestamp);
        """)
        self._conn.commit()

    def _auto_flush(self) -> None:
        self._pending_writes += 1
        if self._pending_writes >= self._FLUSH_EVERY:
            self._conn.commit()
            self._pending_writes = 0

    def save_state(self, state: TileState) -> None:
        """Store a tile state snapshot (thread-safe, batched commits)."""
        with self._write_lock:
            self._conn.execute(
                "INSERT INTO tile_snapshots (tile_id, timestamp, composite, data_json) "
                "VALUES (?, ?, ?, ?)",
                (
                    state.tile_id,
                    state.timestamp,
                    state.composite_score,
                    json.dumps(state.as_dict()),
                ),
            )
            self._auto_flush()

    def save_event(
        self,
        tile_id: str,
        event_type: str,
        magnitude: float = 0.0,
        detail: Optional[Dict] = None,
    ) -> None:
        """Store a discrete event (thread-safe, batched commits)."""
        with self._write_lock:
            self._conn.execute(
                "INSERT INTO tile_events (tile_id, timestamp, event_type, magnitude, detail_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (tile_id, time.time(), event_type, magnitude, json.dumps(detail or {})),
            )
            self._auto_flush()

    def get_recent_states(
        self, tile_id: str, hours: float = 24.0
    ) -> List[Dict]:
        """Get recent state snapshots for a tile."""
        cutoff = time.time() - hours * 3600
        rows = self._conn.execute(
            "SELECT timestamp, composite, data_json FROM tile_snapshots "
            "WHERE tile_id = ? AND timestamp > ? ORDER BY timestamp",
            (tile_id, cutoff),
        ).fetchall()
        return [
            {"timestamp": r[0], "composite": r[1], **json.loads(r[2])}
            for r in rows
        ]

    def get_latest_states(self) -> Dict[str, Dict]:
        """Get the most recent state for every tile."""
        rows = self._conn.execute(
            "SELECT tile_id, timestamp, composite, data_json "
            "FROM tile_snapshots ts1 "
            "WHERE timestamp = (SELECT MAX(timestamp) FROM tile_snapshots ts2 "
            "                    WHERE ts2.tile_id = ts1.tile_id)"
        ).fetchall()
        return {
            r[0]: {"timestamp": r[1], "composite": r[2], **json.loads(r[3])}
            for r in rows
        }

    def flush(self) -> None:
        with self._write_lock:
            self._conn.commit()
            self._pending_writes = 0

    def prune_old(self, days: float = 30.0) -> None:
        """Remove data older than *days*."""
        cutoff = time.time() - days * 86400
        with self._write_lock:
            self._conn.execute(
                "DELETE FROM tile_snapshots WHERE timestamp < ?", (cutoff,)
            )
            self._conn.execute(
                "DELETE FROM tile_events WHERE timestamp < ?", (cutoff,)
            )
            self._conn.commit()
            self._pending_writes = 0

    def close(self) -> None:
        self.flush()
        try:
            self._conn.close()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
