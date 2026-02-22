from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def setup_logging() -> None:
    logfile = LOG_DIR / "sdr_app.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(),
        ],
    )


def write_anomaly_event(event: Dict[str, Any]) -> None:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    path = LOG_DIR / f"anomaly_{ts}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(event, f, indent=2)


