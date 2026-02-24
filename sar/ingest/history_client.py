"""
Social-history event loader — static historical events along the SAF corridor.

Loads events from data/social_history.json and downloads + caches their
Wikipedia Commons images locally for use as map tile overlays.

Data flow
─────────
  data/social_history.json  →  load_history_events()
                             →  List[HistoryEvent]
  HistoryEvent.image_url    →  download_history_images()
                             →  data/history_cache/{slug}.jpg
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_JSON_FILE = _DATA_DIR / "social_history.json"
_CACHE_DIR = _DATA_DIR / "history_cache"

_UA = (
    "SAR-System/1.0 (https://github.com/Lessnullvoid/SAR_system; "
    "educational research project) python-requests"
)


@dataclass
class HistoryEvent:
    date: str
    city: str
    county: str
    lat: float
    lon: float
    title: str
    description: str
    image_url: str = ""
    image_path: str = ""
    theme: str = ""
    period: str = ""


def _slugify(text: str) -> str:
    """Turn a title into a filesystem-safe slug."""
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s[:80].strip("_")


def load_history_events() -> List[HistoryEvent]:
    """Read the social-history JSON and return a list of HistoryEvent."""
    import json

    if not _JSON_FILE.exists():
        log.warning("History JSON not found: %s", _JSON_FILE)
        return []

    with open(_JSON_FILE, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    events: List[HistoryEvent] = []
    for item in raw:
        ev = HistoryEvent(
            date=item.get("date", ""),
            city=item.get("city", ""),
            county=item.get("county", ""),
            lat=item.get("latitude", 0.0),
            lon=item.get("longitude", 0.0),
            title=item.get("title", ""),
            description=item.get("description", ""),
            image_url=item.get("image_url", ""),
            theme=item.get("theme", ""),
            period=item.get("period", ""),
        )
        slug = _slugify(ev.title)
        cached = _CACHE_DIR / f"{slug}.jpg"
        if cached.exists():
            ev.image_path = str(cached)
        events.append(ev)

    log.info("Loaded %d history events from %s", len(events), _JSON_FILE.name)
    return events


def download_history_images(events: List[HistoryEvent]) -> List[HistoryEvent]:
    """Download Wikipedia Commons images for events that lack a local cache.

    Returns the same list with ``image_path`` fields updated.
    """
    import requests
    from PIL import Image
    from io import BytesIO

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0

    for ev in events:
        if ev.image_path:
            skipped += 1
            continue
        if not ev.image_url:
            continue

        slug = _slugify(ev.title)
        dest = _CACHE_DIR / f"{slug}.jpg"

        try:
            resp = requests.get(
                ev.image_url,
                headers={"User-Agent": _UA},
                timeout=30,
            )
            resp.raise_for_status()

            img = Image.open(BytesIO(resp.content))
            img = img.convert("L")  # grayscale for consistency with news images

            max_dim = 512
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)

            img.save(str(dest), "JPEG", quality=85)
            ev.image_path = str(dest)
            downloaded += 1
            log.debug("Downloaded history image: %s", dest.name)

        except Exception as exc:
            failed += 1
            log.warning("Failed to download %s: %s", ev.image_url, exc)

    log.info(
        "History images: %d downloaded, %d cached, %d failed",
        downloaded, skipped, failed,
    )
    return events
