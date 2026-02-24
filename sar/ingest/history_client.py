"""
Social-history event loader — static historical events along the SAF corridor.

Loads events from data/social_history.json and downloads + caches their
images locally for use as map tile overlays.

Image resolution strategy (in order):
  1. Try the ``image_url`` from the JSON directly.
  2. Search Wikipedia for the event title and grab the lead image thumbnail.
  3. Search Wikipedia for "city + date_year" as a broader fallback.

Data flow
─────────
  data/social_history.json  →  load_history_events()
                             →  List[HistoryEvent]
  download_history_images() →  data/history_cache/{slug}.jpg
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from io import BytesIO
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

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_THUMB_SIZE = 512


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


# ── Wikipedia image resolution helpers ────────────────────────────────

def _wiki_search_thumb(session, query: str) -> Optional[str]:
    """Search Wikipedia and return the lead-image thumbnail URL of the top hit."""
    try:
        resp = session.get(
            _WIKI_API,
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": "1",
                "prop": "pageimages",
                "format": "json",
                "pithumbsize": str(_THUMB_SIZE),
            },
            timeout=20,
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            thumb = page.get("thumbnail", {}).get("source")
            if thumb:
                return thumb
    except Exception as exc:
        log.debug("Wiki search failed for '%s': %s", query, exc)
    return None


def _resolve_image_url(session, ev: HistoryEvent) -> Optional[str]:
    """Try multiple strategies to find a usable image URL for an event."""
    # Strategy 1: direct URL from JSON (if it returns 200)
    if ev.image_url:
        try:
            head = session.head(ev.image_url, timeout=10, allow_redirects=True)
            if head.status_code == 200:
                return ev.image_url
        except Exception:
            pass

    # Strategy 2: search Wikipedia by event title
    url = _wiki_search_thumb(session, ev.title)
    if url:
        return url

    # Strategy 3: broader search with city + year
    year = ev.date[:4] if ev.date else ""
    if year and ev.city:
        url = _wiki_search_thumb(session, f"{ev.city} {year}")
        if url:
            return url

    return None


# ── Main download entry point ─────────────────────────────────────────

def download_history_images(events: List[HistoryEvent]) -> List[HistoryEvent]:
    """Download images for events that lack a local cache.

    Uses a multi-strategy approach: tries the JSON URL first, then falls
    back to Wikipedia API search for the event title or city+year.

    Returns the same list with ``image_path`` fields updated.
    """
    import requests
    from PIL import Image

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = _UA

    downloaded = 0
    skipped = 0
    failed = 0

    for ev in events:
        if ev.image_path:
            skipped += 1
            continue

        slug = _slugify(ev.title)
        dest = _CACHE_DIR / f"{slug}.jpg"

        image_url = _resolve_image_url(session, ev)
        if not image_url:
            failed += 1
            log.debug("No image found for: %s", ev.title)
            continue

        try:
            resp = session.get(image_url, timeout=30)
            resp.raise_for_status()

            img = Image.open(BytesIO(resp.content))
            img = img.convert("L")

            max_dim = _THUMB_SIZE
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)

            img.save(str(dest), "JPEG", quality=85)
            ev.image_path = str(dest)
            downloaded += 1
            log.debug("Downloaded history image: %s → %s", ev.title[:50], dest.name)

            time.sleep(1.0)

        except Exception as exc:
            failed += 1
            log.warning("Failed to download image for '%s': %s", ev.title[:50], exc)

    session.close()
    log.info(
        "History images: %d downloaded, %d cached, %d failed",
        downloaded, skipped, failed,
    )
    return events
