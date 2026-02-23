"""
ICE news feed client — fetches immigration enforcement news near the SAF corridor.

Uses the GDELT DOC 2.0 API (free, no API key required) to search for articles
about ICE raids, immigration enforcement, deportation, and detention near
cities along the San Andreas Fault corridor.

Historical backfill
───────────────────
On first run the client fetches ALL articles from 2025-01-20 (inauguration)
through today in weekly chunks and stores them locally.  Subsequent runs
only fetch the latest 7 days and merge/de-duplicate with the local archive.

Articles are geocoded by matching city names in the title, then mapped to
tiles.  Article images (``socialimage``) are downloaded and cached locally
as high-contrast grayscale PNGs for use as tile overlays when zoomed in.

Data flow
─────────
  GDELT API  →  fetch_ice_news() / backfill_historical()
             →  List[ICEArticle]  →  save_articles() (append + dedup)
  ICEArticle.image_url  →  download_article_image()
                         →  data/news_cache/{hash}.png
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── Cache directory ───────────────────────────────────────────────────

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "news_cache"
_ARTICLES_FILE = _CACHE_DIR / "articles.json"
_BACKFILL_FILE = _CACHE_DIR / "backfill_progress.json"

# GDELT DOC 2.0 API
_GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Rate-limit cooldown: skip requests until this timestamp
_rate_limit_until: float = 0.0

# Historical start date — inauguration day
_HISTORY_START = datetime(2025, 1, 20, tzinfo=timezone.utc)

# ── Cities / towns along the San Andreas Fault corridor ──────────────
# Mapped to (lat, lon) for geocoding articles by title match.

_SAF_CITIES: Dict[str, Tuple[float, float]] = {
    # Southern section
    "Salton Sea":       (33.30, -115.85),
    "Palm Springs":     (33.83, -116.55),
    "Coachella":        (33.68, -116.17),
    "Indio":            (33.72, -116.22),
    "San Bernardino":   (34.11, -117.29),
    "Riverside":        (33.95, -117.40),
    "Fontana":          (34.09, -117.44),
    "Redlands":         (34.06, -117.18),
    "Cajon Pass":       (34.32, -117.45),
    "Imperial":         (32.85, -115.57),
    "El Centro":        (32.79, -115.56),
    "Calexico":         (32.68, -115.50),
    "Brawley":          (32.98, -115.53),
    "Hemet":            (33.75, -116.97),
    "Perris":           (33.78, -117.23),
    "Moreno Valley":    (33.94, -117.23),
    "Temecula":         (33.49, -117.15),
    "Murrieta":         (33.55, -117.21),
    "Corona":           (33.88, -117.57),
    # Central section
    "Palmdale":         (34.58, -118.12),
    "Lancaster":        (34.70, -118.14),
    "Tehachapi":        (35.13, -118.45),
    "Bakersfield":      (35.37, -119.02),
    "Taft":             (35.14, -119.46),
    "Coalinga":         (36.14, -120.36),
    "Parkfield":        (35.90, -120.43),
    "Paso Robles":      (35.63, -120.69),
    "Fresno":           (36.74, -119.77),
    "Visalia":          (36.33, -119.29),
    "Madera":           (36.96, -120.06),
    "Hanford":          (36.33, -119.64),
    "Porterville":      (36.07, -119.02),
    "Delano":           (35.77, -119.25),
    "Wasco":            (35.59, -119.34),
    # Northern section
    "Hollister":        (36.85, -121.40),
    "Salinas":          (36.68, -121.66),
    "San Jose":         (37.34, -121.89),
    "San Francisco":    (37.77, -122.42),
    "Oakland":          (37.80, -122.27),
    "Berkeley":         (37.87, -122.27),
    "Santa Rosa":       (38.44, -122.71),
    "Ukiah":            (39.15, -123.21),
    "Eureka":           (40.80, -124.16),
    "Santa Cruz":       (36.97, -122.03),
    "Watsonville":      (36.91, -121.77),
    "Gilroy":           (37.00, -121.57),
    "Fremont":          (37.55, -121.98),
    "Hayward":          (37.67, -122.08),
    "Richmond":         (37.94, -122.35),
    "San Rafael":       (37.97, -122.53),
    "Petaluma":         (38.23, -122.64),
    "Napa":             (38.30, -122.29),
    # Greater LA area (close to SAF southern reach)
    "Los Angeles":      (34.05, -118.24),
    "Pasadena":         (34.15, -118.14),
    "Pomona":           (34.06, -117.75),
    "Ontario":          (34.07, -117.65),
    "Long Beach":       (33.77, -118.19),
    "Santa Ana":        (33.75, -117.87),
    "Anaheim":          (33.84, -117.91),
    "Irvine":           (33.68, -117.83),
    "Glendale":         (34.14, -118.26),
    "Burbank":          (34.18, -118.31),
    "Torrance":         (33.84, -118.34),
    "Inglewood":        (33.96, -118.35),
    "Downey":           (33.94, -118.13),
    "Compton":          (33.90, -118.22),
    "El Monte":         (34.07, -118.03),
    "West Covina":      (34.07, -117.94),
    "San Diego":        (32.72, -117.16),
    "Chula Vista":      (32.64, -117.08),
    "Escondido":        (33.12, -117.09),
    "Oceanside":        (33.20, -117.38),
    "Oxnard":           (34.20, -119.18),
    "Ventura":          (34.27, -119.23),
    "Santa Barbara":    (34.42, -119.70),
    "San Luis Obispo":  (35.28, -120.66),
    # Broader area terms (mapped to central California)
    "Southern California": (34.05, -118.24),
    "Northern California": (37.77, -122.42),
    "Central California":  (36.00, -119.50),
    "Central Valley":      (36.50, -119.60),
    "Bay Area":            (37.60, -122.10),
    "Inland Empire":       (34.00, -117.40),
    "Silicon Valley":      (37.40, -121.95),
    "SoCal":               (34.05, -118.24),
    "NorCal":              (37.77, -122.42),
}


@dataclass
class ICEArticle:
    """One news article about ICE activity."""
    title: str
    url: str
    image_url: str = ""
    date: str = ""              # GDELT seendate format (YYYYMMDDTHHmmSSZ)
    lat: float = 0.0
    lon: float = 0.0
    city: str = ""              # matched city name
    source: str = ""            # domain
    local_image_path: str = ""  # path to cached image


# ── GDELT fetch ──────────────────────────────────────────────────────

# Simple query — GDELT DOC 2.0 needs short queries for historical ranges.
# Quoted phrases can cause issues with some date ranges.
_QUERY = '(ICE OR deportation OR migrant) California'


def _gdelt_fetch(
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
    timespan: Optional[str] = None,
    max_records: int = 250,
) -> List[dict]:
    """Low-level GDELT DOC 2.0 query.  Returns raw article dicts."""
    global _rate_limit_until
    import requests

    if time.time() < _rate_limit_until:
        remaining = int(_rate_limit_until - time.time())
        log.info("GDELT cooldown active, skipping request (%ds remaining)", remaining)
        return []

    params: dict = {
        "query": _QUERY,
        "mode": "ArtList",
        "format": "JSON",
        "maxrecords": str(max_records),
    }
    if start_dt and end_dt:
        params["startdatetime"] = start_dt.strftime("%Y%m%d%H%M%S")
        params["enddatetime"] = end_dt.strftime("%Y%m%d%H%M%S")
    elif timespan:
        params["timespan"] = timespan

    resp = None
    for attempt, timeout in enumerate([45, 60], 1):
        try:
            resp = requests.get(_GDELT_URL, params=params, timeout=timeout)
            if resp.status_code == 429:
                _rate_limit_until = time.time() + 600
                log.warning("GDELT 429 rate-limited — cooling off for 10 minutes")
                return []
            resp.raise_for_status()
            break
        except Exception as exc:
            log.warning("GDELT attempt %d failed: %s", attempt, exc)
            resp = None

    if resp is None:
        return []

    try:
        data = resp.json()
    except Exception:
        body = resp.content[:200].decode("utf-8", errors="replace")
        log.warning("GDELT non-JSON (%d bytes): %s", len(resp.content), body)
        return []

    return data.get("articles", [])


def _parse_articles(raw_articles: List[dict]) -> List[ICEArticle]:
    """Parse raw GDELT dicts into ICEArticle objects, geocoded to SAF."""
    results: List[ICEArticle] = []
    seen_urls: set = set()

    for art in raw_articles:
        url = art.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        title = art.get("title", "")
        image_url = art.get("socialimage", "")
        date_str = art.get("seendate", "")
        domain = art.get("domain", "")

        city, lat, lon = _match_city(title, url)
        if not city:
            continue

        results.append(ICEArticle(
            title=title, url=url, image_url=image_url,
            date=date_str, lat=lat, lon=lon,
            city=city, source=domain,
        ))

    return results


def fetch_ice_news(max_records: int = 250) -> List[ICEArticle]:
    """Fetch recent ICE news (last 7 days) from GDELT.

    This is the lightweight poll used on the 10-min timer.
    """
    raw = _gdelt_fetch(timespan="7d", max_records=max_records)
    if not raw:
        return []
    articles = _parse_articles(raw)
    log.info("GDELT recent: %d articles matched SAF corridor (from %d)",
             len(articles), len(raw))
    return articles


def backfill_historical(
    progress_callback=None,
) -> List[ICEArticle]:
    """Fetch ALL articles from 2025-01-20 to today in weekly chunks.

    Merges with existing local archive, de-duplicates by URL.
    Saves progress so it can resume if interrupted.

    Parameters
    ----------
    progress_callback : callable, optional
        Called with (week_num, total_weeks, articles_so_far) for status updates.

    Returns
    -------
    list[ICEArticle]
        Complete merged archive.
    """
    now = datetime.now(timezone.utc)
    start = _HISTORY_START

    # Load progress to resume where we left off
    done_weeks = _load_backfill_progress()

    # Compute weekly chunks
    weeks: List[Tuple[datetime, datetime]] = []
    cursor = start
    while cursor < now:
        week_end = min(cursor + timedelta(days=7), now)
        weeks.append((cursor, week_end))
        cursor = week_end

    total_weeks = len(weeks)
    all_new: List[ICEArticle] = []

    for i, (w_start, w_end) in enumerate(weeks):
        week_key = w_start.strftime("%Y%m%d")

        if week_key in done_weeks:
            if progress_callback:
                progress_callback(i + 1, total_weeks, len(all_new))
            continue

        log.info("Backfill week %d/%d: %s → %s",
                 i + 1, total_weeks,
                 w_start.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d"))

        raw = _gdelt_fetch(start_dt=w_start, end_dt=w_end, max_records=250)
        articles = _parse_articles(raw)
        all_new.extend(articles)

        # Mark this week as done
        done_weeks.add(week_key)
        _save_backfill_progress(done_weeks)

        if progress_callback:
            progress_callback(i + 1, total_weeks, len(all_new))

        # GDELT rate-limits aggressively — 15s between backfill requests
        time.sleep(15)

    log.info("Backfill complete: %d new articles from %d weeks",
             len(all_new), total_weeks)
    return all_new


def _match_city(title: str, url: str) -> Tuple[str, float, float]:
    """Scan title and URL for known SAF corridor city/area names.

    Since the GDELT query already filters to California, we accept any
    article that mentions California or any known location.  Articles
    that don't match any specific city get assigned to a central California
    point so they still appear on the map.
    """
    text = (title + " " + url).lower()
    # Try specific cities first (more precise location)
    for city, (lat, lon) in _SAF_CITIES.items():
        if city.lower() in text:
            return city, lat, lon
    # Broad California match — the GDELT query already filters to CA,
    # so if we got the article at all, it's California-relevant
    ca_terms = ["california", "calif.", "calif ", " ca ", "golden state"]
    for term in ca_terms:
        if term in text:
            return "California", 35.60, -119.90
    # Last resort: if nothing matched, still accept it (GDELT already
    # filtered to California) — place at central CA
    return "California", 35.60, -119.90


# ── Image download ───────────────────────────────────────────────────

def _image_cache_path(image_url: str) -> Path:
    """Deterministic cache path for an image URL."""
    h = hashlib.md5(image_url.encode()).hexdigest()[:16]
    return _CACHE_DIR / f"{h}.png"


_TILE_IMG_SIZE = 512  # all news images are square at this resolution


def download_article_image(article: ICEArticle) -> Optional[Path]:
    """Download and cache the article's social image.

    Center-cropped to a square, resized to 512x512, high-contrast grayscale.
    Every image has the same 1:1 aspect ratio for stable map display.
    """
    if not article.image_url:
        return None

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _image_cache_path(article.image_url)

    if out_path.exists():
        article.local_image_path = str(out_path)
        return out_path

    import requests
    from PIL import Image
    from io import BytesIO
    import numpy as np

    try:
        resp = requests.get(article.image_url, timeout=15)
        if resp.status_code != 200 or len(resp.content) < 500:
            return None

        img = Image.open(BytesIO(resp.content)).convert("L")

        # Center-crop to square (keeps the most important central area)
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))

        # Resize to fixed 512x512
        img = img.resize((_TILE_IMG_SIZE, _TILE_IMG_SIZE), Image.LANCZOS)

        arr = np.array(img, dtype=np.float32)

        # Histogram stretch for maximum contrast
        p2 = np.percentile(arr, 2)
        p98 = np.percentile(arr, 98)
        if p98 - p2 > 10:
            arr = (arr - p2) / (p98 - p2) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        img = Image.fromarray(arr, mode="L")
        img.save(out_path, "PNG")
        article.local_image_path = str(out_path)
        return out_path

    except Exception as exc:
        log.debug("Image download failed for %s: %s", article.image_url[:60], exc)
        return None


def download_all_images(articles: List[ICEArticle], max_workers: int = 4) -> int:
    """Download images for all articles in parallel.

    Returns the count of successfully cached images.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    success = 0
    to_download = []
    for a in articles:
        if not a.image_url:
            continue
        if a.local_image_path and Path(a.local_image_path).exists():
            continue
        a.local_image_path = ""
        to_download.append(a)

    if not to_download:
        return sum(1 for a in articles if a.local_image_path)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_article_image, art): art
            for art in to_download
        }
        for future in as_completed(futures):
            try:
                if future.result():
                    success += 1
            except Exception:
                pass

    total = success + sum(1 for a in articles if a.local_image_path and a not in to_download)
    log.info("News images: %d/%d cached", total, len(articles))
    return total


# ── Article persistence (append + dedup) ─────────────────────────────

def save_articles(articles: List[ICEArticle]) -> None:
    """Persist articles to JSON, merging with existing archive."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_articles()

    # Merge: dedup by URL
    by_url = {a.url: a for a in existing}
    for a in articles:
        by_url[a.url] = a  # new data overwrites old (may have image path now)

    merged = sorted(by_url.values(), key=lambda a: a.date, reverse=True)

    try:
        data = [asdict(a) for a in merged]
        _ARTICLES_FILE.write_text(json.dumps(data, indent=2) + "\n")
        log.info("Saved %d articles to local archive (was %d, added %d new)",
                 len(merged), len(existing), len(merged) - len(existing))
    except Exception as exc:
        log.warning("Failed to save articles: %s", exc)


def load_articles() -> List[ICEArticle]:
    """Load all cached articles from disk."""
    if not _ARTICLES_FILE.exists():
        return []
    try:
        data = json.loads(_ARTICLES_FILE.read_text())
        articles = []
        for d in data:
            a = ICEArticle(**{k: v for k, v in d.items() if k in ICEArticle.__dataclass_fields__})
            if a.local_image_path and not Path(a.local_image_path).exists():
                a.local_image_path = ""
            articles.append(a)
        log.info("Loaded %d articles from local archive", len(articles))
        return articles
    except Exception as exc:
        log.warning("Failed to load articles: %s", exc)
        return []


# ── Backfill progress tracking ───────────────────────────────────────

def _load_backfill_progress() -> set:
    """Load set of completed week keys (YYYYMMDD start dates)."""
    if not _BACKFILL_FILE.exists():
        return set()
    try:
        data = json.loads(_BACKFILL_FILE.read_text())
        return set(data.get("done_weeks", []))
    except Exception:
        return set()


def _save_backfill_progress(done_weeks: set) -> None:
    """Save backfill progress."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _BACKFILL_FILE.write_text(
            json.dumps({"done_weeks": sorted(done_weeks)}) + "\n"
        )
    except Exception as exc:
        log.warning("Failed to save backfill progress: %s", exc)


def is_backfill_complete() -> bool:
    """Check if the historical backfill is done."""
    done = _load_backfill_progress()
    now = datetime.now(timezone.utc)
    cursor = _HISTORY_START
    while cursor < now:
        week_key = cursor.strftime("%Y%m%d")
        if week_key not in done:
            return False
        cursor += timedelta(days=7)
    return True
