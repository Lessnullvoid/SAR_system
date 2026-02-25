"""
Narrative engine for the SAF corridor map scanner.

Organises navigation into **cycles**.  Each cycle builds a manifest of
every notable tile (seismic activity, historical images, news) and visits
them all exactly once.  Each new cycle picks a different traversal
strategy so the presentation feels semi-generative and non-linear.

Cycle structure
───────────────
  1. Build a *manifest* — the set of all tiles that merit a visit.
     Tiles are categorised: ALERT (seismic), ECHO (history), DISPATCH
     (news), SURVEY (corridor fill).
  2. Pick a *strategy* that determines the ordering of chapters.
  3. Generate a playlist of Chapter objects (with BREATH inserts).
  4. Execute the playlist one chapter at a time.
  5. When the playlist is exhausted → start a new cycle with a new strategy.

ML anomaly ALERT chapters can interrupt any cycle at any point.

Chapter types
─────────────
  ALERT    — high sensor score or ML anomaly
  ECHO     — historical event + current sensor state
  DISPATCH — current news event with image overlay
  SURVEY   — corridor exploration of undervisited tiles
  BREATH   — full corridor overview (visual reset)
"""
from __future__ import annotations

import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────────────────────────

class ChapterType(Enum):
    ALERT = auto()
    ECHO = auto()
    DISPATCH = auto()
    SURVEY = auto()
    BREATH = auto()


class ShotKind(Enum):
    ESTABLISH = auto()   # wide regional view
    APPROACH = auto()    # zoom toward primary tile
    FOCUS = auto()       # tile close-up, optional overlay
    CONTEXT = auto()     # pan to a neighbor
    RELEASE = auto()     # zoom back out


@dataclass
class Shot:
    kind: ShotKind
    tile_id: Optional[str] = None
    overlay: Optional[str] = None   # "news", "history", or None
    overlay_source_tile: Optional[str] = None  # tile whose images to display (defaults to tile_id)
    overlay_image_idx: Optional[int] = None    # specific image index in source tile's pool
    hold_ms: int = 1500
    anim_s: float = 0.7

    @property
    def total_ms(self) -> int:
        return int(self.anim_s * 1000) + self.hold_ms


@dataclass
class Chapter:
    chapter_type: ChapterType
    primary_tile: str
    context_tiles: List[str] = field(default_factory=list)
    shots: List[Shot] = field(default_factory=list)

    @property
    def total_ms(self) -> int:
        return sum(s.total_ms for s in self.shots)


# ── Traversal strategies ──────────────────────────────────────────────

class _Strategy(Enum):
    NORTH_SOUTH = auto()
    SOUTH_NORTH = auto()
    SECTION_WALK = auto()
    HOTSPOT_FIRST = auto()
    INTERLEAVED = auto()
    RANDOM_WALK = auto()

_ALL_STRATEGIES = list(_Strategy)

_BREATH_EVERY = 20


# ── Narrative engine ──────────────────────────────────────────────────

class NarrativeEngine:
    """Generates chapter playlists organised into complete cycles."""

    def __init__(self) -> None:
        self._fault_tiles: List[str] = []
        self._fault_km: Dict[str, float] = {}
        self._tile_sections: Dict[str, str] = {}

        self._tile_scores: Dict[str, float] = {}
        self._quake_counts: Dict[str, int] = {}
        self._news_tile_ids: Set[str] = set()
        self._history_tile_ids: Set[str] = set()
        self._tile_has_history_image: Dict[str, bool] = {}
        self._tile_has_news_image: Dict[str, bool] = {}
        self._tile_history_themes: Dict[str, int] = {}
        self._news_image_counts: Dict[str, int] = {}
        self._history_image_counts: Dict[str, int] = {}

        self._playlist: List[Chapter] = []
        self._playlist_idx = 0
        self._cycle_count = 0
        self._chapter_count = 0
        self._strategy_history: deque = deque(maxlen=12)

        self._pending_alert_tiles: List[str] = []
        self._alert_cooldown: Dict[str, float] = {}
        self._MAX_PENDING_ALERTS = 6
        self._ALERT_COOLDOWN_S = 120.0

        self._recent_tiles: List[str] = []
        self._MAX_RECENT = 60

        # Global image pool — one entry per individual image across all tiles
        self._image_pool: List[Tuple[str, str, int]] = []  # (tile_id, type, index)
        self._image_pool_dirty = True
        self._cycle_image_usage: Dict[Tuple[str, str, int], int] = {}
        self._global_unseen: Set[Tuple[str, str, int]] = set()
        self._MAX_PER_IMAGE_PER_CYCLE = 2

    # ── External data updates ─────────────────────────────────────────

    def set_tile_info(
        self,
        fault_tiles: List[str],
        fault_km: Dict[str, float],
        sections: Dict[str, str],
    ) -> None:
        self._fault_tiles = list(fault_tiles)
        self._fault_km = dict(fault_km)
        self._tile_sections = dict(sections)

    def update_scores(self, scores: Dict[str, float]) -> None:
        self._tile_scores = dict(scores)

    def update_quake_counts(self, counts: Dict[str, int]) -> None:
        self._quake_counts = dict(counts)

    def update_news_tiles(
        self, tile_ids: Set[str], tiles_with_images: Set[str],
        image_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        self._news_tile_ids = set(tile_ids)
        for tid in tiles_with_images:
            self._tile_has_news_image[tid] = True
        if image_counts:
            self._news_image_counts.update(image_counts)
        self._image_pool_dirty = True

    def update_history_tiles(
        self, tile_ids: Set[str], tiles_with_images: Set[str],
        theme_counts: Optional[Dict[str, int]] = None,
        image_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        self._history_tile_ids = set(tile_ids)
        for tid in tiles_with_images:
            self._tile_has_history_image[tid] = True
        if theme_counts:
            self._tile_history_themes = dict(theme_counts)
        if image_counts:
            self._history_image_counts.update(image_counts)
        self._image_pool_dirty = True

    def queue_anomaly_alert(self, tile_ids: List[str]) -> None:
        now = time.monotonic()
        added = 0
        for tid in tile_ids:
            if len(self._pending_alert_tiles) >= self._MAX_PENDING_ALERTS:
                break
            cooldown_until = self._alert_cooldown.get(tid, 0.0)
            if now < cooldown_until:
                continue
            if tid in self._pending_alert_tiles:
                continue
            self._pending_alert_tiles.append(tid)
            self._alert_cooldown[tid] = now + self._ALERT_COOLDOWN_S
            added += 1
        if added:
            log.info("Narrative: ML anomaly alert queued %d tiles (pending=%d)",
                     added, len(self._pending_alert_tiles))

    # ── Global image pool ───────────────────────────────────────────

    def _rebuild_image_pool(self) -> None:
        """Build a flat list of every individual image across all tiles."""
        pool: List[Tuple[str, str, int]] = []
        for tid, count in self._news_image_counts.items():
            for idx in range(count):
                pool.append((tid, "news", idx))
        for tid, count in self._history_image_counts.items():
            for idx in range(count):
                pool.append((tid, "history", idx))
        new_images = set(pool) - set(self._image_pool)
        self._image_pool = pool
        self._image_pool_dirty = False
        if not self._global_unseen:
            self._global_unseen = set(pool)
        elif new_images:
            self._global_unseen |= new_images
        log.info("Image pool rebuilt: %d images (%d news, %d history), %d unseen",
                 len(pool),
                 sum(self._news_image_counts.values()),
                 sum(self._history_image_counts.values()),
                 len(self._global_unseen))

    def _begin_cycle_image_plan(self) -> None:
        """Reset per-cycle image tracking and build the cycle's image queue."""
        if self._image_pool_dirty:
            self._rebuild_image_pool()
        self._cycle_image_usage = {}
        self._cycle_image_queue: List[Tuple[str, str, int]] = []

        if not self._global_unseen and self._image_pool:
            self._global_unseen = set(self._image_pool)
            log.info("All %d images shown — resetting unseen pool", len(self._image_pool))

        unseen = [s for s in self._image_pool if s in self._global_unseen]
        seen = [s for s in self._image_pool if s not in self._global_unseen]
        random.shuffle(unseen)
        random.shuffle(seen)
        self._cycle_image_queue = unseen + seen

    def _pick_image_for_tile(
        self, tile_id: str,
    ) -> Optional[Tuple[str, str, int]]:
        """Pick the best available image for a chapter near *tile_id*.

        Prefers: unseen images > geographically close > any available.
        Respects the max-2-per-cycle limit.
        """
        if not self._cycle_image_queue and not self._image_pool:
            return None

        km = self._fault_km.get(tile_id, 0.0)

        def _score(slot: Tuple[str, str, int]) -> float:
            dist = abs(self._fault_km.get(slot[0], 0) - km)
            unseen_bonus = -1000.0 if slot in self._global_unseen else 0.0
            return unseen_bonus + dist

        best = None
        best_score = float("inf")
        best_idx = -1

        search_range = min(len(self._cycle_image_queue), 30)
        for i in range(search_range):
            slot = self._cycle_image_queue[i]
            usage = self._cycle_image_usage.get(slot, 0)
            if usage >= self._MAX_PER_IMAGE_PER_CYCLE:
                continue
            s = _score(slot)
            if s < best_score:
                best = slot
                best_score = s
                best_idx = i

        if best is None:
            for slot in self._image_pool:
                usage = self._cycle_image_usage.get(slot, 0)
                if usage < self._MAX_PER_IMAGE_PER_CYCLE:
                    best = slot
                    break

        if best is None:
            return None

        self._cycle_image_usage[best] = self._cycle_image_usage.get(best, 0) + 1
        self._global_unseen.discard(best)
        if best_idx >= 0:
            self._cycle_image_queue.pop(best_idx)

        return best

    # ── Manifest ──────────────────────────────────────────────────────

    def _build_manifest(self) -> Dict[str, ChapterType]:
        """Categorise notable tiles — with per-cycle randomisation.

        Each cycle varies what is shown:
        - Tiles with BOTH history and news images randomly get ECHO or
          DISPATCH so both content pools surface across cycles.
        - Only a random subset (~60-80%) of seismic-only tiles are
          included, deprioritising recently-visited ones.
        - Survey fill uses a random offset so different corridor gaps
          are explored each time.
        """
        manifest: Dict[str, ChapterType] = {}
        recent_set = set(self._recent_tiles)

        has_hist = set(
            t for t in self._fault_tiles
            if self._tile_has_history_image.get(t, False)
        )
        has_news = set(
            t for t in self._fault_tiles
            if self._tile_has_news_image.get(t, False)
        )
        both = has_hist & has_news
        hist_only = has_hist - both
        news_only = has_news - both

        for tid in both:
            manifest[tid] = random.choice([ChapterType.ECHO, ChapterType.DISPATCH])

        for tid in hist_only:
            manifest[tid] = ChapterType.ECHO

        for tid in news_only:
            manifest[tid] = ChapterType.DISPATCH

        seismic_candidates = [
            t for t in self._fault_tiles
            if t not in manifest
            and (self._tile_scores.get(t, 0.0) >= 0.1
                 or self._quake_counts.get(t, 0) >= 1)
        ]
        fresh = [t for t in seismic_candidates if t not in recent_set]
        stale = [t for t in seismic_candidates if t in recent_set]
        random.shuffle(fresh)
        random.shuffle(stale)
        keep_n = max(3, int(len(seismic_candidates) * random.uniform(0.55, 0.80)))
        selected = (fresh + stale)[:keep_n]
        for tid in selected:
            manifest[tid] = ChapterType.ALERT

        remaining = [t for t in self._fault_tiles if t not in manifest]
        if remaining:
            remaining.sort(key=lambda t: self._fault_km.get(t, 0))
            n_surveys = min(15, len(remaining))
            offset = random.randint(0, max(0, len(remaining) - 1))
            step = max(1, len(remaining) // n_surveys)
            for i in range(n_surveys):
                idx = (offset + i * step) % len(remaining)
                manifest[remaining[idx]] = ChapterType.SURVEY

        log.info(
            "Cycle manifest: %d tiles (ALERT=%d, ECHO=%d, DISPATCH=%d, SURVEY=%d)",
            len(manifest),
            sum(1 for v in manifest.values() if v == ChapterType.ALERT),
            sum(1 for v in manifest.values() if v == ChapterType.ECHO),
            sum(1 for v in manifest.values() if v == ChapterType.DISPATCH),
            sum(1 for v in manifest.values() if v == ChapterType.SURVEY),
        )
        return manifest

    # ── Strategy ──────────────────────────────────────────────────────

    def _pick_strategy(self) -> _Strategy:
        recent = list(self._strategy_history)[-3:]
        available = [s for s in _ALL_STRATEGIES if s not in recent]
        if not available:
            available = list(_ALL_STRATEGIES)
        choice = random.choice(available)
        self._strategy_history.append(choice)
        return choice

    # ── Playlist generation ───────────────────────────────────────────

    def _generate_playlist(self) -> List[Chapter]:
        manifest = self._build_manifest()
        if not manifest:
            return [self._make_breath()]

        strategy = self._pick_strategy()
        self._cycle_count += 1
        self._begin_cycle_image_plan()
        log.info(
            "Cycle #%d starting — strategy=%s, %d tiles, %d images in pool",
            self._cycle_count, strategy.name, len(manifest),
            len(self._image_pool),
        )

        ordered = self._apply_strategy(strategy, manifest)

        playlist: List[Chapter] = []
        for i, (tid, ctype) in enumerate(ordered):
            playlist.append(self._make_chapter(ctype, tid))
            if (i + 1) % _BREATH_EVERY == 0 and i < len(ordered) - 1:
                playlist.append(self._make_breath())

        remaining_unseen = len([
            s for s in self._image_pool
            if self._cycle_image_usage.get(s, 0) == 0
        ])
        log.info(
            "Cycle #%d playlist: %d chapters, %d images assigned, %d unseen remain",
            self._cycle_count, len(playlist),
            sum(self._cycle_image_usage.values()),
            remaining_unseen,
        )
        return playlist

    @staticmethod
    def _jitter_nearby(items: List[Tuple[str, ChapterType]]) -> None:
        """Swap adjacent pairs randomly to break identical sub-ordering."""
        for i in range(0, len(items) - 1, 2):
            if random.random() < 0.4:
                items[i], items[i + 1] = items[i + 1], items[i]

    def _apply_strategy(
        self, strategy: _Strategy, manifest: Dict[str, ChapterType],
    ) -> List[Tuple[str, ChapterType]]:
        items = list(manifest.items())

        if strategy == _Strategy.NORTH_SOUTH:
            items.sort(key=lambda x: self._fault_km.get(x[0], 0))
            self._jitter_nearby(items)

        elif strategy == _Strategy.SOUTH_NORTH:
            items.sort(key=lambda x: self._fault_km.get(x[0], 0), reverse=True)
            self._jitter_nearby(items)

        elif strategy == _Strategy.SECTION_WALK:
            by_section: Dict[str, List] = defaultdict(list)
            for tid, ct in items:
                sec = self._tile_sections.get(tid, "unknown")
                by_section[sec].append((tid, ct))
            sections = list(by_section.keys())
            random.shuffle(sections)
            items = []
            for sec in sections:
                group = by_section[sec]
                random.shuffle(group)
                items.extend(group)

        elif strategy == _Strategy.HOTSPOT_FIRST:
            seismic = [(t, c) for t, c in items if c == ChapterType.ALERT]
            echo = [(t, c) for t, c in items if c == ChapterType.ECHO]
            dispatch = [(t, c) for t, c in items if c == ChapterType.DISPATCH]
            survey = [(t, c) for t, c in items if c == ChapterType.SURVEY]
            seismic.sort(
                key=lambda x: self._tile_scores.get(x[0], 0), reverse=True,
            )
            random.shuffle(echo)
            random.shuffle(dispatch)
            random.shuffle(survey)
            buckets = [seismic, echo, dispatch, survey]
            random.shuffle(buckets[1:])
            items = []
            for b in buckets:
                items.extend(b)

        elif strategy == _Strategy.INTERLEAVED:
            seismic = [
                (t, c) for t, c in items
                if c in (ChapterType.ALERT, ChapterType.ECHO)
            ]
            social = [(t, c) for t, c in items if c == ChapterType.DISPATCH]
            survey = [(t, c) for t, c in items if c == ChapterType.SURVEY]
            random.shuffle(seismic)
            random.shuffle(social)
            random.shuffle(survey)
            merged: List[Tuple[str, ChapterType]] = []
            si, so, su = 0, 0, 0
            while si < len(seismic) or so < len(social) or su < len(survey):
                if si < len(seismic):
                    merged.append(seismic[si]); si += 1
                if so < len(social):
                    merged.append(social[so]); so += 1
                if si < len(seismic):
                    merged.append(seismic[si]); si += 1
                if su < len(survey):
                    merged.append(survey[su]); su += 1
            items = merged

        elif strategy == _Strategy.RANDOM_WALK:
            items.sort(key=lambda x: self._fault_km.get(x[0], 0))
            n_swaps = max(1, len(items) // 3)
            for _ in range(n_swaps):
                i = random.randint(0, max(0, len(items) - 2))
                j = min(i + random.randint(1, 4), len(items) - 1)
                items[i], items[j] = items[j], items[i]

        return items

    # ── Public interface ──────────────────────────────────────────────

    def next_chapter(self) -> Chapter:
        self._chapter_count += 1

        if self._pending_alert_tiles:
            tid = self._pending_alert_tiles.pop(0)
            if tid in self._fault_km:
                ch = self._make_alert(tid)
                self._record_visit(tid)
                log.info(
                    "Chapter #%d: ALERT (ML anomaly) → %s",
                    self._chapter_count, tid,
                )
                return ch

        if self._playlist_idx >= len(self._playlist):
            self._playlist = self._generate_playlist()
            self._playlist_idx = 0

        ch = self._playlist[self._playlist_idx]
        self._playlist_idx += 1
        self._record_visit(ch.primary_tile)

        log.info(
            "Chapter #%d: %s → %s [cycle %d, %d/%d]",
            self._chapter_count, ch.chapter_type.name, ch.primary_tile,
            self._cycle_count, self._playlist_idx, len(self._playlist),
        )
        return ch

    def _record_visit(self, tile_id: str) -> None:
        self._recent_tiles.append(tile_id)
        if len(self._recent_tiles) > self._MAX_RECENT:
            self._recent_tiles = self._recent_tiles[-self._MAX_RECENT:]

    # ── Neighbor lookup ───────────────────────────────────────────────

    def _neighbors(self, tile_id: str, n: int = 2) -> List[str]:
        km = self._fault_km.get(tile_id, 0.0)
        candidates = [
            t for t in self._fault_tiles
            if t != tile_id and abs(self._fault_km.get(t, 0) - km) <= 25
        ]
        random.shuffle(candidates)
        return candidates[:n]

    # ── Chapter builders ──────────────────────────────────────────────

    def _make_chapter(self, ct: ChapterType, primary: str) -> Chapter:
        if ct == ChapterType.ALERT:
            return self._make_alert(primary)
        if ct == ChapterType.ECHO:
            return self._make_echo(primary)
        if ct == ChapterType.DISPATCH:
            return self._make_dispatch(primary)
        if ct == ChapterType.SURVEY:
            return self._make_survey(primary)
        return self._make_breath()

    def _make_image_shot(
        self, tile_id: str, hold_ms: int = 6000, anim_s: float = 0.5,
    ) -> Optional[Shot]:
        """Try to allocate an image from the global pool for this tile."""
        img = self._pick_image_for_tile(tile_id)
        if img is None:
            return None
        return Shot(
            ShotKind.FOCUS, tile_id=tile_id,
            overlay=img[1],
            overlay_source_tile=img[0],
            overlay_image_idx=img[2],
            hold_ms=hold_ms, anim_s=anim_s,
        )

    def _make_alert(self, primary: str) -> Chapter:
        neighbors = self._neighbors(primary, 1)
        shots = [
            Shot(ShotKind.APPROACH, tile_id=primary, hold_ms=0, anim_s=0.6),
            Shot(ShotKind.FOCUS, tile_id=primary, hold_ms=3000, anim_s=0.0),
        ]
        for nb in neighbors:
            shots.append(
                Shot(ShotKind.CONTEXT, tile_id=nb, hold_ms=1500, anim_s=0.4),
            )
        if random.random() < 0.55:
            img_shot = self._make_image_shot(primary, hold_ms=5000)
            if img_shot:
                shots.append(img_shot)
        return Chapter(ChapterType.ALERT, primary, neighbors, shots)

    def _make_echo(self, primary: str) -> Chapter:
        neighbors = self._neighbors(primary, 1)
        shots = [
            Shot(ShotKind.APPROACH, tile_id=primary, hold_ms=0, anim_s=0.7),
        ]
        img_shot = self._make_image_shot(primary, hold_ms=8000, anim_s=0.0)
        if img_shot:
            shots.append(img_shot)
        shots.append(
            Shot(ShotKind.FOCUS, tile_id=primary, overlay=None,
                 hold_ms=3000, anim_s=0.0),
        )
        for nb in neighbors:
            shots.append(
                Shot(ShotKind.CONTEXT, tile_id=nb, hold_ms=2500, anim_s=0.5),
            )
        return Chapter(ChapterType.ECHO, primary, neighbors, shots)

    def _make_dispatch(self, primary: str) -> Chapter:
        shots = [
            Shot(ShotKind.APPROACH, tile_id=primary, hold_ms=0, anim_s=0.7),
        ]
        img_shot = self._make_image_shot(primary, hold_ms=7000, anim_s=0.0)
        if img_shot:
            shots.append(img_shot)
        return Chapter(ChapterType.DISPATCH, primary, [], shots)

    def _make_survey(self, primary: str) -> Chapter:
        km = self._fault_km.get(primary, 0.0)
        nearby = [
            t for t in self._fault_tiles
            if t != primary and abs(self._fault_km.get(t, 0) - km) <= 40
        ]
        random.shuffle(nearby)
        survey_tiles = nearby[:4]

        shots = [
            Shot(ShotKind.ESTABLISH, hold_ms=1200, anim_s=0.7),
        ]
        for tid in [primary] + survey_tiles:
            shots.append(
                Shot(ShotKind.FOCUS, tile_id=tid, hold_ms=2000, anim_s=0.5),
            )
        if random.random() < 0.6:
            img_shot = self._make_image_shot(primary, hold_ms=5000)
            if img_shot:
                shots.append(img_shot)
        return Chapter(ChapterType.SURVEY, primary, survey_tiles, shots)

    def _make_breath(self) -> Chapter:
        mid = (
            self._fault_tiles[len(self._fault_tiles) // 2]
            if self._fault_tiles else ""
        )
        shots = [Shot(ShotKind.ESTABLISH, hold_ms=4000, anim_s=1.0)]
        return Chapter(ChapterType.BREATH, mid, [], shots)
