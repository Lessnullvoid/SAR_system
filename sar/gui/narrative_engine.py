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
from collections import defaultdict
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

        self._playlist: List[Chapter] = []
        self._playlist_idx = 0
        self._cycle_count = 0
        self._chapter_count = 0
        self._strategy_history: List[_Strategy] = []

        self._pending_alert_tiles: List[str] = []

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
    ) -> None:
        self._news_tile_ids = set(tile_ids)
        for tid in tiles_with_images:
            self._tile_has_news_image[tid] = True

    def update_history_tiles(
        self, tile_ids: Set[str], tiles_with_images: Set[str],
        theme_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        self._history_tile_ids = set(tile_ids)
        for tid in tiles_with_images:
            self._tile_has_history_image[tid] = True
        if theme_counts:
            self._tile_history_themes = dict(theme_counts)

    def queue_anomaly_alert(self, tile_ids: List[str]) -> None:
        for tid in tile_ids:
            if tid not in self._pending_alert_tiles:
                self._pending_alert_tiles.append(tid)
        log.info("Narrative: ML anomaly alert queued for %d tiles", len(tile_ids))

    # ── Manifest ──────────────────────────────────────────────────────

    def _build_manifest(self) -> Dict[str, ChapterType]:
        """Categorise every notable tile into a chapter type.

        Returns tile_id -> ChapterType for every tile that should be
        visited in this cycle.
        """
        manifest: Dict[str, ChapterType] = {}

        for tid in self._fault_tiles:
            score = self._tile_scores.get(tid, 0.0)
            quakes = self._quake_counts.get(tid, 0)
            if score >= 0.1 or quakes >= 1:
                if self._tile_has_history_image.get(tid, False):
                    manifest[tid] = ChapterType.ECHO
                else:
                    manifest[tid] = ChapterType.ALERT

        for tid in self._fault_tiles:
            if tid in manifest:
                continue
            if self._tile_has_history_image.get(tid, False):
                manifest[tid] = ChapterType.ECHO

        for tid in self._fault_tiles:
            if tid in manifest:
                continue
            if self._tile_has_news_image.get(tid, False):
                manifest[tid] = ChapterType.DISPATCH

        remaining = [t for t in self._fault_tiles if t not in manifest]
        if remaining:
            remaining.sort(key=lambda t: self._fault_km.get(t, 0))
            step = max(1, len(remaining) // 15)
            for i in range(0, len(remaining), step):
                manifest[remaining[i]] = ChapterType.SURVEY

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
        recent = self._strategy_history[-3:]
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
        log.info(
            "Cycle #%d starting — strategy=%s, %d tiles",
            self._cycle_count, strategy.name, len(manifest),
        )

        ordered = self._apply_strategy(strategy, manifest)

        playlist: List[Chapter] = []
        for i, (tid, ctype) in enumerate(ordered):
            playlist.append(self._make_chapter(ctype, tid))
            if (i + 1) % _BREATH_EVERY == 0 and i < len(ordered) - 1:
                playlist.append(self._make_breath())
        return playlist

    def _apply_strategy(
        self, strategy: _Strategy, manifest: Dict[str, ChapterType],
    ) -> List[Tuple[str, ChapterType]]:
        items = list(manifest.items())

        if strategy == _Strategy.NORTH_SOUTH:
            items.sort(key=lambda x: self._fault_km.get(x[0], 0))

        elif strategy == _Strategy.SOUTH_NORTH:
            items.sort(key=lambda x: self._fault_km.get(x[0], 0), reverse=True)

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
                group.sort(key=lambda x: self._fault_km.get(x[0], 0))
                items.extend(group)

        elif strategy == _Strategy.HOTSPOT_FIRST:
            seismic = [(t, c) for t, c in items if c == ChapterType.ALERT]
            echo = [(t, c) for t, c in items if c == ChapterType.ECHO]
            dispatch = [(t, c) for t, c in items if c == ChapterType.DISPATCH]
            survey = [(t, c) for t, c in items if c == ChapterType.SURVEY]
            seismic.sort(
                key=lambda x: self._tile_scores.get(x[0], 0), reverse=True,
            )
            echo.sort(key=lambda x: self._fault_km.get(x[0], 0))
            dispatch.sort(key=lambda x: self._fault_km.get(x[0], 0))
            random.shuffle(survey)
            items = seismic + echo + dispatch + survey

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

        log.info(
            "Chapter #%d: %s → %s [cycle %d, %d/%d]",
            self._chapter_count, ch.chapter_type.name, ch.primary_tile,
            self._cycle_count, self._playlist_idx, len(self._playlist),
        )
        return ch

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

    def _make_alert(self, primary: str) -> Chapter:
        neighbors = self._neighbors(primary, 2)
        shots = [
            Shot(ShotKind.ESTABLISH, hold_ms=1200, anim_s=0.7),
            Shot(ShotKind.APPROACH, tile_id=primary, hold_ms=0, anim_s=0.7),
            Shot(ShotKind.FOCUS, tile_id=primary, hold_ms=4000, anim_s=0.0),
        ]
        for nb in neighbors:
            shots.append(
                Shot(ShotKind.CONTEXT, tile_id=nb, hold_ms=2000, anim_s=0.5),
            )
        return Chapter(ChapterType.ALERT, primary, neighbors, shots)

    def _make_echo(self, primary: str) -> Chapter:
        neighbors = self._neighbors(primary, 1)
        shots = [
            Shot(ShotKind.APPROACH, tile_id=primary, hold_ms=0, anim_s=0.7),
            Shot(ShotKind.FOCUS, tile_id=primary, overlay="history",
                 hold_ms=6000, anim_s=0.0),
            Shot(ShotKind.FOCUS, tile_id=primary, overlay=None,
                 hold_ms=3000, anim_s=0.0),
        ]
        for nb in neighbors:
            shots.append(
                Shot(ShotKind.CONTEXT, tile_id=nb, hold_ms=2000, anim_s=0.5),
            )
        return Chapter(ChapterType.ECHO, primary, neighbors, shots)

    def _make_dispatch(self, primary: str) -> Chapter:
        shots = [
            Shot(ShotKind.APPROACH, tile_id=primary, hold_ms=0, anim_s=0.7),
            Shot(ShotKind.FOCUS, tile_id=primary, overlay="news",
                 hold_ms=5000, anim_s=0.0),
        ]
        return Chapter(ChapterType.DISPATCH, primary, [], shots)

    def _make_survey(self, primary: str) -> Chapter:
        km = self._fault_km.get(primary, 0.0)
        nearby = [
            t for t in self._fault_tiles
            if t != primary and abs(self._fault_km.get(t, 0) - km) <= 40
        ]
        nearby.sort(key=lambda t: self._fault_km.get(t, 0))
        survey_tiles = nearby[:4]

        shots = [
            Shot(ShotKind.ESTABLISH, hold_ms=1200, anim_s=0.7),
        ]
        for tid in [primary] + survey_tiles:
            shots.append(
                Shot(ShotKind.FOCUS, tile_id=tid, hold_ms=2000, anim_s=0.5),
            )
        return Chapter(ChapterType.SURVEY, primary, survey_tiles, shots)

    def _make_breath(self) -> Chapter:
        mid = (
            self._fault_tiles[len(self._fault_tiles) // 2]
            if self._fault_tiles else ""
        )
        shots = [Shot(ShotKind.ESTABLISH, hold_ms=4000, anim_s=1.0)]
        return Chapter(ChapterType.BREATH, mid, [], shots)
