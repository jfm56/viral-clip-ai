"""Twitch Helix clip collection.

Strategy: for each top game, pull that game's most-viewed clips in a window, then pull the
full clip history (same window) of a mix of broadcasters found there. The per-broadcaster
pull is what gives us the low-view clips too; without it we'd only ever see winners.
"""
import math
import random
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

import requests

from .. import config, db

BASE = "https://api.twitch.tv/helix"


class TwitchClient:
    def __init__(self, client_id: str, client_secret: str, session: requests.Session | None = None):
        if not client_id or not client_secret:
            raise RuntimeError("TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET are not set (see .env.example)")
        self.client_id = client_id
        self.client_secret = client_secret
        self.s = session or requests.Session()
        self.token = None

    def _fetch_token(self):
        r = self.s.post(
            "https://id.twitch.tv/oauth2/token",
            params={"client_id": self.client_id, "client_secret": self.client_secret,
                    "grant_type": "client_credentials"},
            timeout=30,
        )
        r.raise_for_status()
        self.token = r.json()["access_token"]

    def get(self, path: str, params: dict) -> dict:
        for attempt in range(6):
            if not self.token:
                self._fetch_token()
            r = self.s.get(
                BASE + path, params=params, timeout=30,
                headers={"Client-Id": self.client_id, "Authorization": f"Bearer {self.token}"},
            )
            if r.status_code == 401:
                self.token = None
                continue
            if r.status_code == 429:
                reset = float(r.headers.get("Ratelimit-Reset", time.time() + 5))
                time.sleep(max(1.0, reset - time.time() + 0.5))
                continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"Twitch API kept failing for {path}")

    def paginate(self, path: str, params: dict, max_pages: int):
        cursor = None
        for _ in range(max_pages):
            p = dict(params)
            if cursor:
                p["after"] = cursor
            data = self.get(path, p)
            items = data.get("data", [])
            yield from items
            cursor = data.get("pagination", {}).get("cursor")
            if not cursor or not items:
                break


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def clip_row(c: dict, game_names: dict[str, str]) -> dict:
    return {
        "clip_id": f"tw:{c['id']}",
        "source": "twitch",
        "native_id": c["id"],
        "url": c["url"],
        "creator_id": c.get("broadcaster_id"),
        "creator_name": c.get("broadcaster_name"),
        "game_id": c.get("game_id") or None,
        "game_name": game_names.get(c.get("game_id")),
        "title": c.get("title"),
        "language": c.get("language"),
        "duration_s": c.get("duration"),
        "created_at": c.get("created_at"),
        "views": c.get("view_count"),
        "extra_json": {k: c.get(k) for k in ("video_id", "vod_offset", "is_featured", "creator_id")},
    }


def resolve_game_names(conn, client: TwitchClient) -> int:
    missing = [r[0] for r in conn.execute(
        "SELECT DISTINCT c.game_id FROM clips c LEFT JOIN games g ON g.source='twitch' AND g.game_id=c.game_id "
        "WHERE c.source='twitch' AND c.game_id IS NOT NULL AND g.game_id IS NULL")]
    names = {}
    for i in range(0, len(missing), 100):
        for g in client.get("/games", {"id": missing[i:i + 100]}).get("data", []):
            names[g["id"]] = g["name"]
    if names:
        db.upsert_games(conn, "twitch", names)
    conn.execute(
        "UPDATE clips SET game_name=(SELECT name FROM games g WHERE g.source='twitch' AND g.game_id=clips.game_id) "
        "WHERE source='twitch' AND game_name IS NULL")
    conn.commit()
    return len(names)


def collect_twitch(conn, n_games=40, days_back=30, pages_per_game=3, broadcasters_per_game=15,
                   pages_per_broadcaster=2, seed=0, log=print) -> int:
    client = TwitchClient(config.TWITCH_CLIENT_ID, config.TWITCH_CLIENT_SECRET)
    rng = random.Random(seed)
    end = datetime.now(timezone.utc) - timedelta(days=config.MIN_AGE_DAYS["twitch"])
    start = end - timedelta(days=days_back)
    window = {"started_at": _iso(start), "ended_at": _iso(end), "first": 100}

    top = client.paginate("/games/top", {"first": 100}, max_pages=math.ceil(n_games * 1.5 / 100) + 1)
    games = [g for g in top if g["name"] not in config.NON_GAMING][:n_games]
    game_names = {g["id"]: g["name"] for g in games}
    db.upsert_games(conn, "twitch", game_names)
    log(f"twitch: {len(games)} games, window {window['started_at']} .. {window['ended_at']}")

    seen_broadcasters: set[str] = set()
    total = 0
    for gi, g in enumerate(games, 1):
        clips = list(client.paginate("/clips", {**window, "game_id": g["id"]}, pages_per_game))
        total += db.upsert_clips(conn, [clip_row(c, game_names) for c in clips])

        # Half the most-clipped broadcasters, half randomly sampled from the long tail.
        counts = Counter(c["broadcaster_id"] for c in clips)
        head_n = math.ceil(broadcasters_per_game / 2)
        head = [b for b, _ in counts.most_common(head_n)]
        tail = [b for b in counts if b not in head]
        rng.shuffle(tail)
        picks = [b for b in head + tail[: broadcasters_per_game - len(head)] if b not in seen_broadcasters]

        for b in picks:
            seen_broadcasters.add(b)
            bclips = list(client.paginate("/clips", {**window, "broadcaster_id": b}, pages_per_broadcaster))
            total += db.upsert_clips(conn, [clip_row(c, game_names) for c in bclips])
        log(f"  [{gi}/{len(games)}] {g['name']}: {len(clips)} game clips, {len(picks)} broadcasters -> {total} rows")

    resolve_game_names(conn, client)
    return total


def refresh_twitch_stats(conn, log=print) -> int:
    client = TwitchClient(config.TWITCH_CLIENT_ID, config.TWITCH_CLIENT_SECRET)
    ids = [r[0] for r in conn.execute("SELECT native_id FROM clips WHERE source='twitch'")]
    rows = []
    for i in range(0, len(ids), 100):
        rows += [clip_row(c, {}) for c in client.get("/clips", {"id": ids[i:i + 100]}).get("data", [])]
    n = db.upsert_clips(conn, rows)
    log(f"twitch: refreshed {n}/{len(ids)} clips")
    return n
