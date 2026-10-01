"""YouTube Data API v3 collection of gaming Shorts.

Quota: 10,000 units/day by default. search.list costs 100; videos/channels/playlistItems cost 1.
Strategy: per game, one search sorted by date (an unbiased sample, including flops) and one sorted
by viewCount (the winners). Then pull each channel's recent uploads so every creator gets a baseline.
"""
import re
from datetime import datetime, timedelta, timezone

import requests

from .. import config, db

BASE = "https://www.googleapis.com/youtube/v3"
COST = {"search": 100}
_DUR = re.compile(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?$")


class QuotaExhausted(Exception):
    pass


def parse_iso_duration(s: str | None) -> float | None:
    m = _DUR.match(s or "")
    if not m:
        return None
    d, h, mi, sec = (float(x) if x else 0.0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + sec


class YouTubeClient:
    def __init__(self, api_key: str, budget: int, session: requests.Session | None = None):
        if not api_key:
            raise RuntimeError("YOUTUBE_API_KEY is not set (see .env.example)")
        self.key = api_key
        self.budget = budget
        self.used = 0
        self.s = session or requests.Session()

    def can_spend(self, units: int) -> bool:
        return self.used + units <= self.budget

    def get(self, endpoint: str, params: dict) -> dict:
        cost = COST.get(endpoint, 1)
        if not self.can_spend(cost):
            raise QuotaExhausted(f"budget {self.budget} reached")
        r = self.s.get(f"{BASE}/{endpoint}", params={**params, "key": self.key}, timeout=30)
        if r.status_code == 403 and "quota" in r.text.lower():
            raise QuotaExhausted(r.text[:200])
        r.raise_for_status()
        self.used += cost
        return r.json()

    def videos(self, ids: list[str]) -> list[dict]:
        out = []
        for i in range(0, len(ids), 50):
            out += self.get("videos", {"part": "snippet,statistics,contentDetails",
                                       "id": ",".join(ids[i:i + 50])}).get("items", [])
        return out

    def channels(self, ids: list[str]) -> list[dict]:
        out = []
        for i in range(0, len(ids), 50):
            out += self.get("channels", {"part": "statistics,contentDetails",
                                         "id": ",".join(ids[i:i + 50])}).get("items", [])
        return out


def _int(x):
    return int(x) if x is not None else None


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def match_game(text: str, games: list[str]) -> str | None:
    t = text.lower()
    # Longest names first so "Call of Duty Warzone" beats "Call of Duty".
    for g in sorted(games, key=len, reverse=True):
        if g.lower() in t:
            return g
    return None


def video_row(v: dict, subs: dict[str, int | None], game: str | None, game_source: str) -> dict:
    sn, st = v.get("snippet", {}), v.get("statistics", {})
    return {
        "clip_id": f"yt:{v['id']}",
        "source": "youtube",
        "native_id": v["id"],
        "url": f"https://www.youtube.com/shorts/{v['id']}",
        "creator_id": sn.get("channelId"),
        "creator_name": sn.get("channelTitle"),
        "game_name": game,
        "title": sn.get("title"),
        "language": sn.get("defaultAudioLanguage") or sn.get("defaultLanguage"),
        "duration_s": parse_iso_duration(v.get("contentDetails", {}).get("duration")),
        "created_at": sn.get("publishedAt"),
        "views": _int(st.get("viewCount")),
        "likes": _int(st.get("likeCount")),
        "comments": _int(st.get("commentCount")),
        "creator_followers": subs.get(sn.get("channelId")),
        "extra_json": {"tags": sn.get("tags", [])[:30], "category_id": sn.get("categoryId"),
                       "game_source": game_source},
    }


def _is_short(v: dict, max_duration: float) -> bool:
    d = parse_iso_duration(v.get("contentDetails", {}).get("duration"))
    return d is not None and 0 < d <= max_duration and v.get("snippet", {}).get("liveBroadcastContent") == "none"


def _game_list(conn) -> list[str]:
    names = [r[0] for r in conn.execute("SELECT name FROM games WHERE source='twitch' AND name IS NOT NULL")]
    names = [n for n in names if n not in config.NON_GAMING]
    return names or list(config.DEFAULT_GAMES)


def collect_youtube(conn, days_back=30, budget=9000, max_duration=180, max_games=None,
                    detail_reserve=1500, log=print) -> int:
    yt = YouTubeClient(config.YOUTUBE_API_KEY, budget)
    games = _game_list(conn)[: max_games or None]
    end = datetime.now(timezone.utc) - timedelta(days=config.MIN_AGE_DAYS["youtube"])
    start = end - timedelta(days=days_back)

    # 1) Search: game -> candidate video ids
    found: dict[str, str] = {}
    try:
        for g in games:
            for order in ("date", "viewCount"):
                if not yt.can_spend(100 + detail_reserve):
                    raise QuotaExhausted("search budget used")
                res = yt.get("search", {
                    "part": "id", "type": "video", "videoCategoryId": "20", "videoDuration": "short",
                    "maxResults": 50, "q": f"{g} #shorts", "order": order,
                    "publishedAfter": _iso(start), "publishedBefore": _iso(end),
                })
                for it in res.get("items", []):
                    found.setdefault(it["id"]["videoId"], g)
            log(f"  youtube search '{g}': {len(found)} ids so far (quota {yt.used})")
    except QuotaExhausted as e:
        log(f"  youtube: stopping search ({e})")

    total = 0
    try:
        # 2) Details for search hits
        vids = [v for v in yt.videos(list(found)) if _is_short(v, max_duration)]
        chan_ids = sorted({v["snippet"]["channelId"] for v in vids})
        chans = yt.channels(chan_ids)
        subs = {c["id"]: (None if c["statistics"].get("hiddenSubscriberCount")
                          else _int(c["statistics"].get("subscriberCount"))) for c in chans}
        total += db.upsert_clips(conn, [video_row(v, subs, found[v["id"]], "search_query") for v in vids])
        log(f"youtube: {len(vids)} shorts from search across {len(chan_ids)} channels")

        # 3) Channel baselines: recent uploads of each channel (2 units per channel)
        oldest = end - timedelta(days=days_back * 3)
        for i, c in enumerate(chans, 1):
            if not yt.can_spend(2):
                raise QuotaExhausted("baseline budget used")
            uploads = c.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
            if not uploads:
                continue
            items = yt.get("playlistItems", {"part": "contentDetails", "playlistId": uploads,
                                             "maxResults": 50}).get("items", [])
            ids = [it["contentDetails"]["videoId"] for it in items]
            rows = []
            for v in yt.videos(ids):
                pub = datetime.fromisoformat(v["snippet"]["publishedAt"].replace("Z", "+00:00"))
                if _is_short(v, max_duration) and oldest <= pub <= end:
                    text = v["snippet"].get("title", "") + " " + " ".join(v["snippet"].get("tags", []))
                    rows.append(video_row(v, subs, match_game(text, games), "title_match"))
            total += db.upsert_clips(conn, rows)
            if i % 25 == 0:
                log(f"  channel baselines {i}/{len(chans)} (quota {yt.used})")
    except QuotaExhausted as e:
        log(f"  youtube: stopping early ({e})")

    log(f"youtube: {total} rows written, quota used {yt.used}/{budget}")
    return total


def refresh_youtube_stats(conn, budget=2000, log=print) -> int:
    yt = YouTubeClient(config.YOUTUBE_API_KEY, budget)
    existing = {r["native_id"]: dict(r) for r in conn.execute(
        "SELECT native_id, game_name, creator_followers, extra_json FROM clips WHERE source='youtube'")}
    rows = []
    try:
        for v in yt.videos(list(existing)):
            e = existing[v["id"]]
            row = video_row(v, {v["snippet"]["channelId"]: e["creator_followers"]}, e["game_name"], "refresh")
            row["extra_json"] = e["extra_json"]
            rows.append(row)
    except QuotaExhausted as ex:
        log(f"  youtube refresh stopped: {ex}")
    n = db.upsert_clips(conn, rows)
    log(f"youtube: refreshed {n}/{len(existing)} clips")
    return n
