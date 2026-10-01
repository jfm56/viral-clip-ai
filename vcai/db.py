"""SQLite storage: clip metadata, view-count snapshots, extracted features."""
import json
import sqlite3
from datetime import datetime, timezone

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
    clip_id TEXT PRIMARY KEY,          -- "tw:<id>" or "yt:<id>"
    source TEXT NOT NULL,              -- twitch | youtube
    native_id TEXT NOT NULL,
    url TEXT NOT NULL,
    creator_id TEXT,
    creator_name TEXT,
    game_id TEXT,
    game_name TEXT,
    title TEXT,
    language TEXT,
    duration_s REAL,
    created_at TEXT,
    views INTEGER,
    likes INTEGER,
    comments INTEGER,
    creator_followers INTEGER,         -- YouTube subscribers (NULL for Twitch)
    extra_json TEXT,
    first_seen_at TEXT,
    stats_updated_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_clips_creator ON clips(source, creator_id);

CREATE TABLE IF NOT EXISTS clip_stats (
    clip_id TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    views INTEGER, likes INTEGER, comments INTEGER,
    PRIMARY KEY (clip_id, observed_at)
);

CREATE TABLE IF NOT EXISTS games (
    source TEXT NOT NULL, game_id TEXT NOT NULL, name TEXT,
    PRIMARY KEY (source, game_id)
);

CREATE TABLE IF NOT EXISTS clip_features (
    clip_id TEXT PRIMARY KEY,
    extracted_at TEXT,
    extractor_version TEXT,
    status TEXT,                       -- ok | error
    error TEXT,
    features_json TEXT
);
"""

CLIP_COLS = [
    "clip_id", "source", "native_id", "url", "creator_id", "creator_name", "game_id", "game_name",
    "title", "language", "duration_s", "created_at", "views", "likes", "comments",
    "creator_followers", "extra_json", "first_seen_at", "stats_updated_at",
]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(path=None) -> sqlite3.Connection:
    path = path or config.DB_PATH
    if str(path) != ":memory:":
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def upsert_clips(conn: sqlite3.Connection, rows: list[dict]) -> int:
    if not rows:
        return 0
    now = utcnow_iso()
    prepared = []
    for r in rows:
        r = {c: r.get(c) for c in CLIP_COLS}
        r["first_seen_at"] = now
        r["stats_updated_at"] = now
        if isinstance(r["extra_json"], dict):
            r["extra_json"] = json.dumps(r["extra_json"])
        prepared.append(r)
    cols = ", ".join(CLIP_COLS)
    vals = ", ".join(f":{c}" for c in CLIP_COLS)
    conn.executemany(
        f"""INSERT INTO clips ({cols}) VALUES ({vals})
            ON CONFLICT(clip_id) DO UPDATE SET
              views=excluded.views, likes=excluded.likes, comments=excluded.comments,
              title=excluded.title,
              creator_followers=COALESCE(excluded.creator_followers, clips.creator_followers),
              game_id=COALESCE(excluded.game_id, clips.game_id),
              game_name=COALESCE(excluded.game_name, clips.game_name),
              stats_updated_at=excluded.stats_updated_at""",
        prepared,
    )
    conn.executemany(
        "INSERT OR IGNORE INTO clip_stats (clip_id, observed_at, views, likes, comments) "
        "VALUES (:clip_id, :stats_updated_at, :views, :likes, :comments)",
        prepared,
    )
    conn.commit()
    return len(prepared)


def upsert_games(conn, source: str, games: dict[str, str]) -> None:
    conn.executemany(
        "INSERT INTO games (source, game_id, name) VALUES (?, ?, ?) "
        "ON CONFLICT(source, game_id) DO UPDATE SET name=excluded.name",
        [(source, gid, name) for gid, name in games.items()],
    )
    conn.commit()


def save_features(conn, clip_id: str, version: str, status: str, features: dict | None, error: str | None):
    conn.execute(
        """INSERT INTO clip_features (clip_id, extracted_at, extractor_version, status, error, features_json)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(clip_id) DO UPDATE SET extracted_at=excluded.extracted_at,
             extractor_version=excluded.extractor_version, status=excluded.status,
             error=excluded.error, features_json=excluded.features_json""",
        (clip_id, utcnow_iso(), version, status, error, json.dumps(features) if features else None),
    )
