"""Build the modelling table: virality labels + metadata/title features + extracted A/V features.

Label ("viral_score"): how much better a clip did than that creator's typical clip,
    log1p(views) - leave-one-out median of log1p(views) over the creator's other clips,
then de-trended for clip age (per source). A score of +1.1 = ~3x the creator's normal views.
This takes creator size out of the target so the model learns what makes a CLIP go viral,
not which channel is famous.
"""
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config

HYPE_WORDS = {
    "clutch", "ace", "insane", "crazy", "wtf", "omg", "impossible", "unbelievable", "epic", "rage",
    "nuke", "headshot", "victory", "destroyed", "broke", "glitch", "bug", "lmao", "lol", "funniest",
    "fail", "funny", "best", "worst", "world record", "wr", "no way", "bro", "rip", "noscope", "401",
}
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
_VS = re.compile(r"\b1v[1-5]\b", re.I)
_HASHTAG = re.compile(r"#\w+")


def title_features(title: str | None) -> dict:
    t = title or ""
    letters = [c for c in t if c.isalpha()]
    low = t.lower()
    return {
        "title_len": len(t),
        "title_words": len(t.split()),
        "title_caps_ratio": sum(c.isupper() for c in letters) / len(letters) if letters else 0.0,
        "title_exclaims": t.count("!"),
        "title_question": int("?" in t),
        "title_emojis": len(_EMOJI.findall(t)),
        "title_has_number": int(any(c.isdigit() for c in t)),
        "title_hashtags": len(_HASHTAG.findall(t)),
        "title_hype_words": sum(w in low for w in HYPE_WORDS),
        "title_1vX": int(bool(_VS.search(t))),
    }


def loo_median(values: np.ndarray) -> np.ndarray:
    """Median of every other element, for each element."""
    n = len(values)
    out = np.empty(n)
    for i in range(n):
        out[i] = np.median(np.delete(values, i))
    return out


def _parse_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce", format="ISO8601")


def load_frame(conn) -> pd.DataFrame:
    df = pd.read_sql_query(
        """SELECT c.*, f.features_json FROM clips c
           JOIN clip_features f ON f.clip_id=c.clip_id AND f.status='ok'""", conn)
    if df.empty:
        return df
    feats = pd.DataFrame([json.loads(s) for s in df.pop("features_json")], index=df.index)
    # Metadata duration is authoritative; drop the decoded duplicate unless metadata is missing.
    df["duration_s"] = df["duration_s"].fillna(feats.pop("duration_s"))
    return pd.concat([df, feats], axis=1)


def build_dataset(conn, now: datetime | None = None) -> pd.DataFrame:
    df = load_frame(conn)
    if df.empty:
        return df
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    created = _parse_ts(df["created_at"])
    observed = _parse_ts(df["stats_updated_at"]).fillna(now)
    df["age_days"] = (observed - created).dt.total_seconds() / 86400
    df = df[df["views"].notna() & df["age_days"].notna()].copy()
    df = df[df.apply(lambda r: r["age_days"] >= config.MIN_AGE_DAYS.get(r["source"], 3), axis=1)]
    df = df[~df["game_name"].isin(config.NON_GAMING)]

    df["log_views"] = np.log1p(df["views"].astype(float))
    df["log_age_days"] = np.log(df["age_days"])
    df["n_creator_clips"] = df.groupby(["source", "creator_id"])["clip_id"].transform("count")
    df = df[df["n_creator_clips"] >= config.MIN_CREATOR_CLIPS].copy()
    if df.empty:
        return df

    df["creator_baseline"] = np.nan
    for _, idx in df.groupby(["source", "creator_id"]).groups.items():
        df.loc[idx, "creator_baseline"] = loo_median(df.loc[idx, "log_views"].to_numpy())
    raw = df["log_views"] - df["creator_baseline"]

    # Remove the part of the score explained by age (older clips have had longer to accumulate).
    df["viral_score"] = raw
    for src, idx in df.groupby("source").groups.items():
        x, y = df.loc[idx, "log_age_days"].to_numpy(), raw.loc[idx].to_numpy()
        if len(idx) >= 20 and np.ptp(x) > 0:
            slope, icpt = np.polyfit(x, y, 1)
            df.loc[idx, "viral_score"] = y - (slope * x + icpt)
    df["views_multiple"] = np.exp(raw)  # plain-English "x times the creator's usual views"
    df["breakout"] = df.groupby("source")["viral_score"].transform(
        lambda s: (s >= s.quantile(config.BREAKOUT_QUANTILE)).astype(int))

    created = _parse_ts(df["created_at"])
    df["post_hour_utc"] = created.dt.hour
    df["post_weekday"] = created.dt.weekday
    df["is_youtube"] = (df["source"] == "youtube").astype(int)
    df["lang_en"] = df["language"].fillna("").str.lower().str.startswith("en").astype(int)
    df["log_creator_baseline_views"] = df["creator_baseline"]
    df["log_duration"] = np.log1p(df["duration_s"].astype(float))
    df = pd.concat([df, pd.DataFrame([title_features(t) for t in df["title"]], index=df.index)], axis=1)
    return df.reset_index(drop=True)


# Feature groups. "content" = only things our clip generator controls when cutting a user's video.
CONTENT_PREFIXES = ("a_", "v_", "x_")
CONTENT_EXTRA = ["duration_s", "log_duration", "aspect_ratio", "is_vertical", "is_youtube", "game"]
CONTEXT = ["post_hour_utc", "post_weekday", "lang_en", "log_creator_baseline_views", "log_age_days"]
TITLE = list(title_features("").keys())


def feature_columns(df: pd.DataFrame, feature_set: str) -> list[str]:
    content = [c for c in df.columns if c.startswith(CONTENT_PREFIXES)] + \
        [c for c in CONTENT_EXTRA if c in df.columns]
    if feature_set == "content":
        return content
    if feature_set == "full":
        return content + [c for c in CONTEXT + TITLE if c in df.columns]
    raise ValueError(feature_set)


def add_game_category(df: pd.DataFrame, top_n: int = 40) -> pd.DataFrame:
    g = df["game_name"].fillna("unknown")
    keep = set(g.value_counts().index[:top_n])
    df["game"] = pd.Categorical(g.where(g.isin(keep), "other"))
    return df
