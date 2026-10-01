from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from vcai import db, process
from vcai.collect.youtube import match_game, parse_iso_duration
from vcai.dataset import build_dataset, loo_median, title_features
from vcai.train import train
from tests.conftest import make_video


@pytest.mark.parametrize("s,expected", [("PT45S", 45), ("PT1M5S", 65), ("PT2M", 120), ("P0D", 0),
                                        ("PT1H", 3600), ("PT12.5S", 12.5), ("garbage", None)])
def test_parse_iso_duration(s, expected):
    assert parse_iso_duration(s) == expected


def test_loo_median():
    assert list(loo_median(np.array([1.0, 2, 3, 100]))) == [3, 3, 2, 2]


def test_title_features():
    f = title_features("INSANE 1v5 CLUTCH!! 🔥 #valorant")
    assert f["title_1vX"] == 1 and f["title_exclaims"] == 2 and f["title_emojis"] == 1
    assert f["title_hashtags"] == 1 and f["title_hype_words"] >= 2 and f["title_caps_ratio"] > 0.5


def test_match_game_prefers_longest():
    games = ["Call of Duty", "Call of Duty Warzone", "Minecraft"]
    assert match_game("crazy call of duty warzone win", games) == "Call of Duty Warzone"
    assert match_game("nothing here", games) is None


def _clip(i, creator, views, age_days, now, game="Valorant"):
    return {"clip_id": f"tw:c{i}", "source": "twitch", "native_id": f"c{i}", "url": f"https://x/{i}",
            "creator_id": creator, "creator_name": creator, "game_id": "1", "game_name": game,
            "title": "clip", "language": "en", "duration_s": 30.0,
            "created_at": (now - timedelta(days=age_days)).strftime("%Y-%m-%dT%H:%M:%SZ"), "views": views}


def test_labels_are_creator_relative(tmp_data):
    conn = db.connect()
    now = datetime.now(timezone.utc)
    # Big creator: typical 10k views, one clip at 100k. Small creator: typical 100, one at 1k.
    rows = [_clip(i, "big", 10_000, 10, now) for i in range(6)] + [_clip(6, "big", 100_000, 10, now)]
    rows += [_clip(10 + i, "small", 100, 10, now) for i in range(6)] + [_clip(16, "small", 1_000, 10, now)]
    db.upsert_clips(conn, rows)
    for r in rows:
        db.save_features(conn, r["clip_id"], "t", "ok", {"duration_s": 30.0, "a_x": 1.0}, None)
    df = build_dataset(conn, now=now).set_index("clip_id")
    # Both 10x outliers should score the same despite 100x creator size difference.
    assert df.loc["tw:c6", "views_multiple"] == pytest.approx(10, rel=0.01)
    assert df.loc["tw:c16", "views_multiple"] == pytest.approx(10, rel=0.02)
    assert df.loc["tw:c0", "viral_score"] == pytest.approx(0, abs=0.01)


def test_train_recovers_planted_signal(tmp_data):
    """Virality driven by a_max_loud_jump_db + noise; the model should find it and beat chance."""
    rng = np.random.default_rng(0)
    conn = db.connect()
    now = datetime.now(timezone.utc)
    rows, feats = [], {}
    for i in range(600):
        creator = f"cr{i % 40}"
        size = 10 ** (2 + (i % 40) / 10)  # creators from 100 to ~10^6 typical views
        jump = rng.uniform(0, 30)
        noise = {f"v_noise{k}": float(rng.normal()) for k in range(8)}
        views = int(size * np.exp(0.08 * (jump - 15) + rng.normal(0, 0.4)))
        rows.append(_clip(i, creator, views, rng.uniform(4, 30), now, game=["A", "B", "C"][i % 3]))
        feats[f"tw:c{i}"] = {"duration_s": 30.0, "a_max_loud_jump_db": jump, **noise}
    db.upsert_clips(conn, rows)
    for cid, f in feats.items():
        db.save_features(conn, cid, "t", "ok", f, None)
    conn.commit()

    metrics = train(conn, min_rows=100, log=lambda *_: None, now=now)
    assert metrics["all_content"]["spearman"] > 0.5
    assert metrics["all_content"]["breakout_auc"] > 0.75
    report = (tmp_data / "reports" / "latest.md").read_text(encoding="utf-8")
    drivers = report.split("## Drivers - all_content")[1]
    assert drivers.index("a_max_loud_jump_db") < drivers.index("v_noise0")
    assert "higher = more viral" in drivers.split("\n")[4]  # top row, planted positive effect


def test_process_worker_end_to_end(tmp_data, monkeypatch):
    """Download is stubbed with a synthetic clip; everything after is real."""
    src = make_video(tmp_data / "src.mp4", seconds=6)

    def fake_download(url, clip_id, media_dir):
        media_dir.mkdir(parents=True, exist_ok=True)
        dst = media_dir / f"{process.safe_name(clip_id)}.mp4"
        dst.write_bytes(src.read_bytes())
        return dst

    monkeypatch.setattr(process, "download", fake_download)
    (tmp_data / "timelines").mkdir()
    cid, status, feats, err = process._work("tw:abc-1", "u", False, str(tmp_data / "media"),
                                            str(tmp_data / "timelines"))
    assert status == "ok", err
    assert feats["v_has_video"] == 1 and feats["a_has_audio"] == 1
    assert (tmp_data / "timelines" / "tw_abc-1.npz").exists()
    assert not list((tmp_data / "media").glob("*"))  # media deleted after extraction
