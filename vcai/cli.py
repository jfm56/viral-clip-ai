"""Command line: python -m vcai <command>"""
import argparse

from . import config, db


def cmd_status(conn, _):
    for src, n, creators in conn.execute(
            "SELECT source, COUNT(*), COUNT(DISTINCT creator_id) FROM clips GROUP BY source"):
        print(f"{src:>8}: {n} clips, {creators} creators")
    for status, n in conn.execute("SELECT status, COUNT(*) FROM clip_features GROUP BY status"):
        print(f"features {status}: {n}")
    from .dataset import build_dataset
    df = build_dataset(conn)
    print(f"labelled + featured (trainable): {len(df)}")


def cmd_collect_twitch(conn, a):
    from .collect.twitch import collect_twitch
    collect_twitch(conn, n_games=a.games, days_back=a.days, pages_per_game=a.pages,
                   broadcasters_per_game=a.broadcasters, pages_per_broadcaster=a.broadcaster_pages)


def cmd_collect_youtube(conn, a):
    from .collect.youtube import collect_youtube
    collect_youtube(conn, days_back=a.days, budget=a.budget, max_games=a.games)


def cmd_refresh(conn, a):
    if config.TWITCH_CLIENT_ID:
        from .collect.twitch import refresh_twitch_stats
        refresh_twitch_stats(conn)
    if config.YOUTUBE_API_KEY:
        from .collect.youtube import refresh_youtube_stats
        refresh_youtube_stats(conn, budget=a.budget)


def cmd_process(conn, a):
    from .process import process_pending
    process_pending(conn, limit=a.limit, workers=a.workers, keep_media=a.keep_media,
                    source=a.source, retry_errors=a.retry_errors)


def cmd_train(conn, a):
    from .train import train
    train(conn, min_rows=a.min_rows)


def main(argv=None):
    p = argparse.ArgumentParser(prog="vcai", description="Viral game-clip learning pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="counts of collected / processed / trainable clips")

    t = sub.add_parser("collect-twitch", help="pull clip metadata from Twitch Helix")
    t.add_argument("--games", type=int, default=40)
    t.add_argument("--days", type=int, default=30)
    t.add_argument("--pages", type=int, default=3, help="pages of 100 clips per game")
    t.add_argument("--broadcasters", type=int, default=15, help="broadcasters to deep-pull per game")
    t.add_argument("--broadcaster-pages", type=int, default=2)

    y = sub.add_parser("collect-youtube", help="pull gaming Shorts metadata from YouTube Data API")
    y.add_argument("--days", type=int, default=30)
    y.add_argument("--budget", type=int, default=9000, help="quota units to spend (daily limit 10k)")
    y.add_argument("--games", type=int, default=None, help="limit number of games searched")

    r = sub.add_parser("refresh-stats", help="re-fetch view counts (builds view-velocity history)")
    r.add_argument("--budget", type=int, default=2000)

    pr = sub.add_parser("process", help="download clips + extract audio/visual features")
    pr.add_argument("--limit", type=int, default=500)
    pr.add_argument("--workers", type=int, default=6)
    pr.add_argument("--source", choices=["twitch", "youtube"])
    pr.add_argument("--keep-media", action="store_true", help="keep downloaded video files")
    pr.add_argument("--retry-errors", action="store_true")

    tr = sub.add_parser("train", help="train virality models and write report")
    tr.add_argument("--min-rows", type=int, default=200)

    a = p.parse_args(argv)
    conn = db.connect()
    {
        "status": cmd_status, "collect-twitch": cmd_collect_twitch, "collect-youtube": cmd_collect_youtube,
        "refresh-stats": cmd_refresh, "process": cmd_process, "train": cmd_train,
    }[a.cmd](conn, a)
