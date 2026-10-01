"""Download -> extract features -> save timeline -> delete media, in parallel worker processes."""
import re
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import config, db
from .features.extract import EXTRACTOR_VERSION, extract
from .features.media import FFMPEG

FORMAT = ("best[height<=480][vcodec!=none][acodec!=none]/bv*[height<=480]+ba"
          "/best[height<=720][vcodec!=none][acodec!=none]/best")


def safe_name(clip_id: str) -> str:
    return re.sub(r"[^\w-]", "_", clip_id)


def download(url: str, clip_id: str, media_dir: Path) -> Path:
    import yt_dlp

    stem = safe_name(clip_id)
    media_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": FORMAT, "outtmpl": str(media_dir / f"{stem}.%(ext)s"), "merge_output_format": "mp4",
        "ffmpeg_location": FFMPEG, "quiet": True, "no_warnings": True, "noprogress": True,
        "retries": 3, "socket_timeout": 30, "noplaylist": True,
    }
    if shutil.which("node"):
        opts["js_runtimes"] = {"node": {}}
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    files = [p for p in media_dir.glob(f"{stem}.*") if p.suffix not in (".part", ".ytdl")]
    if not files:
        raise FileNotFoundError(f"download produced no file for {clip_id}")
    return max(files, key=lambda p: p.stat().st_size)


def _work(clip_id: str, url: str, keep_media: bool, media_dir: str, timeline_dir: str):
    path = None
    try:
        path = download(url, clip_id, Path(media_dir))
        feats, timeline = extract(path)
        np.savez_compressed(Path(timeline_dir) / f"{safe_name(clip_id)}.npz", **timeline)
        return clip_id, "ok", feats, None
    except Exception as e:  # noqa: BLE001 - one bad clip must not kill the batch
        msg = f"{type(e).__name__}: {e}"
        if not isinstance(e, (FileNotFoundError, ValueError)):
            msg += " | " + traceback.format_exc(limit=2).replace("\n", " ")
        return clip_id, "error", None, msg[:800]
    finally:
        if path is not None and not keep_media:
            for p in Path(media_dir).glob(f"{safe_name(clip_id)}.*"):
                p.unlink(missing_ok=True)


def pending_clips(conn, limit: int, source: str | None = None, retry_errors: bool = False,
                  min_creator_clips: int = config.MIN_CREATOR_CLIPS) -> list[tuple[str, str]]:
    """Clips still needing features, restricted to creators with enough clips to be labelled.
    Creators with the most clips come first since they give the most reliable baselines."""
    status_filter = "(f.clip_id IS NULL OR f.status='error')" if retry_errors else "f.clip_id IS NULL"
    q = f"""
        WITH cc AS (SELECT source, creator_id, COUNT(*) n FROM clips GROUP BY source, creator_id)
        SELECT c.clip_id, c.url FROM clips c
        JOIN cc ON cc.source=c.source AND cc.creator_id=c.creator_id
        LEFT JOIN clip_features f ON f.clip_id=c.clip_id
        WHERE {status_filter} AND cc.n >= ? {"AND c.source=?" if source else ""}
        ORDER BY cc.n DESC, c.creator_id, c.clip_id
        LIMIT ?"""
    params = [min_creator_clips] + ([source] if source else []) + [limit]
    return [(r[0], r[1]) for r in conn.execute(q, params)]


def process_pending(conn, limit=500, workers=6, keep_media=False, source=None, retry_errors=False, log=print):
    todo = pending_clips(conn, limit, source, retry_errors)
    config.TIMELINE_DIR.mkdir(parents=True, exist_ok=True)
    log(f"processing {len(todo)} clips with {workers} workers")
    ok = err = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_work, cid, url, keep_media, str(config.MEDIA_DIR), str(config.TIMELINE_DIR))
                for cid, url in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            cid, status, feats, error = fut.result()
            db.save_features(conn, cid, EXTRACTOR_VERSION, status, feats, error)
            ok += status == "ok"
            err += status == "error"
            if status == "error":
                log(f"  ! {cid}: {error[:160]}")
            if i % 10 == 0 or i == len(todo):
                conn.commit()
                log(f"  {i}/{len(todo)} done ({ok} ok, {err} errors)")
    conn.commit()
    return ok, err
