"""Full per-clip feature extraction: clip-level aggregates plus a per-second timeline.

The per-second timeline is saved alongside features; it is the input for the later
"find the viral moment inside a user's long video" model.
"""
import math
from pathlib import Path

import numpy as np

from .audio import SR, audio_features
from .media import probe, read_audio, read_frames
from .visual import visual_features

EXTRACTOR_VERSION = "v1"
FRAME_FPS = 4
FRAME_WIDTH = 160


def _z(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=np.nanmean(x) if np.isfinite(x).any() else 0.0)
    return (x - x.mean()) / (x.std() + 1e-6)


def cross_features(tl: dict, n_sec: int) -> dict:
    if "loud_db" not in tl or "motion" not in tl or n_sec < 4:
        return {}
    lz, mz = _z(tl["loud_db"]), _z(tl["motion"])
    combo = lz + mz
    return {
        "loud_motion_corr": float(np.corrcoef(lz, mz)[0, 1]) if lz.std() > 0 and mz.std() > 0 else 0.0,
        "climax_time_rel": float((np.argmax(combo) + 0.5) / n_sec),
        "joint_peak_seconds": int(np.sum((lz > 1) & (mz > 1))),
    }


def extract(path: Path) -> tuple[dict, dict]:
    info = probe(path)
    frames = read_frames(path, FRAME_FPS, FRAME_WIDTH, info["width"], info["height"])
    duration = info["duration"] or len(frames) / FRAME_FPS
    n_sec = max(1, math.ceil(duration))

    feats = {
        "duration_s": float(duration),
        "width": info["width"],
        "height": info["height"],
        "aspect_ratio": info["width"] / info["height"],
        "is_vertical": int(info["height"] > info["width"] * 1.1),
        "src_fps": info["fps"],
    }
    a_feats, a_tl = audio_features(read_audio(path, SR), n_sec) if info["has_audio"] else ({"has_audio": 0}, {})
    v_feats, v_tl = visual_features(frames, FRAME_FPS, n_sec)
    timeline = {**a_tl, **v_tl}
    feats |= {f"a_{k}": v for k, v in a_feats.items()}
    feats |= {f"v_{k}": v for k, v in v_feats.items()}
    feats |= {f"x_{k}": v for k, v in cross_features(timeline, n_sec).items()}
    return feats, timeline
