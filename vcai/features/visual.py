"""Visual features from low-res frames sampled at a fixed fps: pacing, motion, color, layout."""
import cv2
import numpy as np
from scipy.signal import find_peaks

from .audio import HOOK_S, per_second

CUT_THRESHOLD = 0.4


def _hsv_hists(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hists, sats = [], []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
        hists.append(h / (h.sum() + 1e-9))
        sats.append(hsv[..., 1].mean() / 255.0)
    return np.array(hists), np.array(sats)


def visual_features(frames: np.ndarray, fps: float, n_sec: int) -> tuple[dict, dict]:
    n = len(frames)
    if n < 3:
        return {"has_video": 0}, {}
    f = frames.astype(np.float32) / 255.0
    gray = f @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    h, w = gray.shape[1:]
    dur = n / fps
    t = np.arange(1, n) / fps  # time stamp of each frame-to-frame transition

    diff = np.abs(np.diff(gray, axis=0))
    motion = diff.mean(axis=(1, 2))
    center = diff[:, h // 4: 3 * h // 4, w // 4: 3 * w // 4].mean(axis=(1, 2))

    hists, sats = _hsv_hists(frames)
    hist_diff = 0.5 * np.abs(np.diff(hists, axis=0)).sum(axis=1)
    cuts, _ = find_peaks(np.concatenate([[0], hist_diff, [0]]), height=CUT_THRESHOLD,
                         distance=max(1, int(0.5 * fps)))
    cuts = cuts - 1  # undo left pad -> index into hist_diff
    cut_t = t[cuts]

    bright = gray.mean(axis=(1, 2))
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorful = np.sqrt(rg.std(axis=(1, 2)) ** 2 + yb.std(axis=(1, 2)) ** 2) + \
        0.3 * np.sqrt(rg.mean(axis=(1, 2)) ** 2 + yb.mean(axis=(1, 2)) ** 2)
    edges = np.mean([(cv2.Canny((gray[i] * 255).astype(np.uint8), 80, 160) > 0).mean()
                     for i in range(0, n, max(1, n // 20))])

    pix_std = gray.std(axis=0)
    col_mean, row_mean = gray.mean(axis=(0, 1)), gray.mean(axis=(0, 2))
    hook = t < HOOK_S
    mm = motion.mean() + 1e-6
    peak_i = int(np.argmax(motion))

    feats = {
        "has_video": 1,
        "motion_mean": float(motion.mean()),
        "motion_std": float(motion.std()),
        "motion_p95": float(np.percentile(motion, 95)),
        "motion_burstiness": float(np.percentile(motion, 95) / mm),
        "motion_peak_time_rel": float(t[peak_i] / dur),
        "hook_motion_ratio": float(motion[hook].mean() / mm) if hook.any() else 1.0,
        "motion_slope": float(np.polyfit(t / dur, motion, 1)[0] / mm),
        "center_motion_ratio": float(center.mean() / mm),
        "cut_count": int(len(cuts)),
        "cut_rate": float(len(cuts) / dur),
        "first_cut_s": float(cut_t[0]) if len(cuts) else float(dur),
        "hook_cut_count": int(np.sum(cut_t < HOOK_S)),
        "brightness_mean": float(bright.mean()),
        "brightness_std": float(bright.std()),
        "flash_count": int(np.sum(np.diff(bright) > 0.12)),
        "saturation_mean": float(sats.mean()),
        "colorfulness_mean": float(colorful.mean()),
        "edge_density": float(edges),
        "static_pixel_ratio": float((pix_std < 0.02).mean()),  # HUD, overlays, letterbox
        "black_border_ratio": float(((col_mean < 0.06).mean() + (row_mean < 0.06).mean()) / 2),
    }
    timeline = {
        "motion": per_second(motion, t, n_sec),
        "cuts": per_second(np.isin(np.arange(len(t)), cuts).astype(np.float32), t, n_sec, agg=np.sum),
        "brightness": per_second(bright[1:], t, n_sec),
    }
    return feats, timeline
