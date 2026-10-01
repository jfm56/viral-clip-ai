import numpy as np

from vcai.features.extract import extract
from vcai.features.media import probe
from tests.conftest import make_video


def test_probe_and_vertical(tmp_path):
    p = make_video(tmp_path / "v.mp4", size="180x320", seconds=5)
    info = probe(p)
    assert (info["width"], info["height"]) == (180, 320)
    assert info["has_audio"]
    assert abs(info["duration"] - 5) < 0.2
    feats, _ = extract(p)
    assert feats["is_vertical"] == 1


def test_audio_burst_is_located(tmp_path):
    p = make_video(tmp_path / "a.mp4", seconds=8, burst_at=4.0)
    feats, tl = extract(p)
    assert feats["a_has_audio"] == 1
    assert abs(feats["a_loud_peak_time_s"] - 4.25) < 0.4
    assert 0.45 < feats["a_loud_peak_time_rel"] < 0.62
    assert feats["a_max_loud_jump_db"] > 15
    assert feats["a_loud_burst_count"] == 1
    assert len(tl["loud_db"]) == 8
    assert int(np.nanargmax(tl["loud_db"])) == 4


def test_hard_cut_detected(tmp_path):
    p = make_video(tmp_path / "c.mp4", seconds=6, cut_at=3, burst_at=None)
    feats, tl = extract(p)
    assert feats["v_cut_count"] == 1
    assert abs(feats["v_first_cut_s"] - 3) <= 0.5
    assert feats["v_hook_cut_count"] == 0
    assert tl["cuts"].sum() == 1


def test_static_vs_moving_motion(tmp_path):
    static = extract(make_video(tmp_path / "s.mp4", seconds=6, color="gray", burst_at=None))[0]
    moving = extract(make_video(tmp_path / "m.mp4", seconds=6))[0]
    assert moving["v_motion_mean"] > static["v_motion_mean"]
    assert static["v_static_pixel_ratio"] > moving["v_static_pixel_ratio"]
