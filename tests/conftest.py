import subprocess
from pathlib import Path

import pytest

from vcai import config
from vcai.features.media import FFMPEG


def make_video(path: Path, *, seconds=8, size="320x180", burst_at=4.0, cut_at=None, color=None) -> Path:
    """Synthetic clip: quiet tone with one loud burst; optional hard color cut."""
    burst = f"if(between(t,{burst_at},{burst_at + 0.5}),0.8*sin(2*PI*880*t),0)" if burst_at is not None else "0"
    audio = f"aevalsrc='0.03*sin(2*PI*440*t)+{burst}':s=16000:d={seconds}"
    if cut_at is None:
        src = f"color=c={color}:" if color else "testsrc2="
        video = ["-f", "lavfi", "-i", f"{src}size={size}:rate=30:d={seconds}"]
        vmap = ["-map", "0:v"]
    else:
        video = ["-f", "lavfi", "-i", f"color=c=red:size={size}:rate=30:d={cut_at}",
                 "-f", "lavfi", "-i", f"color=c=blue:size={size}:rate=30:d={seconds - cut_at}"]
        vmap = ["-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]"]
    n_vid = 1 if cut_at is None else 2
    cmd = [FFMPEG, "-y", "-v", "error", *video, "-f", "lavfi", "-i", audio, *vmap,
           "-map", f"{n_vid}:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)]
    subprocess.run(cmd, check=True)
    return path


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    for name, sub in [("DATA_DIR", ""), ("MEDIA_DIR", "media"), ("TIMELINE_DIR", "timelines"),
                      ("MODEL_DIR", "models"), ("REPORT_DIR", "reports")]:
        monkeypatch.setattr(config, name, tmp_path / sub if sub else tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.sqlite")
    return tmp_path
