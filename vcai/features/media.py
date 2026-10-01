"""ffmpeg-based decoding (uses the static binary bundled with imageio-ffmpeg)."""
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

_DUR = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_SIZE = re.compile(r"\b(\d{2,5})x(\d{2,5})\b")
_FPS = re.compile(r"(\d+(?:\.\d+)?) fps")
_ROT = re.compile(r"rotation of (-?\d+(?:\.\d+)?) degrees")


def probe(path: Path) -> dict:
    err = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)],
                         capture_output=True, text=True, errors="replace").stderr
    video_line = next((ln for ln in err.splitlines() if "Video:" in ln), None)
    if video_line is None:
        raise ValueError(f"no video stream in {path}")
    w, h = map(int, _SIZE.search(video_line).groups())
    rot = _ROT.search(err)
    if rot and abs(abs(float(rot.group(1))) - 90) < 1:
        w, h = h, w
    d = _DUR.search(err)
    duration = int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3)) if d else None
    fps = _FPS.search(video_line)
    return {
        "width": w, "height": h, "duration": duration,
        "fps": float(fps.group(1)) if fps else None,
        "has_audio": any("Audio:" in ln for ln in err.splitlines()),
    }


def read_audio(path: Path, sr: int = 16000) -> np.ndarray:
    out = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
                          "-f", "s16le", "-"], capture_output=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


def read_frames(path: Path, fps: float, width: int, src_w: int, src_h: int) -> np.ndarray:
    """Decode at a low fixed fps and small size, preserving aspect ratio. Returns (n, h, w, 3) uint8 RGB."""
    h = max(2, int(round(width * src_h / src_w / 2)) * 2)
    buf = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-an",
                          "-vf", f"fps={fps},scale={width}:{h}:flags=area",
                          "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    frame_bytes = width * h * 3
    n = len(buf) // frame_bytes
    return np.frombuffer(buf[: n * frame_bytes], np.uint8).reshape(n, h, width, 3)
