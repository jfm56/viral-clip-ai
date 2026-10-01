"""Audio features: loudness dynamics, hype spikes, speech/brightness balance.

Everything is computed on 50 ms non-overlapping hops of 16 kHz mono audio.
"""
import numpy as np
from scipy.signal import find_peaks

SR = 16000
HOP_S = 0.05
HOOK_S = 3.0


def per_second(values: np.ndarray, times: np.ndarray, n_sec: int, agg=np.mean) -> np.ndarray:
    out = np.full(n_sec, np.nan, dtype=np.float32)
    idx = np.minimum(times.astype(int), n_sec - 1)
    for s in range(n_sec):
        m = idx == s
        if m.any():
            out[s] = agg(values[m])
    return out


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) < n or n < 2:
        return x.copy()
    # Edge-pad: zero-padding would pull negative dB values toward 0 and fake loud edges.
    padded = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(padded, np.ones(n) / n, mode="valid")


def audio_features(y: np.ndarray, n_sec: int) -> tuple[dict, dict]:
    hop = int(SR * HOP_S)
    n = y.size // hop
    if n < int(1.0 / HOP_S):
        return {"has_audio": 0}, {}

    fr = y[: n * hop].reshape(n, hop)
    t = (np.arange(n) + 0.5) * HOP_S
    rms = np.sqrt((fr ** 2).mean(axis=1))
    db = np.maximum(20 * np.log10(rms + 1e-6), -80.0)
    db_s = _smooth(db, int(0.25 / HOP_S))

    spec = np.abs(np.fft.rfft(fr * np.hanning(hop), n=1024, axis=1))
    freqs = np.fft.rfftfreq(1024, 1 / SR)
    power = spec ** 2
    tot = power.sum(axis=1) + 1e-10
    centroid = (power * freqs).sum(axis=1) / tot
    speech = power[:, (freqs >= 300) & (freqs <= 3400)].sum(axis=1) / tot
    high = power[:, freqs >= 4000].sum(axis=1) / tot
    logspec = np.log1p(spec)
    flux = np.concatenate([[0.0], np.maximum(np.diff(logspec, axis=0), 0).sum(axis=1)])

    active = db > -50
    act = active if active.any() else np.ones_like(active)
    dur = t[-1] + HOP_S / 2
    hook = t < min(HOOK_S, 0.2 * dur if dur < 15 else HOOK_S)
    tail = t > dur - min(HOOK_S, 0.2 * dur)

    peaks, _ = find_peaks(flux, height=flux.mean() + 2 * flux.std(), distance=max(1, int(0.2 / HOP_S)))
    jump_n = int(0.5 / HOP_S)
    max_jump = float(np.max(db_s[jump_n:] - db_s[:-jump_n])) if n > jump_n else 0.0
    loud = db_s > np.median(db_s) + 10
    bursts = int(np.sum(loud[1:] & ~loud[:-1]) + loud[0])
    peak_i = int(np.argmax(db_s))

    feats = {
        "has_audio": 1,
        "loud_mean_db": float(db[act].mean()),
        "loud_std_db": float(db[act].std()),
        "loud_p95_db": float(np.percentile(db, 95)),
        "dynamic_range_db": float(np.percentile(db, 95) - np.percentile(db, 10)),
        "silence_ratio": float((~active).mean()),
        "loud_peak_time_s": float(t[peak_i]),
        "loud_peak_time_rel": float(t[peak_i] / dur),
        "hook_loud_delta_db": float(db[hook].mean() - db.mean()) if hook.any() else 0.0,
        "end_loud_delta_db": float(db[tail].mean() - db.mean()) if tail.any() else 0.0,
        "loud_slope": float(np.polyfit(t / dur, db_s, 1)[0]),
        "max_loud_jump_db": max_jump,
        "loud_burst_count": bursts,
        "loud_burst_rate": bursts / dur,
        "onset_rate": len(peaks) / dur,
        "flux_mean": float(flux.mean()),
        "flux_peak_z": float((flux.max() - flux.mean()) / (flux.std() + 1e-6)),
        "centroid_mean_hz": float(centroid[act].mean()),
        "speech_band_ratio": float(speech[act].mean()),
        "high_band_ratio": float(high[act].mean()),
        "hook_onset_rate": float(np.sum(t[peaks] < HOOK_S) / min(HOOK_S, dur)),
    }
    timeline = {
        "loud_db": per_second(db, t, n_sec),
        "flux": per_second(flux, t, n_sec),
        "speech": per_second(speech, t, n_sec),
    }
    return feats, timeline
