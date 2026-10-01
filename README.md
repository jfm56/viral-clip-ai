# viral-clip-ai

Phase 1 of a social tool that auto-clips gamers' uploads: **learn what makes video-game clips go viral**
from real Twitch Clips + YouTube Shorts performance data.

## Pipeline

```
collect-twitch / collect-youtube  ->  SQLite (clip metadata + view snapshots)
process                           ->  download (yt-dlp, <=480p) -> audio/visual features + per-second timeline -> delete video
train                             ->  LightGBM, cross-validated by creator -> SHAP drivers -> data/reports/latest.md
```

### What "viral" means here
Raw views mostly measure how famous the creator is. So the target is **creator-relative**:

`viral_score = log(views) - leave-one-out median log(views) of that creator's other clips`, then de-trended for clip age.
`views_multiple = exp(score)` reads as "3.2x this creator's normal". The top 10% per platform are labelled **breakout**.
Each creator needs at least 5 clips, which is why the collectors deep-pull each sampled creator's full clip history
(so the data includes their flops, not just the winners).

Cross-validation is **grouped by creator**, so the scores reflect creators the model has never seen.

### Two models, on purpose
- **content**: audio, visual, pacing, duration, format and game only. These are the things our clipper controls when it cuts a user's video.
- **full**: content plus title, posting time, creator size and age.

The gap between them shows how much virality comes from the clip itself versus packaging and distribution.

### Features (per clip, plus a per-second timeline saved to `data/timelines/*.npz`)
- **Audio**: loudness level/range, biggest sudden volume jump (reactions, explosions), loud-burst rate, onset rate,
  first-3s audio hook, loudness build-up, voice-band vs high-band energy.
- **Visual** (4 fps): motion level/burstiness/peak position, first-3s motion hook, scene-cut rate and time to first cut,
  flashes, color, clutter, static HUD/letterbox share, vertical vs horizontal.
- **Cross**: whether audio and motion peaks line up, and where the climax sits (0 = start, 1 = end).

## Setup
```bash
python -m venv .venv --system-site-packages
.venv/Scripts/python -m pip install -r requirements.txt
cp .env.example .env   # then fill in keys
```
- Twitch: https://dev.twitch.tv/console/apps -> register an app (Confidential) -> Client ID + Secret
- YouTube: Google Cloud console -> enable **YouTube Data API v3** -> create an API key (10k quota units/day)

## Run
```bash
.venv/Scripts/python -m vcai collect-twitch --games 40 --days 30
.venv/Scripts/python -m vcai collect-youtube --budget 9000      # once per day (quota)
.venv/Scripts/python -m vcai status
.venv/Scripts/python -m vcai process --limit 2000 --workers 8
.venv/Scripts/python -m vcai train
.venv/Scripts/python -m vcai refresh-stats                      # re-run over days to build view-velocity history
```
Tests: `.venv/Scripts/python -m pytest -q`. They use synthetic videos with known bursts and cuts, plus a planted-signal training test.

## Notes
- Only derived features are kept by default (`--keep-media` keeps the videos). Downloading platform videos
  is limited by each platform's terms. Fine for internal research, but get legal review before this goes commercial.
- The Twitch `/clips` endpoint returns clips sorted by views, and that bias is why we deep-pull per creator.
  YouTube search runs in both `date` order (unbiased) and `viewCount` order.

## Roadmap
1. **Now**: clip-level virality drivers (this repo).
2. **GPU embeddings**: CLIP/SigLIP frame embeddings plus audio event tags (laughter, screaming, gunfire) on the RTX 5090, to add to the model.
3. **Moment model**: use Twitch `vod_offset` to compare the clipped moment against the surrounding VOD, then
   train a per-second "clip-worthiness" scorer on the saved timelines.
4. **Clipper**: slide the moment model over a user's upload, pick the best windows, trim using the learned hook
   and duration patterns, reframe to 9:16, and score each candidate with the virality model.
5. **Web app**: upload, candidate clips with scores, export.
