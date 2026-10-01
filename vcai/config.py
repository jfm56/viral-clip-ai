"""Paths, credentials and tunables. Everything can be overridden via .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("VCAI_DATA_DIR", ROOT / "data"))
DB_PATH = DATA_DIR / "vcai.sqlite"
MEDIA_DIR = DATA_DIR / "media"
TIMELINE_DIR = DATA_DIR / "timelines"
MODEL_DIR = DATA_DIR / "models"
REPORT_DIR = DATA_DIR / "reports"

TWITCH_CLIENT_ID = os.getenv("TWITCH_CLIENT_ID", "")
TWITCH_CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET", "")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

# Minimum age before a clip's view count is treated as "settled" enough to label.
MIN_AGE_DAYS = {"twitch": 3, "youtube": 7}
# A creator needs this many labelled clips for us to know what "normal" looks like for them.
MIN_CREATOR_CLIPS = 5
# Top X% of creator-relative scores (per source) are flagged as "breakout" clips.
BREAKOUT_QUANTILE = 0.90

# Twitch categories that are not video games.
NON_GAMING = {
    "Just Chatting", "IRL", "Music", "Art", "Talk Shows & Podcasts", "Sports", "ASMR",
    "Pools, Hot Tubs, and Beaches", "Special Events", "Travel & Outdoors", "Food & Drink",
    "Science & Technology", "Software and Game Development", "Makers & Crafting", "Slots",
    "Virtual Casino", "Crypto", "Politics", "Animals, Aquariums, and Zoos",
    "Beauty & Body Art", "Fitness & Health", "Always On", "Co-working & Studying",
    "Dancing", "Writing & Reading", "Stocks and Bonds", "Watch Parties", "Tabletop RPGs",
}

# Fallback YouTube search list when no Twitch top-games list has been collected yet.
DEFAULT_GAMES = [
    "Fortnite", "Valorant", "Counter-Strike 2", "Apex Legends", "Minecraft", "Call of Duty Warzone",
    "League of Legends", "Grand Theft Auto V", "Rocket League", "Overwatch 2", "Marvel Rivals",
    "Rainbow Six Siege", "Dead by Daylight", "Elden Ring", "Roblox", "Rust", "EA Sports FC",
    "Escape from Tarkov", "PUBG", "Dota 2", "Lethal Company", "Fall Guys",
]
