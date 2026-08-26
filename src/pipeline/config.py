"""
CineMind Data Pipeline — Centralized Configuration.

All paths, API settings, rate limits, and discovery parameters
are defined here. API credentials come from environment variables.
"""

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Project paths
# ============================================================

# src/pipeline/config.py → parent.parent = src/
SRC_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SRC_DIR / "data"

RAW_TMDB_DIR = DATA_DIR / "raw" / "tmdb"
RAW_MAL_DIR = DATA_DIR / "raw" / "mal"
STAGING_DIR = DATA_DIR / "staging"
CANONICAL_DIR = DATA_DIR / "canonical"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
REPORTS_DIR = DATA_DIR / "reports"


# ============================================================
# API credentials (from environment — never hard-coded)
# ============================================================

TMDB_API_KEY: str | None = os.getenv("TMDB_API_KEY")
MAL_CLIENT_ID: str | None = os.getenv("MAL_CLIENT_ID")


# ============================================================
# API base URLs
# ============================================================

TMDB_BASE_URL = "https://api.tmdb.org/3"
MAL_BASE_URL = "https://api.myanimelist.net/v2"


# ============================================================
# Rate limiting (seconds between requests)
# ============================================================

TMDB_REQUEST_DELAY = 0.26   # ~3.8 req/s  (TMDB allows ~40 per 10s)
MAL_REQUEST_DELAY = 0.75    # ~1.3 req/s  (MAL is sensitive)


# ============================================================
# Retry / timeout
# ============================================================

MAX_RETRIES = 6
REQUEST_TIMEOUT = 30


# ============================================================
# TMDB discovery settings (Targeted ~100k total universe)
# ============================================================

TMDB_RESULTS_PER_PAGE = 20      # Results per discover page

# Page caps per year to acquire ~45k movies & ~25k TV series across 126 years
TMDB_MAX_PAGES_PER_YEAR_MOVIE = 18   # ~360 movies/year = ~45,000 movies
TMDB_MAX_PAGES_PER_YEAR_TV = 10      # ~200 TV shows/year = ~25,000 TV series
TMDB_MAX_PAGES = 18                  # Default cap for discover loop

# Year range for discovery
DISCOVERY_START_YEAR = 1900
DISCOVERY_END_YEAR = datetime.now().year

# If total_pages >= this threshold, also do language segmentation
HIGH_VOLUME_PAGE_THRESHOLD = 15

# Languages used for TMDB language-segmentation strategy
TMDB_SEGMENT_LANGUAGES = [
    "ja", "ko", "hi", "zh", "fr", "de", "es", "it", "pt",
    "ru", "th", "tr", "ar", "pl", "sv", "da", "nl", "fi",
    "no", "ta", "te", "ml", "bn", "id", "tl",
]


# ============================================================
# MAL discovery settings (Targeted ~30k anime)
# ============================================================

MAL_RANKING_TYPES = [
    "all", "airing", "upcoming", "tv", "movie",
    "ova", "special", "bypopularity", "favorite",
]

MAL_RESULTS_PER_PAGE = 100            # Max per MAL request
MAL_MAX_RANKING_OFFSET = 3000          # ~3,000 anime per ranking category (30 pages)
MAL_MAX_SEARCH_OFFSET = 300            # ~300 anime per search query (3 pages)

# Fields to request from the official MAL API v2
MAL_FIELDS = ",".join([
    "id", "title", "alternative_titles", "start_date", "end_date",
    "synopsis", "mean", "rank", "popularity", "num_list_users",
    "num_scoring_users", "num_favorites", "media_type", "status",
    "genres", "num_episodes", "start_season", "source",
    "average_episode_duration", "studios", "rating",
])

# Systematic search vocabulary (all terms >= 3 characters)
MAL_SEARCH_VOCABULARY = [
    # --- Japanese romanized syllables / words ---
    "aku", "shi", "kou", "sen", "ten", "kai", "ren", "sou",
    "sei", "kin", "gin", "tai", "chi", "rin", "mon", "hana",
    "kaze", "yume", "hoshi", "tsuki", "mizu", "kuro", "shiro",
    "midori", "neko", "kami", "yoru", "sora", "umi", "koi",
    "uta", "haru", "natsu", "fuyu", "aki", "tora", "ryu",
    "inu", "usagi", "sakura", "yuki", "tsumi", "hikari",
    "kage", "tetsu", "densetsu", "bouken", "shikai", "ore",
    "boku", "kimi", "ano", "sono", "kono", "dake", "mama",
    "suki", "daisuki", "kanojo", "kareshi", "tomodachi",
    # --- Common anime title words ---
    "dragon", "sword", "knight", "princess", "demon", "angel",
    "ghost", "spirit", "hunter", "fighter", "gundam", "mecha",
    "magical", "school", "detective", "samurai", "ninja",
    "pirate", "robot", "monster", "slayer", "witch", "zero",
    "hero", "brave", "strike", "blade", "cross", "genesis",
    "chaos", "code", "gate", "chronicle", "frontier", "order",
    "force", "break", "burst", "drive", "rise", "final",
    "tales", "saga", "index", "mobile", "super", "ultra",
    # --- Genre / theme terms ---
    "romance", "horror", "comedy", "mystery", "fantasy",
    "adventure", "action", "sports", "music", "cooking",
    "idol", "isekai", "slice", "psychological", "military",
    "vampire", "zombie", "apocalypse", "survival", "dungeon",
    "reincarnation", "harem", "revenge", "otome", "villainess",
    # --- Common English words in anime titles ---
    "the", "love", "star", "world", "night", "dream", "king",
    "queen", "blood", "black", "white", "red", "blue", "golden",
    "silver", "dark", "light", "shadow", "legend", "story",
    "last", "first", "new", "great", "little", "wild",
    "secret", "lost", "dead", "steel", "crystal", "moon",
    "sun", "war", "peace", "hope", "fate", "soul", "death",
    "life", "fire", "ice", "wind", "earth", "thunder",
    "girl", "boy", "man", "child", "prince", "saint",
    # --- Romanized Japanese concepts ---
    "hime", "ouji", "mahou", "sekai", "gakuen", "senshi",
    "shoujo", "shounen", "josei", "seinen", "otaku", "chuuni",
    "senpai", "baka", "kawaii", "sugoi", "tensei", "yuusha",
    "maou", "kenja", "toaru", "konosuba", "danmachi",
]


# ============================================================
# Checkpoint settings
# ============================================================

CHECKPOINT_SAVE_INTERVAL = 5    # Save state every N pages/tasks
