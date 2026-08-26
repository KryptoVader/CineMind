"""
TMDB API v3 client.

Thin wrapper over ResilientClient. Provides typed methods for
the discover, genre-list, and configuration endpoints.
"""

import json
import logging
from typing import Any

from pipeline.config import (
    RAW_TMDB_DIR,
    TMDB_API_KEY,
    TMDB_BASE_URL,
    TMDB_REQUEST_DELAY,
)
from pipeline.http_client import ResilientClient

logger = logging.getLogger(__name__)


class TMDBClient:
    """Client for the TMDB API v3."""

    def __init__(self) -> None:
        if not TMDB_API_KEY:
            raise RuntimeError(
                "TMDB_API_KEY not found in environment. "
                "Check your .env file."
            )

        self.client = ResilientClient(
            base_url=TMDB_BASE_URL,
            delay=TMDB_REQUEST_DELAY,
            default_params={"api_key": TMDB_API_KEY},
        )

    # ----------------------------------------------------------
    # Discovery endpoints
    # ----------------------------------------------------------

    def discover_movies(
        self,
        page: int = 1,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call /discover/movie with given filters."""
        params: dict[str, Any] = {
            "page": page,
            "sort_by": "popularity.desc",
        }
        params.update(kwargs)
        return self.client.get("/discover/movie", params=params)

    def discover_tv(
        self,
        page: int = 1,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call /discover/tv with given filters."""
        params: dict[str, Any] = {
            "page": page,
            "sort_by": "popularity.desc",
        }
        params.update(kwargs)
        return self.client.get("/discover/tv", params=params)

    # ----------------------------------------------------------
    # Genre list
    # ----------------------------------------------------------

    def get_movie_genres(self) -> list[dict[str, Any]]:
        data = self.client.get(
            "/genre/movie/list", {"language": "en-US"}
        )
        return data.get("genres", [])

    def get_tv_genres(self) -> list[dict[str, Any]]:
        data = self.client.get(
            "/genre/tv/list", {"language": "en-US"}
        )
        return data.get("genres", [])

    def get_all_genres(self) -> dict[int, str]:
        """Fetch and merge movie + TV genre lists into {id: name}."""
        movie_genres = self.get_movie_genres()
        tv_genres = self.get_tv_genres()
        genre_map: dict[int, str] = {}
        for g in movie_genres + tv_genres:
            genre_map[g["id"]] = g["name"]
        return genre_map

    def load_genre_map(self) -> dict[int, str]:
        """Load genre map from cache, or fetch and cache it."""
        cache_path = RAW_TMDB_DIR / "genre_map.json"
        RAW_TMDB_DIR.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            data = json.loads(
                cache_path.read_text(encoding="utf-8")
            )
            return {int(k): v for k, v in data.items()}

        logger.info("Fetching TMDB genre list...")
        genre_map = self.get_all_genres()
        cache_path.write_text(
            json.dumps(
                {str(k): v for k, v in genre_map.items()},
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Cached %d genres.", len(genre_map))
        return genre_map

    # ----------------------------------------------------------
    # Keywords endpoints
    # ----------------------------------------------------------

    def get_movie_keywords(self, movie_id: int) -> list[str]:
        """Fetch keywords for a movie as a list of strings."""
        data = self.client.get(f"/movie/{movie_id}/keywords")
        kws = data.get("keywords", [])
        return [k["name"] for k in kws if isinstance(k, dict) and k.get("name")]

    def get_tv_keywords(self, tv_id: int) -> list[str]:
        """Fetch keywords for a TV series as a list of strings."""
        data = self.client.get(f"/tv/{tv_id}/keywords")
        kws = data.get("results", []) or data.get("keywords", [])
        return [k["name"] for k in kws if isinstance(k, dict) and k.get("name")]

    # ----------------------------------------------------------
    # Convenience
    # ----------------------------------------------------------

    @property
    def request_count(self) -> int:
        return self.client.request_count

    def close(self) -> None:
        self.client.close()
