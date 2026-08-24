"""
Official MyAnimeList API v2 client.

Uses X-MAL-CLIENT-ID authentication (no OAuth required for
public data). Does NOT use Jikan.
"""

import logging
from typing import Any

from pipeline.config import (
    MAL_BASE_URL,
    MAL_CLIENT_ID,
    MAL_FIELDS,
    MAL_REQUEST_DELAY,
)
from pipeline.http_client import ResilientClient

logger = logging.getLogger(__name__)


class MALClient:
    """Client for the official MAL API v2."""

    def __init__(self) -> None:
        if not MAL_CLIENT_ID:
            raise RuntimeError(
                "MAL_CLIENT_ID not found in environment. "
                "Check your .env file."
            )

        self.client = ResilientClient(
            base_url=MAL_BASE_URL,
            delay=MAL_REQUEST_DELAY,
            default_headers={
                "X-MAL-CLIENT-ID": MAL_CLIENT_ID,
            },
        )

    def get_ranking(
        self,
        ranking_type: str = "all",
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        GET /anime/ranking

        Returns ranked anime with requested fields.
        """
        return self.client.get(
            "/anime/ranking",
            params={
                "ranking_type": ranking_type,
                "limit": limit,
                "offset": offset,
                "fields": MAL_FIELDS,
            },
        )

    def search_anime(
        self,
        q: str,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        GET /anime

        Search anime by query string (>= 3 characters).
        """
        return self.client.get(
            "/anime",
            params={
                "q": q,
                "limit": limit,
                "offset": offset,
                "fields": MAL_FIELDS,
            },
        )

    @property
    def request_count(self) -> int:
        return self.client.request_count

    def close(self) -> None:
        self.client.close()
