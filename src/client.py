import os
from typing import Any

import requests
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()


class TMDBClient:
    BASE_URL = "https://api.themoviedb.org/3"

    def __init__(self) -> None:
        token = os.getenv("TMDB_API_KEY")

        if not token:
            raise RuntimeError(
                "TMDB_API_KEY not found. Check your .env file."
            )

        self.session = requests.Session()
        self.session.headers.update(
            {
                "accept": "application/json",
            }
        )

        self.api_key = token

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        params = params or {}
        params["api_key"] = self.api_key

        response = self.session.get(
            f"{self.BASE_URL}{endpoint}",
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        return response.json()

    def discover_movies(self, page: int) -> dict[str, Any]:
        return self.get(
            "/discover/movie",
            {
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": page,
            },
        )

    def discover_tv(self, page: int) -> dict[str, Any]:
        return self.get(
            "/discover/tv",
            {
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": page,
            },
        )

    def movie_details(self, movie_id: int) -> dict[str, Any]:
        return self.get(f"/movie/{movie_id}")

    def movie_credits(self, movie_id: int) -> dict[str, Any]:
        return self.get(f"/movie/{movie_id}/credits")

    def movie_keywords(self, movie_id: int) -> dict[str, Any]:
        return self.get(f"/movie/{movie_id}/keywords")

    def tv_details(self, tv_id: int) -> dict[str, Any]:
        return self.get(f"/tv/{tv_id}")

    def tv_credits(self, tv_id: int) -> dict[str, Any]:
        return self.get(f"/tv/{tv_id}/credits")

    def tv_keywords(self, tv_id: int) -> dict[str, Any]:
        return self.get(f"/tv/{tv_id}/keywords")