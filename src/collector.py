import json
import time
from pathlib import Path

from client import TMDBClient


class TMDBCollector:

    def __init__(self, output_dir: str = "data/raw/tmdb") -> None:
        self.client = TMDBClient()
        self.output_dir = Path(output_dir)

        self.movies_dir = self.output_dir / "movies"
        self.tv_dir = self.output_dir / "tv"

        self.movies_dir.mkdir(parents=True, exist_ok=True)
        self.tv_dir.mkdir(parents=True, exist_ok=True)

    def save_movie(self, movie_id: int) -> None:
        output_file = self.movies_dir / f"{movie_id}.json"

        if output_file.exists():
            return

        data = {
            "details": self.client.movie_details(movie_id),
            "credits": self.client.movie_credits(movie_id),
            "keywords": self.client.movie_keywords(movie_id),
        }

        output_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        time.sleep(0.05)

    def save_tv(self, tv_id: int) -> None:
        output_file = self.tv_dir / f"{tv_id}.json"

        if output_file.exists():
            return

        data = {
            "details": self.client.tv_details(tv_id),
            "credits": self.client.tv_credits(tv_id),
            "keywords": self.client.tv_keywords(tv_id),
        }

        output_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        time.sleep(0.05)

    def collect_movies(self, pages: int = 1) -> None:
        for page in range(1, pages + 1):

            print(f"Collecting movie page {page}/{pages}")

            data = self.client.discover_movies(page)

            for movie in data["results"]:
                self.save_movie(movie["id"])

    def collect_tv(self, pages: int = 1) -> None:
        for page in range(1, pages + 1):

            print(f"Collecting TV page {page}/{pages}")

            data = self.client.discover_tv(page)

            for show in data["results"]:
                self.save_tv(show["id"])