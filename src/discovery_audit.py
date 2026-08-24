import os
import requests
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")

BASE_URL = "https://api.themoviedb.org/3"

START_YEAR = 1900
END_YEAR = 2029


# ============================================================
# TMDB REQUEST
# ============================================================

def tmdb_get(endpoint, params=None):
    """Make a request to TMDB and return the JSON response."""

    if params is None:
        params = {}

    params["api_key"] = API_KEY

    url = f"{BASE_URL}/{endpoint}"

    response = requests.get(url, params=params, timeout=30)

    response.raise_for_status()

    return response.json()


# ============================================================
# DISCOVERY COUNTS
# ============================================================

def get_movie_count(start_year, end_year):
    """Return number of movies released within a year range."""

    data = tmdb_get(
        "discover/movie",
        {
            "language": "en-US",
            "page": 1,

            "primary_release_date.gte": f"{start_year}-01-01",
            "primary_release_date.lte": f"{end_year}-12-31",

            # Don't restrict to popular movies.
            # We want the entire discoverable universe.
            "sort_by": "popularity.desc",
        },
    )

    return data["total_results"]


def get_tv_count(start_year, end_year):
    """Return number of TV titles first aired within a year range."""

    data = tmdb_get(
        "discover/tv",
        {
            "language": "en-US",
            "page": 1,

            "first_air_date.gte": f"{start_year}-01-01",
            "first_air_date.lte": f"{end_year}-12-31",

            "sort_by": "popularity.desc",
        },
    )

    return data["total_results"]


# ============================================================
# YEAR BINS
# ============================================================

def generate_decades(start_year, end_year):
    """
    Generate 10-year bins.

    Example:
        1980-1989
        1990-1999
        2000-2009
    """

    decades = []

    current = start_year

    while current <= end_year:

        decade_start = current
        decade_end = min(current + 9, end_year)

        decades.append((decade_start, decade_end))

        current += 10

    return decades


# ============================================================
# MAIN
# ============================================================

def main():

    if not API_KEY:
        raise RuntimeError(
            "TMDB_API_KEY was not found.\n"
            "Make sure it exists in your .env file."
        )

    decades = generate_decades(
        START_YEAR,
        END_YEAR,
    )

    print("=" * 70)
    print("CINEMIND DISCOVERY AUDIT")
    print("=" * 70)

    print()
    print(
        f"{'YEAR':<15}"
        f"{'MOVIES':>15}"
        f"{'TV':>15}"
        f"{'TOTAL':>15}"
    )

    print("-" * 70)

    total_movies = 0
    total_tv = 0

    for start_year, end_year in decades:

        print(
            f"{start_year}-{end_year:<10}",
            end="",
            flush=True,
        )

        try:
            movie_count = get_movie_count(
                start_year,
                end_year,
            )

            tv_count = get_tv_count(
                start_year,
                end_year,
            )

        except requests.HTTPError as e:

            print(f"ERROR: {e}")

            continue

        total = movie_count + tv_count

        total_movies += movie_count
        total_tv += tv_count

        print(
            f"{movie_count:>15,}"
            f"{tv_count:>15,}"
            f"{total:>15,}"
        )

    print("-" * 70)

    print(
        f"{'TOTAL':<15}"
        f"{total_movies:>15,}"
        f"{total_tv:>15,}"
        f"{total_movies + total_tv:>15,}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()