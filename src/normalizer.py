import json
from pathlib import Path
from typing import Any

import pandas as pd


RAW_DIR = Path("data/raw/tmdb")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "tmdb_titles.parquet"


def extract_names(items: list[dict[str, Any]] | None) -> list[str]:
    """Extract the 'name' field from a TMDB list."""
    if not items:
        return []

    return [
        item["name"]
        for item in items
        if isinstance(item, dict) and item.get("name")
    ]


def extract_cast(credits: dict[str, Any], limit: int = 10) -> list[str]:
    """Return the top N cast members."""
    cast = credits.get("cast", [])

    return [
        person["name"]
        for person in cast[:limit]
        if person.get("name")
    ]


def extract_crew(credits: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Extract directors and writers from crew."""
    crew = credits.get("crew", [])

    directors = []
    writers = []

    for person in crew:
        name = person.get("name")
        job = person.get("job")

        if not name:
            continue

        if job == "Director":
            directors.append(name)

        elif job in {"Writer", "Screenplay", "Story"}:
            writers.append(name)

    return list(dict.fromkeys(directors)), list(dict.fromkeys(writers))


def extract_keywords(data: dict[str, Any], media_type: str) -> list[str]:
    """
    Extract keywords.

    Movies:
        {"keywords": [...]}

    TV:
        {"results": [...]}
    """

    if media_type == "movie":
        keywords = data.get("keywords", [])
    else:
        keywords = data.get("results", [])

    return extract_names(keywords)


def normalize_movie(data: dict[str, Any]) -> dict[str, Any]:

    details = data["details"]
    credits = data["credits"]
    keyword_data = data["keywords"]

    directors, writers = extract_crew(credits)

    return {
        "source": "tmdb",
        "source_id": details.get("id"),

        "type": "movie",

        "title": details.get("title"),
        "original_title": details.get("original_title"),
        "overview": details.get("overview"),

        "release_date": details.get("release_date"),

        "original_language": details.get("original_language"),
        "origin_country": details.get("origin_country", []),
        "spoken_languages": extract_names(
            details.get("spoken_languages")
        ),
        "production_countries": extract_names(
            details.get("production_countries")
        ),

        "genres": extract_names(details.get("genres")),
        "keywords": extract_keywords(keyword_data, "movie"),

        "runtime": details.get("runtime"),

        "number_of_seasons": None,
        "number_of_episodes": None,

        "production_companies": extract_names(
            details.get("production_companies")
        ),
        "networks": [],

        "cast": extract_cast(credits),
        "directors": directors,
        "writers": writers,

        "status": details.get("status"),

        "vote_average": details.get("vote_average"),
        "vote_count": details.get("vote_count"),
        "popularity": details.get("popularity"),

        # Useful later for the frontend
        "poster_path": details.get("poster_path"),
        "backdrop_path": details.get("backdrop_path"),
    }


def normalize_tv(data: dict[str, Any]) -> dict[str, Any]:

    details = data["details"]
    credits = data["credits"]
    keyword_data = data["keywords"]

    directors, writers = extract_crew(credits)

    return {
        "source": "tmdb",
        "source_id": details.get("id"),

        "type": "tv",

        "title": details.get("name"),
        "original_title": details.get("original_name"),
        "overview": details.get("overview"),

        "release_date": details.get("first_air_date"),

        "original_language": details.get("original_language"),
        "origin_country": details.get("origin_country", []),
        "spoken_languages": extract_names(
            details.get("spoken_languages")
        ),
        "production_countries": extract_names(
            details.get("production_countries")
        ),

        "genres": extract_names(details.get("genres")),
        "keywords": extract_keywords(keyword_data, "tv"),

        "runtime": (
            details.get("episode_run_time", [None])[0]
            if details.get("episode_run_time")
            else None
        ),

        "number_of_seasons": details.get("number_of_seasons"),
        "number_of_episodes": details.get("number_of_episodes"),

        "production_companies": extract_names(
            details.get("production_companies")
        ),
        "networks": extract_names(
            details.get("networks")
        ),

        "cast": extract_cast(credits),
        "directors": directors,
        "writers": writers,

        "status": details.get("status"),

        "vote_average": details.get("vote_average"),
        "vote_count": details.get("vote_count"),
        "popularity": details.get("popularity"),

        "poster_path": details.get("poster_path"),
        "backdrop_path": details.get("backdrop_path"),
    }


def load_json(path: Path) -> dict[str, Any]:

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def normalize_all() -> pd.DataFrame:

    records = []

    movie_files = sorted(
        (RAW_DIR / "movies").glob("*.json")
    )

    tv_files = sorted(
        (RAW_DIR / "tv").glob("*.json")
    )

    print(f"Movie files found: {len(movie_files)}")
    print(f"TV files found: {len(tv_files)}")

    # -------------------------
    # Movies
    # -------------------------

    for path in movie_files:

        try:
            data = load_json(path)
            record = normalize_movie(data)
            records.append(record)

        except Exception as exc:
            print(f"Failed movie: {path.name}")
            print(f"Reason: {exc}")

    # -------------------------
    # TV
    # -------------------------

    for path in tv_files:

        try:
            data = load_json(path)
            record = normalize_tv(data)
            records.append(record)

        except Exception as exc:
            print(f"Failed TV: {path.name}")
            print(f"Reason: {exc}")

    df = pd.DataFrame(records)

    return df


def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = normalize_all()

    # Remove duplicate source IDs if any exist.
    df = df.drop_duplicates(
        subset=["source", "source_id"]
    )

    # Convert release date into datetime.
    df["release_date"] = pd.to_datetime(
        df["release_date"],
        errors="coerce",
    )

    # Derive release year.
    df["release_year"] = (
        df["release_date"]
        .dt.year
        .astype("Int64")
    )

    # Save.
    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 60)
    print("NORMALIZATION COMPLETE")
    print("=" * 60)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Output: {OUTPUT_FILE}")

    print("\nType distribution:")
    print(df["type"].value_counts())

    print("\nMissing values:")
    print(
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )


if __name__ == "__main__":
    main()