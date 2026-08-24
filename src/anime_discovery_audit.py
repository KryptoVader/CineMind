from collections import Counter

from jikanpy import Jikan
from jikanpy.exceptions import APIException


jikan = Jikan()


def inspect_response(name, response):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    data = response.get("data", [])
    pagination = response.get("pagination", {})

    print(f"Records returned : {len(data)}")

    if pagination:
        print(f"Last page        : {pagination.get('last_visible_page')}")
        print(f"Has next page    : {pagination.get('has_next_page')}")
        print(f"Total pages      : {pagination.get('last_visible_page')}")

    if data:
        print()
        print("First 5 results:")

        for anime in data[:5]:
            print(
                f"  {anime.get('mal_id')} | "
                f"{anime.get('title')} | "
                f"{anime.get('type')} | "
                f"{anime.get('score')}"
            )


def test_genre():
    """
    Test Jikan genre discovery.

    Genre ID 1 = Action
    """

    print("\nTesting GENRE discovery...")

    try:
        response = jikan.genres(
            "anime",
            1,
            page=1,
        )

        inspect_response(
            "GENRE: ACTION",
            response,
        )

    except APIException as e:
        print(f"GENRE FAILED: {e}")

    except Exception as e:
        print(f"GENRE FAILED: {type(e).__name__}: {e}")


def test_top():
    """
    Test top anime discovery.
    """

    print("\nTesting TOP anime discovery...")

    try:
        response = jikan.top(
            "anime",
            page=1,
        )

        inspect_response(
            "TOP ANIME",
            response,
        )

    except APIException as e:
        print(f"TOP FAILED: {e}")

    except Exception as e:
        print(f"TOP FAILED: {type(e).__name__}: {e}")


def test_season():
    """
    Test seasonal anime discovery.
    """

    print("\nTesting SEASON discovery...")

    try:
        response = jikan.seasons(
            year=2025,
            season="winter",
            page=1,
        )

        inspect_response(
            "WINTER 2025",
            response,
        )

    except APIException as e:
        print(f"SEASON FAILED: {e}")

    except Exception as e:
        print(
            f"SEASON FAILED: "
            f"{type(e).__name__}: {e}"
        )


def test_multiple_genres():
    """
    Test several genre endpoints.

    This is important because one genre endpoint
    failing shouldn't necessarily mean all discovery
    endpoints are broken.
    """

    genres = {
        1: "Action",
        2: "Adventure",
        4: "Comedy",
        8: "Drama",
        10: "Fantasy",
        14: "Horror",
        7: "Mystery",
        22: "Romance",
        24: "Sci-Fi",
        30: "Sports",
    }

    print()
    print("=" * 70)
    print("MULTI-GENRE DISCOVERY TEST")
    print("=" * 70)

    results = Counter()

    for genre_id, genre_name in genres.items():

        print(
            f"\nTesting {genre_name} "
            f"(ID={genre_id})..."
        )

        try:

            response = jikan.genres(
                "anime",
                genre_id,
                page=1,
            )

            data = response.get("data", [])

            results[genre_name] = len(data)

            print(
                f"SUCCESS: "
                f"{len(data)} records"
            )

        except APIException as e:

            print(
                f"FAILED: {e}"
            )

        except Exception as e:

            print(
                f"FAILED: "
                f"{type(e).__name__}: {e}"
            )

    print()
    print("-" * 70)
    print("GENRE SUMMARY")
    print("-" * 70)

    for genre, count in results.items():

        print(
            f"{genre:<15} "
            f"{count:>5}"
        )


def main():

    print("=" * 70)
    print("CINEMIND JIKAN DISCOVERY TEST")
    print("=" * 70)

    # Keep these tests separate.
    # We want to know exactly which endpoint works.

    test_genre()

    test_top()

    test_season()

    # Multiple genre requests are useful for checking
    # whether genre discovery is consistently available.
    test_multiple_genres()

    print()
    print("=" * 70)
    print("DISCOVERY TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()