import os
import requests
from dotenv import load_dotenv


load_dotenv()

CLIENT_ID = os.environ["MAL_CLIENT_ID"]

URL = "https://api.myanimelist.net/v2/anime"

HEADERS = {
    "X-MAL-CLIENT-ID": CLIENT_ID,
}


QUERIES = [
    "the",
    "and",
    "for",
    "you",
    "one",
    "two",
    "love",
    "star",
    "world",
    "life",
    "night",
    "day",
    "girl",
    "boy",
    "school",
    "city",
    "dragon",
    "king",
    "magic",
    "heart",
    "time",
    "death",
    "dream",
    "dark",
    "fire",
    "sky",
    "moon",
    "hero",
    "war",
    "zero",
]


def get_query_count(query):
    offset = 0
    total = 0
    unique_ids = set()

    while True:
        params = {
            "q": query,
            "limit": 10,
            "offset": offset,
            "fields": "id",
        }

        response = requests.get(
            URL,
            headers=HEADERS,
            params=params,
            timeout=30,
        )

        if response.status_code != 200:
            print(
                f"{query:<10} ERROR "
                f"{response.status_code}: {response.text}"
            )
            return 0, set()

        data = response.json()

        results = data.get("data", [])
        paging = data.get("paging", {})

        for item in results:
            unique_ids.add(item["node"]["id"])

        total += len(results)

        if "next" not in paging:
            break

        offset += 10

    return total, unique_ids


def main():

    print("=" * 80)
    print("CINEMIND — MAL SEARCH VOCABULARY AUDIT")
    print("=" * 80)

    global_ids = set()

    print()
    print(
        f"{'QUERY':<12}"
        f"{'RESULTS':>10}"
        f"{'NEW IDS':>12}"
        f"{'UNION':>12}"
    )

    print("-" * 80)

    for query in QUERIES:

        total, ids = get_query_count(query)

        before = len(global_ids)

        global_ids.update(ids)

        new_ids = len(global_ids) - before

        print(
            f"{query:<12}"
            f"{total:>10}"
            f"{new_ids:>12}"
            f"{len(global_ids):>12}"
        )

    print("-" * 80)

    print()
    print(f"TOTAL UNIQUE MAL IDs: {len(global_ids):,}")


if __name__ == "__main__":
    main()