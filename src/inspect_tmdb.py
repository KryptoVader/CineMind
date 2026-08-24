import json
from pathlib import Path


def inspect_file(path: Path) -> None:
    print("\n" + "=" * 70)
    print(path)
    print("=" * 70)

    data = json.loads(path.read_text(encoding="utf-8"))

    for section, content in data.items():
        print(f"\n[{section}]")

        if isinstance(content, dict):
            print("Keys:")
            for key in content.keys():
                print(f"  - {key}")

        elif isinstance(content, list):
            print(f"List with {len(content)} items")


movies = list(Path("data/raw/tmdb/movies").glob("*.json"))
tv = list(Path("data/raw/tmdb/tv").glob("*.json"))

print(f"Movies collected: {len(movies)}")
print(f"TV collected: {len(tv)}")

if movies:
    inspect_file(movies[0])

if tv:
    inspect_file(tv[0])