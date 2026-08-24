from collector import TMDBCollector

def main() -> None:

    collector = TMDBCollector()

    collector.collect_movies(pages=2)
    collector.collect_tv(pages=2)


if __name__ == "__main__":
    main()