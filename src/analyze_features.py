from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/cinemind_features.parquet"
)


def main() -> None:

    df = pd.read_parquet(INPUT_FILE)

    metadata = {
        "source",
        "source_id",
        "title",
        "type",
    }

    features = [
        column
        for column in df.columns
        if column not in metadata
    ]

    print("=" * 75)
    print("CINEMIND FEATURE ANALYSIS")
    print("=" * 75)

    print(f"\nTitles: {len(df)}")
    print(f"Features: {len(features)}")

    rows = []

    for feature in features:

        yes_count = int(df[feature].sum())
        no_count = len(df) - yes_count

        yes_ratio = yes_count / len(df)

        rows.append(
            {
                "feature": feature,
                "yes": yes_count,
                "no": no_count,
                "yes_ratio": yes_ratio,
            }
        )

    result = pd.DataFrame(rows)

    result["balance"] = (
        result["yes_ratio"]
        .apply(lambda x: min(x, 1 - x))
    )

    result = result.sort_values(
        "balance",
        ascending=False,
    )

    print("\nFEATURE DISTRIBUTIONS")
    print("-" * 75)

    print(
        result.to_string(
            index=False,
            formatters={
                "yes_ratio": "{:.3f}".format,
                "balance": "{:.3f}".format,
            },
        )
    )

    print("\n\nBEST BALANCED FEATURES")
    print("-" * 75)

    print(
        result.head(10).to_string(
            index=False,
            formatters={
                "yes_ratio": "{:.3f}".format,
                "balance": "{:.3f}".format,
            },
        )
    )

    print("\n\nVERY RARE FEATURES")
    print("-" * 75)

    rare = result[
        result["yes_ratio"] < 0.10
    ]

    print(
        rare.to_string(
            index=False,
            formatters={
                "yes_ratio": "{:.3f}".format,
                "balance": "{:.3f}".format,
            },
        )
    )


if __name__ == "__main__":
    main()