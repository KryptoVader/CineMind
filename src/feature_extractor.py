from pathlib import Path
import numpy as np
import pandas as pd
import yaml


DATA_DIR = Path("data/processed")
CONFIG_FILE = Path("../config/features.yaml")

INPUT_FILE = DATA_DIR / "tmdb_titles.parquet"
OUTPUT_FILE = DATA_DIR / "cinemind_features.parquet"


def load_features() -> list[dict]:
    """Load feature definitions from YAML."""

    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config["features"]


def contains_value(values, target: str) -> int:
    """Return 1 if target exists in a list-like value."""

    if values is None:
        return 0

    if isinstance(values, (list, tuple, np.ndarray)):
        return int(target in values)

    return int(values == target)

def contains_any(values, targets: list[str]) -> int:
    """Return 1 if any target exists in a list-like value."""

    if values is None:
        return 0

    if isinstance(values, (list, tuple, np.ndarray)):
        return int(
            any(target in values for target in targets)
        )

    return 0


def evaluate_feature(
    row: pd.Series,
    feature: dict,
) -> int:

    source = feature["source"]
    value = row[source]

    # ---------------------------------
    # Type-based features
    # ---------------------------------

    if source == "type":
        return int(value == feature["value"])

    # ---------------------------------
    # Anime
    # ---------------------------------

    if source == "anime_source":
        # Anime metadata will be added later.
        return 0

    # ---------------------------------
    # List-based features
    # ---------------------------------

    if "value" in feature:

        target = feature["value"]

        if isinstance(value, (list, tuple)):
            return int(target in value)

        # Handle numpy arrays without
        # importing numpy explicitly.
        if hasattr(value, "tolist"):

            converted = value.tolist()

            if isinstance(converted, list):
                return int(target in converted)

            return int(converted == target)

        return int(value == target)

    # ---------------------------------
    # Multiple possible values
    # ---------------------------------

    if "values" in feature:

        targets = feature["values"]

        if isinstance(value, (list, tuple)):
            return int(
                any(target in value for target in targets)
            )

        if hasattr(value, "tolist"):

            converted = value.tolist()

            if isinstance(converted, list):
                return int(
                    any(
                        target in converted
                        for target in targets
                    )
                )

        return 0

    # ---------------------------------
    # Numeric conditions
    # ---------------------------------

    if "condition" in feature:

        if pd.isna(value):
            return 0

        condition = feature["condition"]

        if condition.startswith(">="):
            threshold = float(condition[2:])
            return int(value >= threshold)

        if condition.startswith("<="):
            threshold = float(condition[2:])
            return int(value <= threshold)

        if condition.startswith(">"):
            threshold = float(condition[1:])
            return int(value > threshold)

        if condition.startswith("<"):
            threshold = float(condition[1:])
            return int(value < threshold)

    return 0


def build_feature_matrix(
    df: pd.DataFrame,
    features: list[dict],
) -> pd.DataFrame:

    result = pd.DataFrame(index=df.index)

    # Keep identifiers for candidate tracking.
    result["source"] = df["source"]
    result["source_id"] = df["source_id"]
    result["title"] = df["title"]
    result["type"] = df["type"]

    for feature in features:

        feature_name = feature["name"]

        print(f"Extracting: {feature_name}")

        try:
            result[feature_name] = df.apply(
                lambda row: evaluate_feature(
                    row,
                    feature,
                ),
                axis=1,
            )

        except Exception as exc:
            print(f"\nFAILED FEATURE: {feature_name}")
            print(f"Source column: {feature.get('source')}")
            print(f"Feature definition: {feature}")
            raise

    return result


def main() -> None:

    print("Loading dataset...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Titles: {len(df)}")

    features = load_features()

    print(f"Features defined: {len(features)}")

    feature_matrix = build_feature_matrix(
        df,
        features,
    )

    feature_matrix.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)

    print(
        f"Rows: {feature_matrix.shape[0]}"
    )

    print(
        f"Columns: {feature_matrix.shape[1]}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print("\nFeature matrix:")
    print(
        feature_matrix.head()
    )


if __name__ == "__main__":
    main()