from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw/archive")
REVIEWS_PATH = RAW_DIR / "IMDB_reviews.json"
MOVIES_PATH = RAW_DIR / "IMDB_movie_details.json"


def read_jsonl_sample(path: Path, n: int = 20_000) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            if i >= n:
                break
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def count_jsonl_rows(path: Path) -> int:
    with path.open("rb") as file:
        return sum(1 for _ in file)


def main() -> None:
    if not REVIEWS_PATH.exists() or not MOVIES_PATH.exists():
        raise FileNotFoundError(
            "IMDb files not found. Put IMDB_reviews.json and "
            "IMDB_movie_details.json under data/raw/archive/."
        )

    reviews_total = count_jsonl_rows(REVIEWS_PATH)
    movies_total = count_jsonl_rows(MOVIES_PATH)
    sample = read_jsonl_sample(REVIEWS_PATH)

    print("IMDb dataset found")
    print(f"Reviews file: {REVIEWS_PATH} ({REVIEWS_PATH.stat().st_size / 1024**2:.1f} MB)")
    print(f"Movie details file: {MOVIES_PATH} ({MOVIES_PATH.stat().st_size / 1024**2:.1f} MB)")
    print(f"Total reviews: {reviews_total:,}")
    print(f"Total movie rows: {movies_total:,}")
    print(f"Review columns: {list(sample.columns)}")
    print(f"Sample spoiler rate: {sample['is_spoiler'].mean():.2%}")
    print("Sample rows by label:")
    print(sample["is_spoiler"].value_counts().rename(index={True: "spoiler", False: "safe"}))


if __name__ == "__main__":
    main()
