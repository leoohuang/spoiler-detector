from __future__ import annotations

import argparse
import html
import json
import random
import re
from pathlib import Path

import pandas as pd


IMDB_REVIEWS_PATH = Path("data/raw/archive/IMDB_reviews.json")
SYNTHETIC_PATH = Path("data/raw/gpt_synthetic_reviews.csv")
OUTPUT_PATH = Path("data/processed/cleaned.csv")


def clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def keep_text(text: str, min_chars: int, max_chars: int) -> bool:
    return min_chars <= len(text) <= max_chars


def reservoir_add(bucket: list[dict], row: dict, seen: int, limit: int) -> None:
    if len(bucket) < limit:
        bucket.append(row)
        return
    index = random.randint(0, seen - 1)
    if index < limit:
        bucket[index] = row


def load_imdb_sample(
    path: Path,
    safe_limit: int,
    spoiler_limit: int,
    min_chars: int,
    max_chars: int,
    seed: int,
) -> pd.DataFrame:
    random.seed(seed)
    safe_rows: list[dict] = []
    spoiler_rows: list[dict] = []
    safe_seen = 0
    spoiler_seen = 0

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            text = clean_text(item.get("review_text"))
            if not keep_text(text, min_chars, max_chars):
                continue

            has_spoiler = bool(item.get("is_spoiler"))
            row = {
                "text": text,
                "source": "imdb",
                "has_spoiler": has_spoiler,
                "severity": "",
                "movie_id": item.get("movie_id", ""),
            }
            if has_spoiler:
                spoiler_seen += 1
                reservoir_add(spoiler_rows, row, spoiler_seen, spoiler_limit)
            else:
                safe_seen += 1
                reservoir_add(safe_rows, row, safe_seen, safe_limit)

    print(f"IMDb candidates after length filter: safe={safe_seen:,}, spoiler={spoiler_seen:,}")
    return pd.DataFrame(safe_rows + spoiler_rows)


def load_synthetic(path: Path, min_chars: int, max_chars: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].map(lambda text: keep_text(text, min_chars, max_chars))]
    df["source"] = "gpt_synthetic"
    df["movie_id"] = ""
    return df[["text", "source", "has_spoiler", "severity", "movie_id"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imdb-safe", type=int, default=2_000)
    parser.add_argument("--imdb-spoiler", type=int, default=2_000)
    parser.add_argument("--min-chars", type=int, default=30)
    parser.add_argument("--max-chars", type=int, default=1_500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not IMDB_REVIEWS_PATH.exists():
        raise FileNotFoundError(f"Missing IMDb reviews file: {IMDB_REVIEWS_PATH}")
    if not SYNTHETIC_PATH.exists():
        raise FileNotFoundError(f"Missing synthetic data file: {SYNTHETIC_PATH}")

    imdb = load_imdb_sample(
        IMDB_REVIEWS_PATH,
        safe_limit=args.imdb_safe,
        spoiler_limit=args.imdb_spoiler,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        seed=args.seed,
    )
    synthetic = load_synthetic(SYNTHETIC_PATH, args.min_chars, args.max_chars)

    combined = pd.concat([imdb, synthetic], ignore_index=True)
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["text"]).sample(frac=1, random_state=args.seed)
    combined = combined.reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(combined):,} rows to {OUTPUT_PATH}")
    print(f"Removed duplicates: {before_dedup - len(combined):,}")
    print("\nSource distribution:")
    print(combined["source"].value_counts())
    print("\nSpoiler distribution:")
    print(combined["has_spoiler"].value_counts())
    print("\nSeverity distribution for rows that already have severity:")
    print(combined.loc[combined["severity"].fillna("") != "", "severity"].value_counts())


if __name__ == "__main__":
    main()
