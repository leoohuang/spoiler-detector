from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_PATH = Path("data/processed/labeled.csv")
OUTPUT_DIR = Path("data/processed")


def save_split(df: pd.DataFrame, name: str) -> None:
    path = OUTPUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"Saved {name}: {len(df):,} rows -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    args = parser.parse_args()

    total = args.train_size + args.val_size + args.test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train-size + val-size + test-size must equal 1.0")

    df = pd.read_csv(INPUT_PATH)
    required_columns = {"text", "source", "has_spoiler", "severity"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["severity"].isna().any():
        raise ValueError("Found rows without severity labels")

    train, temp = train_test_split(
        df,
        train_size=args.train_size,
        random_state=args.seed,
        stratify=df["severity"],
    )
    relative_val_size = args.val_size / (args.val_size + args.test_size)
    val, test = train_test_split(
        temp,
        train_size=relative_val_size,
        random_state=args.seed,
        stratify=temp["severity"],
    )

    save_split(train, "train")
    save_split(val, "val")
    save_split(test, "test")

    print("\nSeverity distribution:")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        print(f"\n{name}")
        print(split["severity"].value_counts(normalize=True).mul(100).round(1))

    print("\nSource distribution:")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        print(f"\n{name}")
        print(split["source"].value_counts(normalize=True).mul(100).round(1))


if __name__ == "__main__":
    main()
