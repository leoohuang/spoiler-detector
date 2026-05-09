from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


INPUT_PATH = Path("data/processed/cleaned.csv")
OUTPUT_PATH = Path("data/processed/labeled.csv")
MODEL = "gpt-4o-mini"


SYSTEM_PROMPT = """
You are labeling movie-review snippets for spoiler severity.
Return valid JSON only.
"""

USER_PROMPT = """
Each item below is already known to contain some spoiler content.
Classify each item as exactly one of:

- Mild: mentions atmosphere, setup, a non-critical scene, emotional tone, or
  broad plot direction without revealing a key twist, death, identity, ending,
  solution, or final outcome.
- Major: reveals a key plot turn, death, hidden identity, ending, solution,
  final outcome, or a fact that would substantially reduce surprise.

Return a JSON array with objects:
{{"id": <id>, "severity": "Mild" or "Major"}}

Items:
{items_json}
"""


def parse_json_array(content: str) -> list[dict]:
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    parsed = json.loads(content)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array")
    return parsed


def classify_batch(client: OpenAI, batch: list[dict], max_retries: int = 3) -> dict[int, str]:
    items_json = json.dumps(batch, ensure_ascii=False)
    prompt = USER_PROMPT.format(items_json=items_json)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content or "[]"
            rows = parse_json_array(content)
            labels: dict[int, str] = {}
            for row in rows:
                severity = str(row.get("severity", "")).strip().title()
                if severity not in {"Mild", "Major"}:
                    continue
                labels[int(row["id"])] = severity
            missing = {item["id"] for item in batch} - set(labels)
            if missing:
                raise ValueError(f"Missing labels for ids: {sorted(missing)[:5]}")
            return labels
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)

    raise RuntimeError("Unreachable retry state")


def needs_label(row: pd.Series) -> bool:
    severity = "" if pd.isna(row.get("severity")) else str(row.get("severity")).strip()
    return bool(row["has_spoiler"]) and severity == ""


def save(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Label only N pending IMDb spoiler rows")
    args = parser.parse_args()

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    input_path = OUTPUT_PATH if OUTPUT_PATH.exists() else INPUT_PATH
    df = pd.read_csv(input_path)
    df["severity"] = df["severity"].fillna("")
    df.loc[~df["has_spoiler"].astype(bool), "severity"] = "Safe"

    pending_index = [idx for idx, row in df.iterrows() if needs_label(row)]
    if args.limit:
        pending_index = pending_index[: args.limit]

    print(f"Rows loaded from {input_path}: {len(df):,}")
    print(f"Pending IMDb spoiler rows to label now: {len(pending_index):,}")

    batches = []
    for start in range(0, len(pending_index), args.batch_size):
        batch_ids = pending_index[start : start + args.batch_size]
        batch = [
            {
                "id": int(idx),
                "text": str(df.at[idx, "text"])[:1_500],
            }
            for idx in batch_ids
        ]
        batches.append(batch)

    if not batches:
        save(df)
        print(f"No pending rows. Saved {OUTPUT_PATH}")
        return

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(classify_batch, client, batch) for batch in batches]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Labeling batches"):
            labels = future.result()
            for idx, severity in labels.items():
                df.at[idx, "severity"] = severity
            completed += len(labels)
            if completed % 100 == 0:
                save(df)

    save(df)
    print(f"Saved labeled data to {OUTPUT_PATH}")
    print("\nSeverity distribution:")
    print(df["severity"].value_counts())
    print("\nSource x severity:")
    print(pd.crosstab(df["source"], df["severity"]))


if __name__ == "__main__":
    main()
