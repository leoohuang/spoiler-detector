"""
Generate a small synthetic spoiler-review dataset with GPT.

The synthetic set is used as the project's second data source after IMDb. Keep
the generated rows clearly marked with source="gpt_synthetic" so the final
report can discuss data provenance and limitations transparently.
"""

from __future__ import annotations

import json
import os
import re
import time
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


OUTPUT_PATH = Path("data/raw/gpt_synthetic_reviews.csv")
MODEL = "gpt-4o-mini"


PROMPT = """
Generate {n} short English movie-review snippets for a spoiler detector.

Return valid JSON only, as an array of objects with these fields:
- text: the review snippet, 30 to 220 words
- has_spoiler: true or false
- severity: one of Safe, Mild, Major

Definitions:
- Safe: opinion only, no plot reveal.
- Mild: mentions mood, setup, or non-critical scene information.
- Major: reveals deaths, twists, identities, ending, or solution.

Balance the output across Safe, Mild, and Major. Do not include real movie
titles or copyrighted plot text; use fictional names and generic scenarios.

Important content rules:
- Do not write meta phrases such as "I won't spoil it", "no spoilers", "without
  giving anything away", or "go in blind".
- Safe examples must not reveal plot events.
- Mild examples may mention setup, tone, or non-critical scenes, but not the
  ending, killer, secret identity, death, solution, or final twist.
- Major examples must explicitly reveal a fictional plot fact, such as "the
  mentor is the villain", "the sister dies", "the missing pilot is alive", or
  "the artifact destroys the city in the finale".
"""


def parse_json_array(content: str) -> list[dict]:
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    parsed = json.loads(content)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array")
    return parsed


def normalize_rows(rows: list[dict]) -> list[dict]:
    valid_severities = {"Safe", "Mild", "Major"}
    banned_meta = (
        "won't spoil",
        "will not spoil",
        "no spoilers",
        "without giving",
        "go in blind",
    )
    normalized = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        severity = str(row.get("severity", "")).strip().title()
        if not text or severity not in valid_severities:
            continue
        lowered = text.lower()
        if any(phrase in lowered for phrase in banned_meta):
            continue
        normalized.append(
            {
                "text": text,
                "has_spoiler": severity != "Safe",
                "severity": severity,
                "source": "gpt_synthetic",
            }
        )
    return normalized


def generate_batch(client: OpenAI, n: int) -> list[dict]:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.8,
        messages=[
            {"role": "system", "content": "You create clean JSON datasets."},
            {"role": "user", "content": PROMPT.format(n=n)},
        ],
    )
    content = response.choices[0].message.content or "[]"
    return normalize_rows(parse_json_array(content))


def main(total: int = 900, batch_size: int = 30, max_retries: int = 3) -> None:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    rows: list[dict] = []

    for _ in tqdm(range(0, total, batch_size), desc="Generating synthetic rows"):
        for attempt in range(max_retries):
            try:
                rows.extend(generate_batch(client, batch_size))
                break
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise
                wait_seconds = 2**attempt
                print(f"Batch failed ({exc}); retrying in {wait_seconds}s")
                time.sleep(wait_seconds)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--total", type=int, default=900)
    parser.add_argument("--batch-size", type=int, default=30)
    args = parser.parse_args()
    main(total=args.total, batch_size=args.batch_size)
