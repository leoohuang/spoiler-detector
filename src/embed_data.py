from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder


PROCESSED_DIR = Path("data/processed")
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"
DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"


def encode_split(
    model: SentenceTransformer,
    split_name: str,
    label_encoder: LabelEncoder,
    batch_size: int,
) -> None:
    path = PROCESSED_DIR / f"{split_name}.csv"
    df = pd.read_csv(path)
    texts = df["text"].astype(str).tolist()
    labels = label_encoder.transform(df["severity"].astype(str))

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(EMBEDDINGS_DIR / f"X_{split_name}.npy", embeddings.astype(np.float32))
    np.save(EMBEDDINGS_DIR / f"y_{split_name}.npy", labels.astype(np.int64))
    print(
        f"{split_name}: X={embeddings.shape}, y={labels.shape} -> "
        f"{EMBEDDINGS_DIR / f'X_{split_name}.npy'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    label_encoder = LabelEncoder()
    label_encoder.fit(train["severity"].astype(str))

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "embedding_model": args.model_name,
        "normalize_embeddings": True,
        "label_classes": label_encoder.classes_.tolist(),
    }
    (EMBEDDINGS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Loading embedding model: {args.model_name}")
    model = SentenceTransformer(args.model_name)
    print(f"Label classes: {label_encoder.classes_.tolist()}")

    for split_name in ["train", "val", "test"]:
        encode_split(model, split_name, label_encoder, args.batch_size)


if __name__ == "__main__":
    main()
