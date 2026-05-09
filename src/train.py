from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


EMBEDDINGS_DIR = Path("data/processed/embeddings")
MODELS_DIR = Path("models")
REPORT_DIR = Path("report")


def load_array(name: str) -> np.ndarray:
    return np.load(EMBEDDINGS_DIR / name)


def build_models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2_000,
            class_weight="balanced",
            random_state=42,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            C=3.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            alpha=1e-4,
            batch_size=64,
            learning_rate_init=1e-3,
            max_iter=80,
            early_stopping=True,
            random_state=42,
        ),
    }


def evaluate(model: object, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "weighted_f1": f1_score(y, pred, average="weighted"),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((EMBEDDINGS_DIR / "metadata.json").read_text(encoding="utf-8"))
    X_train = load_array("X_train.npy")
    y_train = load_array("y_train.npy")
    X_val = load_array("X_val.npy")
    y_val = load_array("y_val.npy")

    results = []
    best_name = ""
    best_score = -1.0

    for name, model in build_models().items():
        print(f"\nTraining {name}...")
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start

        metrics = evaluate(model, X_val, y_val)
        metrics["train_seconds"] = train_seconds
        metrics["model"] = name
        results.append(metrics)

        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump({"model": model, "metadata": metadata}, model_path)
        print(
            f"{name}: macro_f1={metrics['macro_f1']:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}, saved={model_path}"
        )

        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_name = name

    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    results_path = REPORT_DIR / "validation_results.csv"
    results_df.to_csv(results_path, index=False)

    best_payload = joblib.load(MODELS_DIR / f"{best_name}.joblib")
    best_payload["best_model_name"] = best_name
    joblib.dump(best_payload, MODELS_DIR / "best_model.joblib")

    print("\nValidation results:")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_name} (macro_f1={best_score:.4f})")
    print(f"Saved best model -> {MODELS_DIR / 'best_model.joblib'}")
    print(f"Saved validation table -> {results_path}")


if __name__ == "__main__":
    main()
