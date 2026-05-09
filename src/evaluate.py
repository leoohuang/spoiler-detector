from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


EMBEDDINGS_DIR = Path("data/processed/embeddings")
MODELS_DIR = Path("models")
REPORT_DIR = Path("report")
FIGURES_DIR = REPORT_DIR / "figures"
MPL_CACHE_DIR = Path(".matplotlib_cache")
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR.resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR.resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import seaborn as sns


MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "SVM RBF": "svm_rbf.joblib",
    "Random Forest": "random_forest.joblib",
    "MLP": "mlp.joblib",
}


def load_test_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_test = np.load(EMBEDDINGS_DIR / "X_test.npy")
    y_test = np.load(EMBEDDINGS_DIR / "y_test.npy")
    metadata = json.loads((EMBEDDINGS_DIR / "metadata.json").read_text(encoding="utf-8"))
    return X_test, y_test, metadata["label_classes"]


def evaluate_model(name: str, model: object, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float | str]:
    pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        pred,
        average="macro",
        zero_division=0,
    )
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "weighted_f1": f1_score(y_test, pred, average="weighted"),
    }


def save_model_comparison(results: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4.5))
    order = results.sort_values("macro_f1", ascending=False)
    ax = sns.barplot(data=order, x="macro_f1", y="model", color="#4C78A8")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro F1 on test set")
    ax.set_ylabel("")
    ax.set_title("Model Comparison")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=4)
    plt.tight_layout()
    path = FIGURES_DIR / "models_comparison.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved {path}")


def save_confusion_matrix(y_test: np.ndarray, pred: np.ndarray, class_names: list[str]) -> None:
    matrix = confusion_matrix(y_test, pred)
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Best Model Confusion Matrix")
    plt.tight_layout()
    path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved {path}")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_test, y_test, class_names = load_test_data()
    rows = []
    predictions = {}

    for display_name, filename in MODEL_FILES.items():
        payload = joblib.load(MODELS_DIR / filename)
        model = payload["model"]
        pred = model.predict(X_test)
        predictions[display_name] = pred
        rows.append(evaluate_model(display_name, model, X_test, y_test))

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    results_path = REPORT_DIR / "test_results.csv"
    results.to_csv(results_path, index=False)

    best_name = str(results.iloc[0]["model"])
    best_pred = predictions[best_name]
    report_text = classification_report(
        y_test,
        best_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    (REPORT_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")

    save_model_comparison(results)
    save_confusion_matrix(y_test, best_pred, class_names)

    print("\nTest results:")
    print(results.to_string(index=False))
    print(f"\nBest test model: {best_name}")
    print("\nClassification report:")
    print(report_text)
    print(f"Saved test table -> {results_path}")


if __name__ == "__main__":
    main()
