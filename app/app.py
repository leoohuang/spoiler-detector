from __future__ import annotations

from pathlib import Path

import gradio as gr
import joblib
import numpy as np
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
HF_MODEL_REPO = "leoole/spoiler-detector"


LABEL_DETAILS = {
    "Safe": {
        "emoji": "🟢",
        "title": "Safe",
        "description": "No meaningful spoiler detected.",
    },
    "Mild": {
        "emoji": "🟡",
        "title": "Mild Spoiler",
        "description": "Contains broad setup, tone, or non-critical plot information.",
    },
    "Major": {
        "emoji": "🔴",
        "title": "Major Spoiler",
        "description": "May reveal a key twist, death, identity, ending, or outcome.",
    },
}


def load_pipeline() -> tuple[object, SentenceTransformer, list[str]]:
    model_path = MODEL_PATH
    if not model_path.exists():
        model_path = Path(hf_hub_download(repo_id=HF_MODEL_REPO, filename="best_model.joblib"))
    payload = joblib.load(model_path)
    metadata = payload["metadata"]
    model = payload["model"]
    embedder = SentenceTransformer(metadata["embedding_model"])
    return model, embedder, metadata["label_classes"]


CLASSIFIER, EMBEDDER, LABEL_CLASSES = load_pipeline()


def confidence_from_model(model: object, embedding: np.ndarray, label_id: int) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(embedding)[0]
        return float(probabilities[label_id])
    if hasattr(model, "decision_function"):
        scores = model.decision_function(embedding)
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        shifted = scores[0] - np.max(scores[0])
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        return float(probabilities[label_id])
    return 0.0


def analyze_review(review: str) -> tuple[str, str, str]:
    text = review.strip()
    if not text:
        return "Paste a movie review first.", "", ""

    embedding = EMBEDDER.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    label_id = int(CLASSIFIER.predict(embedding)[0])
    label = LABEL_CLASSES[label_id]
    confidence = confidence_from_model(CLASSIFIER, embedding, label_id)
    details = LABEL_DETAILS[label]

    result = f"{details['emoji']} {details['title']} ({confidence:.0%})"
    explanation = details["description"]
    original = text
    return result, explanation, original


EXAMPLES = [
    [
        "The performances are excellent and the pacing is tense throughout, but I can recommend it without saying anything about the plot."
    ],
    [
        "The second act has a tense confrontation that changes how the hero sees their mission, but the movie saves its biggest answers for later."
    ],
    [
        "The final twist reveals that the hero's closest friend was secretly working for the villain the entire time."
    ],
]


with gr.Blocks(title="Multi-Source Spoiler Detector") as demo:
    gr.Markdown("# Multi-Source Spoiler Detector")
    gr.Markdown("Classify a movie review as Safe, Mild Spoiler, or Major Spoiler.")

    with gr.Row():
        review_input = gr.Textbox(
            label="Movie review",
            placeholder="Paste a movie review here...",
            lines=8,
        )

    analyze_button = gr.Button("Analyze", variant="primary")
    result_output = gr.Textbox(label="Result", interactive=False)
    explanation_output = gr.Textbox(label="Why", interactive=False)

    with gr.Accordion("Original text", open=False):
        original_output = gr.Textbox(label="Original", lines=6, interactive=False)

    gr.Examples(
        examples=EXAMPLES,
        inputs=review_input,
    )

    analyze_button.click(
        analyze_review,
        inputs=review_input,
        outputs=[result_output, explanation_output, original_output],
    )


if __name__ == "__main__":
    demo.launch()
