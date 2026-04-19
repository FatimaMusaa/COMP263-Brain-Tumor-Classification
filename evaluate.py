from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import DEFAULT_MODEL_NAME, MODELS_DIR, OUTPUTS_DIR
from data_utils import create_datasets
from model_factory import create_model, list_supported_models


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_name: str) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {output_name}")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / f"{output_name}_confusion_matrix.png")
    plt.close()


def evaluate_model(model_name: str) -> dict:
    _, _, test_ds, class_names = create_datasets()
    model_path = MODELS_DIR / f"{model_name}.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model weights: {model_path}")

    bundle = create_model(model_name)
    bundle.model.load_weights(model_path)

    probabilities = bundle.model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    y_true_batches = [np.argmax(labels.numpy(), axis=1) for _, labels in test_ds]
    y_true = np.concatenate(y_true_batches)

    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    with open(OUTPUTS_DIR / f"{model_name}_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    plot_confusion_matrix(cm, class_names, model_name)

    print(f"\nEvaluation for {model_name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")

    return {
        "model_name": model_name,
        "display_name": bundle.display_name,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "model_path": str(model_path),
    }


def save_leaderboard(results: list[dict]) -> None:
    ranked = sorted(results, key=lambda item: item["macro_f1"], reverse=True)
    with open(OUTPUTS_DIR / "model_comparison.json", "w", encoding="utf-8") as file:
        json.dump(ranked, file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained brain MRI models.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        choices=["all", *list_supported_models()],
        help="Model to evaluate.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    chosen_models = list_supported_models() if args.model == "all" else [args.model]

    results: list[dict] = []
    for chosen_model in chosen_models:
        model_file = Path(MODELS_DIR / f"{chosen_model}.keras")
        if model_file.exists():
            results.append(evaluate_model(chosen_model))
        else:
            print(f"Skipping {chosen_model}: model file not found.")

    if results:
        save_leaderboard(results)
