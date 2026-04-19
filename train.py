from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import tensorflow as tf

from config import DEFAULT_MODEL_NAME, EPOCHS, MODELS_DIR, OUTPUTS_DIR
from data_utils import create_datasets
from model_factory import compile_model, create_model, list_supported_models


def plot_history(history_dict: dict, output_prefix: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(history_dict["accuracy"], label="Train Accuracy")
    plt.plot(history_dict["val_accuracy"], label="Validation Accuracy")
    plt.title(f"Accuracy - {output_prefix}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / f"{output_prefix}_accuracy.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(history_dict["loss"], label="Train Loss")
    plt.plot(history_dict["val_loss"], label="Validation Loss")
    plt.title(f"Loss - {output_prefix}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / f"{output_prefix}_loss.png")
    plt.close()


def save_history(history_dict: dict, filename: str) -> None:
    with open(OUTPUTS_DIR / filename, "w", encoding="utf-8") as file:
        json.dump(history_dict, file, indent=2)


def merge_histories(*histories: tf.keras.callbacks.History) -> dict[str, list]:
    merged: dict[str, list] = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(values)
    return merged


def get_common_callbacks(model_name: str):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / f"{model_name}.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def save_training_summary(model_name: str, history_dict: dict, class_names: list[str]) -> None:
    summary = {
        "model_name": model_name,
        "class_names": class_names,
        "best_val_accuracy": max(history_dict["val_accuracy"]),
        "best_val_loss": min(history_dict["val_loss"]),
        "final_train_accuracy": history_dict["accuracy"][-1],
        "final_val_accuracy": history_dict["val_accuracy"][-1],
        "epochs_ran": len(history_dict["accuracy"]),
    }
    with open(OUTPUTS_DIR / f"{model_name}_training_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def train_single_model(model_name: str) -> None:
    train_ds, val_ds, _, class_names = create_datasets()
    bundle = create_model(model_name)
    callbacks = get_common_callbacks(model_name)

    print(f"\nTraining {bundle.display_name}...")
    print(bundle.recommendation)

    if bundle.backbone is None:
        history = bundle.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1,
        )
        history_dict = history.history
    else:
        frozen_epochs = max(5, EPOCHS // 3)
        print(f"Stage 1: training classifier head for {frozen_epochs} epochs")
        stage_one = bundle.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=frozen_epochs,
            callbacks=callbacks,
            verbose=1,
        )

        print("Stage 2: fine-tuning top backbone layers")
        bundle.backbone.trainable = True
        for layer in bundle.backbone.layers[:-30]:
            layer.trainable = False

        compile_model(
            bundle.model,
            learning_rate=1e-5,
            class_weights=None,
        )

        stage_two = bundle.model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=frozen_epochs,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1,
        )
        history_dict = merge_histories(stage_one, stage_two)

    plot_history(history_dict, model_name)
    save_history(history_dict, f"{model_name}_history.json")
    save_training_summary(model_name, history_dict, class_names)

    print(f"Finished training {model_name}")
    print(f"Classes: {class_names}")
    print(f"Best val_accuracy: {max(history_dict['val_accuracy']):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train brain MRI tumor classifiers.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        choices=["all", *list_supported_models()],
        help="Model to train.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_models = (
        list_supported_models() if args.model == "all" else [args.model]
    )

    for selected_model in selected_models:
        train_single_model(selected_model)
