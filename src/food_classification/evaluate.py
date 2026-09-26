"""
Evaluation script — Module 4: Food Image Classification
============================================================
Run:
    python src/food_classification/evaluate.py

Requires a trained model at models/food_classification/food_cnn.keras
and the test manifest (from preprocess.py).

Saves:
    reports/metrics/food_classification_metrics.json
    reports/figures/food_classification/confusion_matrix.png
    reports/figures/food_classification/misclassified_samples.png   (error analysis)

IMPORTANT: All metrics are computed from the actual trained model's
predictions on the real held-out test set. No numbers are hardcoded.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow import keras

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.config import FIGURES_DIR, FOOD101_PROCESSED_DIR, FOOD_MODEL_DIR, METRICS_DIR, ensure_dirs
from src.common.utils import save_metrics_json
from src.food_classification.model import IMG_SIZE

FIG_DIR = os.path.join(FIGURES_DIR, "food_classification")
MODEL_PATH = os.path.join(FOOD_MODEL_DIR, "food_cnn.keras")
METRICS_PATH = os.path.join(METRICS_DIR, "food_classification_metrics.json")


def load_test_manifest():
    return pd.read_csv(os.path.join(FOOD101_PROCESSED_DIR, "test_manifest.csv"))


def load_class_names():
    with open(os.path.join(FOOD_MODEL_DIR, "class_names.json"), "r") as f:
        return json.load(f)


def build_eval_dataset(df: pd.DataFrame, batch_size: int = 32) -> tf.data.Dataset:
    def _load(filepath, label):
        image = tf.io.read_file(filepath)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((df["filepath"].values, df["label_id"].values))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def plot_confusion_matrix(y_true, y_pred, class_names):
    ensure_dirs(FIG_DIR)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Food Image Classification")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_misclassified_samples(test_df, y_true, y_pred, class_names, n_samples=12):
    """Error analysis: show a grid of images the model got wrong, with true vs predicted labels."""
    ensure_dirs(FIG_DIR)
    wrong_idx = np.where(y_true != y_pred)[0]

    if len(wrong_idx) == 0:
        print("[evaluate] No misclassified samples found — skipping error analysis plot.")
        return

    sample_idx = np.random.RandomState(42).choice(wrong_idx, size=min(n_samples, len(wrong_idx)), replace=False)

    n_cols = 4
    n_rows = (len(sample_idx) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, idx in zip(axes, sample_idx):
        filepath = test_df.iloc[idx]["filepath"]
        try:
            img = Image.open(filepath)
            ax.imshow(img)
        except (FileNotFoundError, OSError):
            ax.text(0.5, 0.5, "image not found", ha="center")
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=9, color="red")
        ax.axis("off")

    for j in range(len(sample_idx), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Error Analysis — Sample Misclassified Images", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "misclassified_samples.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Error analysis plot saved ({len(sample_idx)} misclassified samples shown)")


def run_evaluation():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run train.py first."
        )

    test_df = load_test_manifest()
    class_names = load_class_names()
    model = keras.models.load_model(MODEL_PATH)

    test_ds = build_eval_dataset(test_df)
    y_true = test_df["label_id"].values

    y_proba = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "test_set_size": int(len(y_true)),
        "num_classes": len(class_names),
        "per_class_report": classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        ),
    }

    print("[evaluate] Test set metrics:")
    print(f"  accuracy: {metrics['accuracy']}")
    print(f"  precision_macro: {metrics['precision_macro']}")
    print(f"  recall_macro: {metrics['recall_macro']}")
    print(f"  f1_macro: {metrics['f1_macro']}")

    plot_confusion_matrix(y_true, y_pred, class_names)
    plot_misclassified_samples(test_df, y_true, y_pred, class_names)
    save_metrics_json(metrics, METRICS_PATH)

    print("[evaluate] Done.")
    return metrics


if __name__ == "__main__":
    run_evaluation()
