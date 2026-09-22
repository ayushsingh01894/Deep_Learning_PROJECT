"""
Evaluation script — Module 1: Hotel Booking Prediction
==========================================================
Run:
    python src/booking_prediction/evaluate.py

Requires a trained model at models/booking_prediction/booking_ann.h5
(produced by train.py) and processed test data (produced by preprocess.py).

Saves:
    reports/metrics/booking_prediction_metrics.json
    reports/figures/booking_prediction/confusion_matrix.png
    reports/figures/booking_prediction/roc_curve.png

IMPORTANT: This script computes metrics ONLY from the actual trained model's
predictions on the real held-out test set. No numbers are hardcoded.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tensorflow import keras

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.config import BOOKING_MODEL_DIR, BOOKING_PROCESSED_DIR, FIGURES_DIR, METRICS_DIR, ensure_dirs
from src.common.utils import save_metrics_json

FIG_DIR = os.path.join(FIGURES_DIR, "booking_prediction")
MODEL_PATH = os.path.join(BOOKING_MODEL_DIR, "booking_ann.keras")
METRICS_PATH = os.path.join(METRICS_DIR, "booking_prediction_metrics.json")


def load_test_data():
    X_test = np.load(os.path.join(BOOKING_PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(BOOKING_PROCESSED_DIR, "y_test.npy"))
    return X_test, y_test


def plot_confusion_matrix(y_true, y_pred):
    ensure_dirs(FIG_DIR)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Canceled", "Canceled"],
        yticklabels=["Not Canceled", "Canceled"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Booking Cancellation Prediction")
    fig.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Booking Cancellation Prediction")
    ax.legend()
    fig.savefig(os.path.join(FIG_DIR, "roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return auc


def run_evaluation():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run train.py first."
        )

    X_test, y_test = load_test_data()
    model = keras.models.load_model(MODEL_PATH)

    y_proba = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "test_set_size": int(len(y_test)),
        "positive_class_ratio_in_test": float(y_test.mean()),
    }

    print("[evaluate] Test set metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba)
    save_metrics_json(metrics, METRICS_PATH)

    print("[evaluate] Done.")
    return metrics


if __name__ == "__main__":
    run_evaluation()
