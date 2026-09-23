"""
Training script — Module 2: Hotel Review Sentiment Analysis
================================================================
Run:
    python src/sentiment_analysis/train.py

Requires that src/sentiment_analysis/preprocess.py has already been run.

Saves:
    models/sentiment_analysis/sentiment_bilstm.keras
    reports/figures/sentiment_analysis/training_curves.png
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.config import FIGURES_DIR, RANDOM_SEED, REVIEWS_PROCESSED_DIR, SENTIMENT_MODEL_DIR, ensure_dirs
from src.common.utils import set_global_seed
from src.sentiment_analysis.model import build_sentiment_model

FIG_DIR = os.path.join(FIGURES_DIR, "sentiment_analysis")

EPOCHS = 15
BATCH_SIZE = 128


def load_processed_data():
    X_train = np.load(os.path.join(REVIEWS_PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(REVIEWS_PROCESSED_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(REVIEWS_PROCESSED_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(REVIEWS_PROCESSED_DIR, "y_val.npy"))
    return X_train, y_train, X_val, y_val


def load_preprocess_config():
    with open(os.path.join(SENTIMENT_MODEL_DIR, "preprocess_config.json"), "r") as f:
        return json.load(f)


def plot_training_curves(history):
    ensure_dirs(FIG_DIR)
    hist = history.history

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist["loss"], label="train_loss")
    axes[0].plot(hist["val_loss"], label="val_loss")
    axes[0].set_title("Loss over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(hist["accuracy"], label="train_accuracy")
    axes[1].plot(hist["val_accuracy"], label="val_accuracy")
    axes[1].set_title("Accuracy over epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[train] Training curves saved to {FIG_DIR}/training_curves.png")


def run_training():
    set_global_seed(RANDOM_SEED)
    ensure_dirs(SENTIMENT_MODEL_DIR)

    X_train, y_train, X_val, y_val = load_processed_data()
    config = load_preprocess_config()
    print(f"[train] X_train: {X_train.shape}, X_val: {X_val.shape}, config: {config}")

    model = build_sentiment_model(vocab_size=config["vocab_size"], max_len=config["max_len"])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=4, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(SENTIMENT_MODEL_DIR, "sentiment_bilstm_best.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    final_model_path = os.path.join(SENTIMENT_MODEL_DIR, "sentiment_bilstm.keras")
    model.save(final_model_path)
    print(f"[train] Final model saved to {final_model_path}")

    plot_training_curves(history)
    print("[train] Done.")


if __name__ == "__main__":
    run_training()
