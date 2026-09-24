"""
Training script — Module 3: Room Demand Forecasting
========================================================
Run:
    python src/demand_forecasting/train.py

Requires that src/demand_forecasting/preprocess.py has already been run.

Saves:
    models/demand_forecasting/demand_lstm.keras
    reports/figures/demand_forecasting/training_curves.png
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
from src.common.config import DEMAND_MODEL_DIR, DEMAND_PROCESSED_DIR, FIGURES_DIR, RANDOM_SEED, ensure_dirs
from src.common.utils import set_global_seed
from src.demand_forecasting.model import build_lstm_model

FIG_DIR = os.path.join(FIGURES_DIR, "demand_forecasting")

EPOCHS = 100
BATCH_SIZE = 32


def load_processed_data():
    X_train = np.load(os.path.join(DEMAND_PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DEMAND_PROCESSED_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DEMAND_PROCESSED_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DEMAND_PROCESSED_DIR, "y_val.npy"))
    return X_train, y_train, X_val, y_val


def load_window_size():
    with open(os.path.join(DEMAND_MODEL_DIR, "window_config.json"), "r") as f:
        return json.load(f)["window_size"]


def plot_training_curves(history):
    ensure_dirs(FIG_DIR)
    hist = history.history

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist["loss"], label="train_loss (MSE)")
    axes[0].plot(hist["val_loss"], label="val_loss (MSE)")
    axes[0].set_title("Loss over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(hist["mae"], label="train_mae")
    axes[1].plot(hist["val_mae"], label="val_mae")
    axes[1].set_title("MAE over epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[train] Training curves saved to {FIG_DIR}/training_curves.png")


def run_training():
    set_global_seed(RANDOM_SEED)
    ensure_dirs(DEMAND_MODEL_DIR)

    X_train, y_train, X_val, y_val = load_processed_data()
    window_size = load_window_size()
    print(f"[train] X_train: {X_train.shape}, X_val: {X_val.shape}, window_size: {window_size}")

    model = build_lstm_model(window_size=window_size)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(DEMAND_MODEL_DIR, "demand_lstm_best.keras"),
            monitor="val_loss",
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

    final_model_path = os.path.join(DEMAND_MODEL_DIR, "demand_lstm.keras")
    model.save(final_model_path)
    print(f"[train] Final model saved to {final_model_path}")

    plot_training_curves(history)
    print("[train] Done.")


if __name__ == "__main__":
    run_training()
