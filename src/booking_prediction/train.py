"""
Training script — Module 1: Hotel Booking Prediction
========================================================
Run:
    python src/booking_prediction/train.py

Requires that src/booking_prediction/preprocess.py has already been run
(so that data/processed/booking_demand/*.npy exist).

Saves:
    models/booking_prediction/booking_ann.h5
    reports/figures/booking_prediction/training_curves.png
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.booking_prediction.model import build_ann_model
from src.common.config import BOOKING_MODEL_DIR, BOOKING_PROCESSED_DIR, FIGURES_DIR, RANDOM_SEED, ensure_dirs
from src.common.utils import set_global_seed

FIG_DIR = os.path.join(FIGURES_DIR, "booking_prediction")

EPOCHS = 50
BATCH_SIZE = 256


def load_processed_data():
    X_train = np.load(os.path.join(BOOKING_PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(BOOKING_PROCESSED_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(BOOKING_PROCESSED_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(BOOKING_PROCESSED_DIR, "y_val.npy"))
    return X_train, y_train, X_val, y_val


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
    ensure_dirs(BOOKING_MODEL_DIR)

    X_train, y_train, X_val, y_val = load_processed_data()
    print(f"[train] X_train: {X_train.shape}, X_val: {X_val.shape}")

    model = build_ann_model(input_dim=X_train.shape[1])
    model.summary()

    # Handle class imbalance with class weights (booking cancellation is imbalanced)
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    class_weight = {0: len(y_train) / (2 * n_neg), 1: len(y_train) / (2 * n_pos)}
    print(f"[train] Class weights: {class_weight}")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=8, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(BOOKING_MODEL_DIR, "booking_ann_best.keras"),
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
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    final_model_path = os.path.join(BOOKING_MODEL_DIR, "booking_ann.keras")
    model.save(final_model_path)
    print(f"[train] Final model saved to {final_model_path}")

    plot_training_curves(history)
    print("[train] Done.")


if __name__ == "__main__":
    run_training()
