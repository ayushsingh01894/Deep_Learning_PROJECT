"""
Training script — Module 4: Food Image Classification
==========================================================
Run:
    python src/food_classification/train.py

Requires that src/food_classification/preprocess.py has already been run
(so that data/processed/food101/*_manifest.csv exist).

Training strategy (two phases):
    Phase 1 — train only the classification head, MobileNetV2 base frozen.
    Phase 2 — unfreeze the base and fine-tune end-to-end at a much lower
              learning rate for a few more epochs.

Saves:
    models/food_classification/food_cnn.keras
    reports/figures/food_classification/training_curves.png
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.config import FIGURES_DIR, FOOD101_PROCESSED_DIR, FOOD_MODEL_DIR, RANDOM_SEED, ensure_dirs
from src.common.utils import set_global_seed
from src.food_classification.model import IMG_SIZE, build_food_cnn_model

FIG_DIR = os.path.join(FIGURES_DIR, "food_classification")

BATCH_SIZE = 32
PHASE1_EPOCHS = 10  # frozen-base head training
PHASE2_EPOCHS = 5  # fine-tuning epochs (set to 0 to skip fine-tuning)
FINE_TUNE_LR = 1e-5


def load_manifests():
    train_df = pd.read_csv(os.path.join(FOOD101_PROCESSED_DIR, "train_manifest.csv"))
    val_df = pd.read_csv(os.path.join(FOOD101_PROCESSED_DIR, "val_manifest.csv"))
    return train_df, val_df


def load_class_names():
    with open(os.path.join(FOOD_MODEL_DIR, "class_names.json"), "r") as f:
        return json.load(f)


def _load_and_preprocess(filepath, label, augment: bool):
    image = tf.io.read_file(filepath)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])

    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.15)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)

    return image, label


def build_dataset(df: pd.DataFrame, augment: bool, shuffle: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((df["filepath"].values, df["label_id"].values))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=RANDOM_SEED)
    ds = ds.map(
        lambda fp, lb: _load_and_preprocess(fp, lb, augment),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def plot_training_curves(histories: list):
    """histories: list of (phase_name, keras.callbacks.History) to concatenate onto one plot."""
    ensure_dirs(FIG_DIR)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    offset = 0
    for phase_name, history in histories:
        hist = history.history
        n = len(hist["loss"])
        xs = range(offset, offset + n)
        axes[0].plot(xs, hist["loss"], label=f"{phase_name} train_loss")
        axes[0].plot(xs, hist["val_loss"], label=f"{phase_name} val_loss", linestyle="--")
        axes[1].plot(xs, hist["accuracy"], label=f"{phase_name} train_acc")
        axes[1].plot(xs, hist["val_accuracy"], label=f"{phase_name} val_acc", linestyle="--")
        offset += n

    axes[0].set_title("Loss over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=8)
    axes[1].set_title("Accuracy over epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[train] Training curves saved to {FIG_DIR}/training_curves.png")


def run_training():
    set_global_seed(RANDOM_SEED)
    ensure_dirs(FOOD_MODEL_DIR)

    train_df, val_df = load_manifests()
    class_names = load_class_names()
    print(f"[train] Train images: {len(train_df)}, Val images: {len(val_df)}, Classes: {len(class_names)}")

    train_ds = build_dataset(train_df, augment=True, shuffle=True)
    val_ds = build_dataset(val_df, augment=False, shuffle=False)

    model = build_food_cnn_model(num_classes=len(class_names), base_trainable=False)
    model.summary()

    callbacks_phase1 = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    print("\n[train] === Phase 1: training classification head (base frozen) ===")
    history1 = model.fit(
        train_ds, validation_data=val_ds, epochs=PHASE1_EPOCHS, callbacks=callbacks_phase1,
        shuffle=False, verbose=2,
    )

    histories = [("phase1", history1)]

    if PHASE2_EPOCHS > 0:
        print("\n[train] === Phase 2: fine-tuning (unfreezing MobileNetV2 base) ===")
        # Unfreeze the base model and recompile with a much lower LR.
        # Find the MobileNetV2 base by type rather than a hardcoded name, since
        # Keras versions can name the nested functional model differently.
        base_layer = next(l for l in model.layers if isinstance(l, keras.Model))
        base_layer.trainable = True
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        callbacks_phase2 = [
            keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=3, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(FOOD_MODEL_DIR, "food_cnn_best.keras"),
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
            ),
        ]
        history2 = model.fit(
            train_ds, validation_data=val_ds, epochs=PHASE2_EPOCHS, callbacks=callbacks_phase2,
            shuffle=False, verbose=2,
        )
        histories.append(("phase2_finetune", history2))

    final_model_path = os.path.join(FOOD_MODEL_DIR, "food_cnn.keras")
    model.save(final_model_path)
    print(f"[train] Final model saved to {final_model_path}")

    plot_training_curves(histories)
    print("[train] Done.")


if __name__ == "__main__":
    run_training()
