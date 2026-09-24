"""
Preprocessing pipeline — Module 4: Food Image Classification
=================================================================
Dataset: Food-101 (Kaggle: kmader/food41, mirrors the official ETH Zurich set)

Expected raw layout (standard Food-101 / kmader/food41 structure):
    data/raw/food101/images/<class_name>/<image_id>.jpg
    data/raw/food101/meta/classes.txt
    data/raw/food101/meta/train.txt   (official train split, "class/image_id" per line, no extension)
    data/raw/food101/meta/test.txt    (official test split)

Run:
    python src/food_classification/preprocess.py

Class subset
------------
The full Food-101 set (101 classes, 101,000 images) is large to train on
typical student hardware. We train on a documented SUBSET of 12 classes
relevant to a hotel menu / breakfast buffet context (see SELECTED_CLASSES
below). This is an explicit, documented design decision, not a hidden
shortcut, and matches what is stated in the project README.

We use Food-101's OFFICIAL train/test split (from meta/train.txt and
meta/test.txt) rather than a random split, so that results stay comparable
to published Food-101 benchmarks. The official train split is further
split into train/val (85/15) for model selection.

Produces (under data/processed/food101/):
    - train_manifest.csv   (filepath, label, label_id)
    - val_manifest.csv
    - test_manifest.csv
Produces (under models/food_classification/):
    - class_names.json     (ordered list of the 12 selected class names)
"""

import json
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.config import FOOD101_RAW_DIR, FOOD101_PROCESSED_DIR, FOOD_MODEL_DIR, RANDOM_SEED, ensure_dirs

IMAGES_DIR = os.path.join(FOOD101_RAW_DIR, "images")
META_DIR = os.path.join(FOOD101_RAW_DIR, "meta")
TRAIN_SPLIT_FILE = os.path.join(META_DIR, "train.txt")
TEST_SPLIT_FILE = os.path.join(META_DIR, "test.txt")

# Documented subset: 12 food categories relevant to a hotel menu / breakfast buffet.
# Must match Food-101's exact lowercase-underscore class folder names.
SELECTED_CLASSES = [
    "waffles",
    "pancakes",
    "omelette",
    "french_toast",
    "pizza",
    "sushi",
    "hamburger",
    "caesar_salad",
    "steak",
    "ice_cream",
    "donuts",
    "club_sandwich",
]


def check_raw_data_exists():
    missing = []
    if not os.path.isdir(IMAGES_DIR):
        missing.append(IMAGES_DIR)
    if not os.path.exists(TRAIN_SPLIT_FILE):
        missing.append(TRAIN_SPLIT_FILE)
    if not os.path.exists(TEST_SPLIT_FILE):
        missing.append(TEST_SPLIT_FILE)
    if missing:
        raise FileNotFoundError(
            "Food-101 raw data not found at expected paths:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nDownload the Food-101 dataset from Kaggle (kmader/food41) and place it so "
            "that data/raw/food101/images/<class>/*.jpg and data/raw/food101/meta/{train,test}.txt "
            "exist. See README.md section 12."
        )


def read_split_file(path: str) -> pd.DataFrame:
    """Read a Food-101 meta split file (train.txt or test.txt).
    Each line looks like: 'waffles/1234567' (no file extension)."""
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    records = []
    for line in lines:
        cls = line.split("/")[0]
        if cls in SELECTED_CLASSES:
            filepath = os.path.join(IMAGES_DIR, line + ".jpg")
            records.append({"filepath": filepath, "label": cls})

    return pd.DataFrame(records)


def run_preprocessing():
    ensure_dirs(FOOD101_PROCESSED_DIR, FOOD_MODEL_DIR)
    check_raw_data_exists()

    print(f"[preprocess] Using {len(SELECTED_CLASSES)} selected classes: {SELECTED_CLASSES}")

    train_full = read_split_file(TRAIN_SPLIT_FILE)
    test_df = read_split_file(TEST_SPLIT_FILE)

    print(f"[preprocess] Official train split (selected classes only): {len(train_full)} images")
    print(f"[preprocess] Official test split (selected classes only): {len(test_df)} images")

    if len(train_full) == 0 or len(test_df) == 0:
        raise RuntimeError(
            "No images found for the selected classes. Check that SELECTED_CLASSES "
            "match the actual folder names under data/raw/food101/images/."
        )

    # Split the official train set into train/val (85/15), stratified by class
    train_df, val_df = train_test_split(
        train_full, test_size=0.15, random_state=RANDOM_SEED, stratify=train_full["label"]
    )

    # Build a stable label_id mapping (alphabetical, reproducible)
    class_names = sorted(SELECTED_CLASSES)
    label_to_id = {c: i for i, c in enumerate(class_names)}

    for df in (train_df, val_df, test_df):
        df["label_id"] = df["label"].map(label_to_id)

    train_df.to_csv(os.path.join(FOOD101_PROCESSED_DIR, "train_manifest.csv"), index=False)
    val_df.to_csv(os.path.join(FOOD101_PROCESSED_DIR, "val_manifest.csv"), index=False)
    test_df.to_csv(os.path.join(FOOD101_PROCESSED_DIR, "test_manifest.csv"), index=False)

    with open(os.path.join(FOOD_MODEL_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    print("[preprocess] Done.")
    print(f"  Train: {len(train_df)} images, Val: {len(val_df)} images, Test: {len(test_df)} images")
    print(f"  Saved manifests to {FOOD101_PROCESSED_DIR}")
    print(f"  Saved class_names.json to {FOOD_MODEL_DIR}")


if __name__ == "__main__":
    run_preprocessing()
