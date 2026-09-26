"""
Exploratory Data Analysis — Module 4: Food Image Classification
====================================================================
Run AFTER preprocess.py has generated the manifests:
    python src/food_classification/eda.py

Produces plots under reports/figures/food_classification/.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.config import FIGURES_DIR, FOOD101_PROCESSED_DIR, ensure_dirs

FIG_DIR = os.path.join(FIGURES_DIR, "food_classification")


def load_manifests():
    train_path = os.path.join(FOOD101_PROCESSED_DIR, "train_manifest.csv")
    val_path = os.path.join(FOOD101_PROCESSED_DIR, "val_manifest.csv")
    test_path = os.path.join(FOOD101_PROCESSED_DIR, "test_manifest.csv")

    for p in (train_path, val_path, test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Could not find {p}. Run src/food_classification/preprocess.py first."
            )

    return (
        pd.read_csv(train_path),
        pd.read_csv(val_path),
        pd.read_csv(test_path),
    )


def run_eda():
    ensure_dirs(FIG_DIR)
    train_df, val_df, test_df = load_manifests()

    print("=" * 70)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("=" * 70)

    # 1. Class distribution (train set)
    fig, ax = plt.subplots(figsize=(10, 5))
    train_df["label"].value_counts().sort_index().plot(kind="bar", ax=ax, color="darkorange")
    ax.set_title("Training Images per Class (Selected Food-101 Subset)")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    fig.savefig(os.path.join(FIG_DIR, "01_class_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Sample images grid (one per class)
    classes = sorted(train_df["label"].unique())
    n_cols = 4
    n_rows = (len(classes) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    for i, cls in enumerate(classes):
        sample_path = train_df[train_df["label"] == cls]["filepath"].iloc[0]
        try:
            img = Image.open(sample_path)
            axes[i].imshow(img)
        except (FileNotFoundError, OSError):
            axes[i].text(0.5, 0.5, "image not found", ha="center")
        axes[i].set_title(cls, fontsize=10)
        axes[i].axis("off")
    for j in range(len(classes), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Sample Image per Selected Class", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "02_sample_images_per_class.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Image dimension check on a small sample (Food-101 images vary in size)
    widths, heights = [], []
    sample_paths = train_df["filepath"].sample(min(200, len(train_df)), random_state=42)
    for p in sample_paths:
        try:
            with Image.open(p) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except (FileNotFoundError, OSError):
            continue

    if widths:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(widths, heights, alpha=0.5)
        ax.set_xlabel("Width (px)")
        ax.set_ylabel("Height (px)")
        ax.set_title(f"Raw Image Dimensions (sample of {len(widths)} images)")
        fig.savefig(os.path.join(FIG_DIR, "03_image_dimensions.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[eda] Sampled image size range: width [{min(widths)}-{max(widths)}], "
              f"height [{min(heights)}-{max(heights)}]")
    else:
        print("[eda] Could not open any sample images to check dimensions "
              "(expected if raw images aren't downloaded yet).")

    print(f"\n[eda] Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    run_eda()
