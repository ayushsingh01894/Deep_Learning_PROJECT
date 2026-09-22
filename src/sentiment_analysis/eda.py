"""
Exploratory Data Analysis — Module 2: Hotel Review Sentiment Analysis
=========================================================================
Dataset: 515K Hotel Reviews Data in Europe
(Kaggle: jiashenliu/515k-hotel-reviews-data-in-europe)

Expected raw file: data/raw/hotel_reviews/Hotel_Reviews.csv

Run:
    python src/sentiment_analysis/eda.py

Produces plots under reports/figures/sentiment_analysis/.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.config import FIGURES_DIR, REVIEWS_RAW_DIR, ensure_dirs

RAW_CSV = os.path.join(REVIEWS_RAW_DIR, "Hotel_Reviews.csv")
FIG_DIR = os.path.join(FIGURES_DIR, "sentiment_analysis")


def load_raw_data() -> pd.DataFrame:
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"Could not find {RAW_CSV}.\n"
            "Download the '515K Hotel Reviews Data in Europe' dataset from Kaggle "
            "(jiashenliu/515k-hotel-reviews-data-in-europe) and place Hotel_Reviews.csv "
            "at that path. See README.md section 12."
        )
    return pd.read_csv(RAW_CSV)


def run_eda(df: pd.DataFrame):
    ensure_dirs(FIG_DIR)

    print("=" * 70)
    print("SHAPE:", df.shape)
    print("=" * 70)
    print("\nCOLUMNS:\n", df.columns.tolist())
    print("\nMISSING VALUES (top 10):\n", df.isnull().sum().sort_values(ascending=False).head(10))

    # 1. Reviewer score distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["Reviewer_Score"], bins=20, ax=ax)
    ax.set_title("Distribution of Reviewer Scores")
    ax.set_xlabel("Reviewer Score (0-10)")
    fig.savefig(os.path.join(FIG_DIR, "01_reviewer_score_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Review word count distributions (positive vs negative)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["Review_Total_Positive_Word_Counts"], bins=40, ax=axes[0], color="green")
    axes[0].set_title("Positive Review Word Count")
    axes[0].set_xlim(0, 200)
    sns.histplot(df["Review_Total_Negative_Word_Counts"], bins=40, ax=axes[1], color="red")
    axes[1].set_title("Negative Review Word Count")
    axes[1].set_xlim(0, 200)
    fig.savefig(os.path.join(FIG_DIR, "02_review_word_counts.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. How many reviews actually have empty positive/negative text
    empty_pos = (df["Positive_Review"].str.strip().str.lower() == "no positive").sum()
    empty_neg = (df["Negative_Review"].str.strip().str.lower() == "no negative").sum()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["Empty Positive\n('No Positive')", "Empty Negative\n('No Negative')"],
        [empty_pos, empty_neg],
        color=["salmon", "steelblue"],
    )
    ax.set_title("Count of Placeholder ('No Positive'/'No Negative') Reviews")
    ax.set_ylabel("Count")
    fig.savefig(os.path.join(FIG_DIR, "03_empty_review_placeholders.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n'No Positive' placeholder count: {empty_pos}")
    print(f"'No Negative' placeholder count: {empty_neg}")

    # 4. Top reviewer nationalities
    if "Reviewer_Nationality" in df.columns:
        top_nat = df["Reviewer_Nationality"].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(8, 5))
        top_nat.plot(kind="barh", ax=ax)
        ax.set_title("Top 15 Reviewer Nationalities")
        ax.invert_yaxis()
        fig.savefig(os.path.join(FIG_DIR, "04_top_reviewer_nationalities.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\n[eda] Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    data = load_raw_data()
    run_eda(data)
