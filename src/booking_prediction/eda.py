"""
Exploratory Data Analysis — Module 1: Hotel Booking Prediction
================================================================
Dataset: Hotel Booking Demand (Kaggle: jessemostipak/hotel-booking-demand)
Expected raw file: data/raw/booking_demand/hotel_bookings.csv

Run:
    python src/booking_prediction/eda.py

Produces plots under reports/figures/booking_prediction/ and prints a
text summary of the dataset to the console. This script does NOT train
anything — it is purely for exploration/documentation, matching the
"Perform preprocessing and EDA" requirement of the project brief.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for servers/containers
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.config import BOOKING_RAW_DIR, FIGURES_DIR, ensure_dirs

RAW_CSV = os.path.join(BOOKING_RAW_DIR, "hotel_bookings.csv")
FIG_DIR = os.path.join(FIGURES_DIR, "booking_prediction")


def load_raw_data() -> pd.DataFrame:
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"Could not find {RAW_CSV}.\n"
            "Download the 'Hotel Booking Demand' dataset from Kaggle "
            "(jessemostipak/hotel-booking-demand) and place hotel_bookings.csv "
            "at that path. See README.md section 12 for exact instructions."
        )
    return pd.read_csv(RAW_CSV)


def run_eda(df: pd.DataFrame):
    ensure_dirs(FIG_DIR)

    print("=" * 70)
    print("SHAPE:", df.shape)
    print("=" * 70)
    print("\nCOLUMN DTYPES:\n", df.dtypes)
    print("\nMISSING VALUES (top 15):\n", df.isnull().sum().sort_values(ascending=False).head(15))
    print("\nTARGET DISTRIBUTION (is_canceled):\n", df["is_canceled"].value_counts(normalize=True))

    # 1. Cancellation class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x="is_canceled", data=df, ax=ax)
    ax.set_title("Booking Cancellation Class Balance")
    ax.set_xlabel("is_canceled (0=Not Canceled, 1=Canceled)")
    fig.savefig(os.path.join(FIG_DIR, "01_target_class_balance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Cancellations by hotel type
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="hotel", hue="is_canceled", data=df, ax=ax)
    ax.set_title("Cancellations by Hotel Type")
    fig.savefig(os.path.join(FIG_DIR, "02_cancellations_by_hotel_type.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Lead time distribution vs cancellation
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="is_canceled", y="lead_time", data=df, ax=ax, showfliers=False)
    ax.set_title("Lead Time vs Cancellation")
    fig.savefig(os.path.join(FIG_DIR, "03_lead_time_vs_cancellation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Cancellations by month (seasonality glance — useful context for Module 3 too)
    if "arrival_date_month" in df.columns:
        month_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.countplot(x="arrival_date_month", hue="is_canceled", data=df, order=month_order, ax=ax)
        ax.set_title("Cancellations by Arrival Month")
        plt.xticks(rotation=45)
        fig.savefig(os.path.join(FIG_DIR, "04_cancellations_by_month.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 5. Correlation heatmap (numeric features only)
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (Numeric Features)")
    fig.savefig(os.path.join(FIG_DIR, "05_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[eda] Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    data = load_raw_data()
    run_eda(data)
