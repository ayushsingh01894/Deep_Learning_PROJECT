"""
Exploratory Data Analysis — Module 3: Room Demand Forecasting
=================================================================
Dataset: same raw file as Module 1 (Hotel Booking Demand), but here it is
aggregated by arrival date into a daily bookings/demand time series.

Expected raw file: data/raw/booking_demand/hotel_bookings.csv
(same file used by Module 1 — this is intentional, keeps the system unified)

Run:
    python src/demand_forecasting/eda.py

Produces plots under reports/figures/demand_forecasting/.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.config import BOOKING_RAW_DIR, FIGURES_DIR, ensure_dirs
from src.demand_forecasting.preprocess import build_daily_demand_series

RAW_CSV = os.path.join(BOOKING_RAW_DIR, "hotel_bookings.csv")
FIG_DIR = os.path.join(FIGURES_DIR, "demand_forecasting")


def load_raw_data() -> pd.DataFrame:
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"Could not find {RAW_CSV}.\n"
            "This module reuses the SAME 'Hotel Booking Demand' dataset as Module 1 "
            "(jessemostipak/hotel-booking-demand). Place hotel_bookings.csv at that path. "
            "See README.md section 12."
        )
    return pd.read_csv(RAW_CSV)


def run_eda(df: pd.DataFrame):
    ensure_dirs(FIG_DIR)

    daily = build_daily_demand_series(df)
    print("=" * 70)
    print("DAILY DEMAND SERIES SHAPE:", daily.shape)
    print("=" * 70)
    print(daily.describe())

    # 1. Full time series plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily["arrival_date"], daily["bookings"])
    ax.set_title("Daily Room Bookings Over Time (Both Hotels Combined)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Bookings")
    fig.savefig(os.path.join(FIG_DIR, "01_daily_demand_series.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Weekly seasonality (day of week average)
    daily["day_of_week"] = pd.to_datetime(daily["arrival_date"]).dt.day_name()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_avg = daily.groupby("day_of_week")["bookings"].mean().reindex(dow_order)
    fig, ax = plt.subplots(figsize=(8, 4))
    dow_avg.plot(kind="bar", ax=ax)
    ax.set_title("Average Bookings by Day of Week")
    ax.set_ylabel("Avg Bookings")
    fig.savefig(os.path.join(FIG_DIR, "02_day_of_week_seasonality.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Monthly seasonality
    daily["month"] = pd.to_datetime(daily["arrival_date"]).dt.month_name()
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    month_avg = daily.groupby("month")["bookings"].mean().reindex(month_order)
    fig, ax = plt.subplots(figsize=(10, 4))
    month_avg.plot(kind="bar", ax=ax, color="teal")
    ax.set_title("Average Daily Bookings by Month")
    ax.set_ylabel("Avg Bookings")
    plt.xticks(rotation=45)
    fig.savefig(os.path.join(FIG_DIR, "03_monthly_seasonality.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Rolling 7-day average (smoothed trend)
    daily_sorted = daily.sort_values("arrival_date")
    daily_sorted["rolling_7d"] = daily_sorted["bookings"].rolling(7).mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily_sorted["arrival_date"], daily_sorted["bookings"], alpha=0.3, label="Daily")
    ax.plot(daily_sorted["arrival_date"], daily_sorted["rolling_7d"], color="red", label="7-day rolling avg")
    ax.set_title("Daily Bookings with 7-Day Rolling Average")
    ax.legend()
    fig.savefig(os.path.join(FIG_DIR, "04_rolling_average.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[eda] Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    data = load_raw_data()
    run_eda(data)
