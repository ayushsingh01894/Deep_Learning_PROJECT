"""
Preprocessing pipeline — Module 3: Room Demand Forecasting
==============================================================
Reuses the SAME raw file as Module 1 (Hotel Booking Demand dataset), but
aggregates individual booking records into a DAILY DEMAND TIME SERIES,
which is then windowed into sequences suitable for an LSTM.

Expected raw file: data/raw/booking_demand/hotel_bookings.csv

Run:
    python src/demand_forecasting/preprocess.py

Produces (under data/processed/room_demand/):
    - daily_demand.csv        (the aggregated time series itself, human-readable)
    - X_train.npy, X_val.npy, X_test.npy   (windowed sequences)
    - y_train.npy, y_val.npy, y_test.npy   (next-day demand targets)
Produces (under models/demand_forecasting/):
    - scaler.joblib            (MinMaxScaler fit on the training portion of the series)
    - window_config.json       (window size, used identically at inference time)

Design notes
------------
- "Demand" is defined as the count of bookings that were NOT canceled,
  grouped by arrival_date. This reflects actual realized room demand
  rather than raw booking requests (which would include bookings that
  never materialized because they were canceled).
- The split is chronological (not random) because this is a time series:
  train = earliest 70% of days, val = next 15%, test = final 15%.
  This avoids leaking future information into training, which a random
  shuffle-split would do.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.config import (
    BOOKING_RAW_DIR,
    DEMAND_MODEL_DIR,
    DEMAND_PROCESSED_DIR,
    ensure_dirs,
)

RAW_CSV = os.path.join(BOOKING_RAW_DIR, "hotel_bookings.csv")

WINDOW_SIZE = 14  # use the previous 14 days of demand to predict the next day


def load_raw_data() -> pd.DataFrame:
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"Could not find {RAW_CSV}.\n"
            "This module reuses the SAME 'Hotel Booking Demand' dataset as Module 1 "
            "(jessemostipak/hotel-booking-demand). Place hotel_bookings.csv at that path. "
            "See README.md section 12."
        )
    return pd.read_csv(RAW_CSV)


def build_daily_demand_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw booking records into a daily 'realized demand' time series.

    Returns a DataFrame with columns: arrival_date (datetime), bookings (int)
    covering every calendar day in the dataset's date range (missing days
    filled with 0 bookings).
    """
    df = df.copy()

    # Reconstruct a proper date from the three separate arrival_date_* columns
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str)
        + "-"
        + df["arrival_date_month"]
        + "-"
        + df["arrival_date_day_of_month"].astype(str),
        format="%Y-%B-%d",
        errors="coerce",
    )
    df = df.dropna(subset=["arrival_date"])

    # Determine the calendar range from ALL bookings (not just realized ones),
    # so the series still covers the correct date span even in the edge case
    # where every booking in range happens to be canceled (this would otherwise
    # crash: an empty `realized` set has no min/max date to build a range from).
    full_range = pd.date_range(df["arrival_date"].min(), df["arrival_date"].max(), freq="D")

    # Realized demand = bookings that were NOT canceled (i.e. actual arrivals)
    realized = df[df["is_canceled"] == 0]

    daily = realized.groupby("arrival_date").size().reset_index(name="bookings")
    daily = daily.set_index("arrival_date").reindex(full_range, fill_value=0)
    daily.index.name = "arrival_date"
    daily = daily.reset_index()

    return daily


def create_windows(series: np.ndarray, window_size: int):
    """Turn a 1D array into (X, y) supervised-learning windows for an LSTM.

    X[i] = series[i : i+window_size]         shape (window_size,)
    y[i] = series[i+window_size]             the next value to predict
    """
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)


def run_preprocessing():
    ensure_dirs(DEMAND_PROCESSED_DIR, DEMAND_MODEL_DIR)

    print("[preprocess] Loading raw data...")
    df = load_raw_data()

    daily = build_daily_demand_series(df)
    print(f"[preprocess] Daily demand series length: {len(daily)} days "
          f"({daily['arrival_date'].min().date()} to {daily['arrival_date'].max().date()})")

    daily.to_csv(os.path.join(DEMAND_PROCESSED_DIR, "daily_demand.csv"), index=False)

    values = daily["bookings"].values.astype("float64")

    # Chronological split: 70% train / 15% val / 15% test
    n = len(values)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_series = values[:train_end]
    val_series = values[train_end:val_end]
    test_series = values[val_end:]

    print(f"[preprocess] Chronological split -> train: {len(train_series)} days, "
          f"val: {len(val_series)} days, test: {len(test_series)} days")

    # Scale using ONLY the training portion, to avoid leakage
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_series.reshape(-1, 1)).ravel()
    val_scaled = scaler.transform(val_series.reshape(-1, 1)).ravel()
    test_scaled = scaler.transform(test_series.reshape(-1, 1)).ravel()

    # To build valid windows at the val/test boundaries, prepend the tail of
    # the previous split so windows can "see" the days right before the split.
    val_scaled_ext = np.concatenate([train_scaled[-WINDOW_SIZE:], val_scaled])
    test_scaled_ext = np.concatenate([val_scaled[-WINDOW_SIZE:], test_scaled])

    X_train, y_train = create_windows(train_scaled, WINDOW_SIZE)
    X_val, y_val = create_windows(val_scaled_ext, WINDOW_SIZE)
    X_test, y_test = create_windows(test_scaled_ext, WINDOW_SIZE)

    # Reshape for LSTM: (samples, timesteps, features=1)
    X_train = X_train.reshape((-1, WINDOW_SIZE, 1))
    X_val = X_val.reshape((-1, WINDOW_SIZE, 1))
    X_test = X_test.reshape((-1, WINDOW_SIZE, 1))

    np.save(os.path.join(DEMAND_PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DEMAND_PROCESSED_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(DEMAND_PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DEMAND_PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DEMAND_PROCESSED_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(DEMAND_PROCESSED_DIR, "y_test.npy"), y_test)

    joblib.dump(scaler, os.path.join(DEMAND_MODEL_DIR, "scaler.joblib"))
    with open(os.path.join(DEMAND_MODEL_DIR, "window_config.json"), "w") as f:
        json.dump({"window_size": WINDOW_SIZE}, f, indent=2)

    print(f"[preprocess] Window shapes -> X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print(f"[preprocess] Saved processed data to {DEMAND_PROCESSED_DIR}")
    print(f"[preprocess] Saved scaler + window config to {DEMAND_MODEL_DIR}")


if __name__ == "__main__":
    run_preprocessing()
