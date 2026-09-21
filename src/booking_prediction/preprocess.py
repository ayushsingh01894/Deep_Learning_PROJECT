"""
Preprocessing pipeline — Module 1: Hotel Booking Prediction
=============================================================
Dataset: Hotel Booking Demand (Kaggle: jessemostipak/hotel-booking-demand)
Expected raw file: data/raw/booking_demand/hotel_bookings.csv

Run:
    python src/booking_prediction/preprocess.py

Produces (under data/processed/booking_demand/):
    - X_train.npy, X_val.npy, X_test.npy
    - y_train.npy, y_val.npy, y_test.npy
    - feature_names.json
Produces (under models/booking_prediction/):
    - scaler.joblib          (StandardScaler fit on numeric features)
    - encoder_columns.json   (final one-hot column order, needed at inference time)
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.config import (
    BOOKING_MODEL_DIR,
    BOOKING_PROCESSED_DIR,
    BOOKING_RAW_DIR,
    RANDOM_SEED,
    ensure_dirs,
)
from src.common.utils import set_global_seed

RAW_CSV = os.path.join(BOOKING_RAW_DIR, "hotel_bookings.csv")

# Columns known (from the dataset's documentation) to leak the cancellation
# outcome or to be pure booking-management metadata unrelated to the guest's
# actual cancellation decision. Dropped before modeling.
LEAKY_OR_IRRELEVANT_COLUMNS = [
    "reservation_status",       # directly encodes the outcome (Canceled/Check-Out/No-Show)
    "reservation_status_date",  # date of that outcome
    "assigned_room_type",       # only known after check-in / near-arrival
    "company",                  # >90% missing, high-cardinality ID, low signal
    "agent",                    # very high cardinality booking-agent ID
]

CATEGORICAL_COLUMNS = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
]

TARGET_COLUMN = "is_canceled"


def load_raw_data() -> pd.DataFrame:
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"Could not find {RAW_CSV}.\n"
            "Download the 'Hotel Booking Demand' dataset from Kaggle "
            "(jessemostipak/hotel-booking-demand) and place hotel_bookings.csv "
            "at that path. See README.md section 12 for exact instructions."
        )
    return pd.read_csv(RAW_CSV)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop leakage / irrelevant columns if present
    df = df.drop(columns=[c for c in LEAKY_OR_IRRELEVANT_COLUMNS if c in df.columns])

    # Handle missing values
    if "children" in df.columns:
        df["children"] = df["children"].fillna(0)
    if "country" in df.columns:
        df["country"] = df["country"].fillna("Unknown")

    # A well-known data-quality issue in this dataset: rows with zero
    # adults, children, and babies (i.e. no actual guests). Drop them.
    if {"adults", "children", "babies"}.issubset(df.columns):
        df = df[(df["adults"] + df["children"] + df["babies"]) > 0]

    # Cap extreme outliers in average daily rate (documented data-entry error: adr < 0 or huge spike)
    if "adr" in df.columns:
        df = df[(df["adr"] >= 0) & (df["adr"] < 5000)]

    df = df.reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Total nights stayed (combines weekend + week nights into one useful signal)
    if {"stays_in_weekend_nights", "stays_in_week_nights"}.issubset(df.columns):
        df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]

    # Total guests
    if {"adults", "children", "babies"}.issubset(df.columns):
        df["total_guests"] = df["adults"] + df["children"] + df["babies"]

    # Whether the guest requested a room different from the reserved type
    # is not usable (assigned_room_type dropped as leakage), so skip that.

    return df


def build_feature_matrix(df: pd.DataFrame):
    df = df.copy()
    y = df[TARGET_COLUMN].astype(int).values
    X_df = df.drop(columns=[TARGET_COLUMN])

    # One-hot encode categoricals that are present
    cat_cols_present = [c for c in CATEGORICAL_COLUMNS if c in X_df.columns]
    X_df = pd.get_dummies(X_df, columns=cat_cols_present, drop_first=True)

    # Keep only numeric columns from here on (safety net against stray object dtypes)
    X_df = X_df.select_dtypes(include=["int64", "float64", "bool"])
    X_df = X_df.astype("float64")

    feature_names = X_df.columns.tolist()
    return X_df.values, y, feature_names


def run_preprocessing():
    set_global_seed(RANDOM_SEED)
    ensure_dirs(BOOKING_PROCESSED_DIR, BOOKING_MODEL_DIR)

    print("[preprocess] Loading raw data...")
    df = load_raw_data()
    print(f"[preprocess] Raw shape: {df.shape}")

    df = clean_data(df)
    print(f"[preprocess] Shape after cleaning: {df.shape}")

    df = engineer_features(df)

    X, y, feature_names = build_feature_matrix(df)
    print(f"[preprocess] Feature matrix shape: {X.shape} | #features: {len(feature_names)}")

    # 70/15/15 train/val/test split, stratified on the target
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )

    # Scale numeric features (fit ONLY on train, as is best practice)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Persist processed arrays
    np.save(os.path.join(BOOKING_PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(BOOKING_PROCESSED_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(BOOKING_PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(BOOKING_PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(BOOKING_PROCESSED_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(BOOKING_PROCESSED_DIR, "y_test.npy"), y_test)

    with open(os.path.join(BOOKING_PROCESSED_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    # Persist scaler + the exact one-hot column order (both required for consistent
    # inference later in the Streamlit app)
    joblib.dump(scaler, os.path.join(BOOKING_MODEL_DIR, "scaler.joblib"))
    with open(os.path.join(BOOKING_MODEL_DIR, "encoder_columns.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    print("[preprocess] Done.")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Saved processed data to {BOOKING_PROCESSED_DIR}")
    print(f"  Saved scaler + encoder columns to {BOOKING_MODEL_DIR}")


if __name__ == "__main__":
    run_preprocessing()
