"""
Inference helper — Module 1: Hotel Booking Prediction
==========================================================
Loads the trained ANN + scaler + encoder column order and exposes a single
`predict_cancellation(input_dict)` function that the Streamlit app (and
tests) can call with raw, human-readable booking fields.

This module deliberately reuses `engineer_features` from preprocess.py so
that inference-time feature engineering can never silently drift from what
the model was actually trained on.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.booking_prediction.preprocess import CATEGORICAL_COLUMNS, engineer_features
from src.common.config import BOOKING_MODEL_DIR

MODEL_PATH = os.path.join(BOOKING_MODEL_DIR, "booking_ann.keras")
SCALER_PATH = os.path.join(BOOKING_MODEL_DIR, "scaler.joblib")
ENCODER_COLUMNS_PATH = os.path.join(BOOKING_MODEL_DIR, "encoder_columns.json")


def artifacts_available() -> bool:
    return all(os.path.exists(p) for p in (MODEL_PATH, SCALER_PATH, ENCODER_COLUMNS_PATH))


def load_artifacts():
    if not artifacts_available():
        raise FileNotFoundError(
            "Module 1 (Booking Prediction) model artifacts not found. "
            "Run preprocess.py and train.py for this module first."
        )
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(ENCODER_COLUMNS_PATH, "r") as f:
        feature_names = json.load(f)
    return model, scaler, feature_names


def _build_feature_row(input_dict: dict, feature_names: list) -> pd.DataFrame:
    """Turn a single raw booking dict into the exact one-hot feature row
    the model expects, using the SAME engineering step used at training time."""
    df = pd.DataFrame([input_dict])
    df = engineer_features(df)

    cat_cols_present = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols_present, drop_first=True)
    df = df.select_dtypes(include=["int64", "float64", "bool"]).astype("float64")

    # Reindex to the exact training-time column order; any dummy column not
    # produced by this single row (e.g. a country not selected) becomes 0,
    # and any unexpected column is dropped. This is the standard, correct
    # way to keep one-hot inference consistent with training.
    df = df.reindex(columns=feature_names, fill_value=0.0)
    return df


def predict_cancellation(input_dict: dict) -> dict:
    """
    input_dict expected keys (raw, human-readable):
        hotel, lead_time, arrival_date_year, arrival_date_month,
        arrival_date_week_number, arrival_date_day_of_month,
        stays_in_weekend_nights, stays_in_week_nights, adults, children, babies,
        meal, country, market_segment, distribution_channel, is_repeated_guest,
        previous_cancellations, previous_bookings_not_canceled, reserved_room_type,
        booking_changes, deposit_type, days_in_waiting_list, customer_type, adr,
        required_car_parking_spaces, total_of_special_requests

    Returns: {"cancellation_probability": float, "predicted_label": "Canceled"/"Not Canceled"}
    """
    model, scaler, feature_names = load_artifacts()

    X_row = _build_feature_row(input_dict, feature_names)
    X_scaled = scaler.transform(X_row.values)

    proba = float(model.predict(X_scaled, verbose=0).ravel()[0])
    label = "Canceled" if proba >= 0.5 else "Not Canceled"

    return {"cancellation_probability": proba, "predicted_label": label}
