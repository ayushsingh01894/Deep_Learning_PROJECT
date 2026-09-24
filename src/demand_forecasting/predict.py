"""
Inference helper — Module 3: Room Demand Forecasting
=========================================================
Loads the trained LSTM + scaler and exposes:
    - load_daily_demand_history()       -> the processed daily series (for charting)
    - forecast_next_n_days(n)           -> recursively forecasts n future days
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.config import DEMAND_MODEL_DIR, DEMAND_PROCESSED_DIR

MODEL_PATH = os.path.join(DEMAND_MODEL_DIR, "demand_lstm.keras")
SCALER_PATH = os.path.join(DEMAND_MODEL_DIR, "scaler.joblib")
WINDOW_CONFIG_PATH = os.path.join(DEMAND_MODEL_DIR, "window_config.json")
DAILY_DEMAND_CSV = os.path.join(DEMAND_PROCESSED_DIR, "daily_demand.csv")


def artifacts_available() -> bool:
    return all(
        os.path.exists(p) for p in (MODEL_PATH, SCALER_PATH, WINDOW_CONFIG_PATH, DAILY_DEMAND_CSV)
    )


def load_daily_demand_history() -> pd.DataFrame:
    if not os.path.exists(DAILY_DEMAND_CSV):
        raise FileNotFoundError(
            f"Could not find {DAILY_DEMAND_CSV}. Run src/demand_forecasting/preprocess.py first."
        )
    df = pd.read_csv(DAILY_DEMAND_CSV, parse_dates=["arrival_date"])
    return df


def load_artifacts():
    if not artifacts_available():
        raise FileNotFoundError(
            "Module 3 (Demand Forecasting) model artifacts not found. "
            "Run preprocess.py and train.py for this module first."
        )
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(WINDOW_CONFIG_PATH, "r") as f:
        window_size = json.load(f)["window_size"]
    return model, scaler, window_size


def forecast_next_n_days(n_days: int = 7) -> pd.DataFrame:
    """Recursively forecast the next `n_days` of room demand, feeding each
    prediction back in as input for the following day (standard
    multi-step-ahead LSTM forecasting approach).

    Returns a DataFrame with columns: date, predicted_bookings
    """
    model, scaler, window_size = load_artifacts()
    history = load_daily_demand_history()

    last_window = history["bookings"].values[-window_size:].astype("float64")
    last_window_scaled = scaler.transform(last_window.reshape(-1, 1)).ravel()

    predictions_scaled = []
    current_window = last_window_scaled.copy()

    for _ in range(n_days):
        model_input = current_window.reshape((1, window_size, 1))
        next_scaled = model.predict(model_input, verbose=0).ravel()[0]
        predictions_scaled.append(next_scaled)
        current_window = np.append(current_window[1:], next_scaled)

    predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).ravel()
    predictions = np.clip(predictions, a_min=0, a_max=None)

    last_date = history["arrival_date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_days, freq="D")

    return pd.DataFrame({"date": future_dates, "predicted_bookings": predictions})
