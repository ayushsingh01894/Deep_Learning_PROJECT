"""
Evaluation script — Module 3: Room Demand Forecasting
==========================================================
Run:
    python src/demand_forecasting/evaluate.py

Requires a trained model at models/demand_forecasting/demand_lstm.keras
and processed test windows (from preprocess.py).

Saves:
    reports/metrics/demand_forecasting_metrics.json
    reports/figures/demand_forecasting/actual_vs_predicted.png

IMPORTANT: All metrics are computed from the actual trained model's
predictions on the real held-out (chronologically final) test window.
No numbers are hardcoded.
"""

import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from tensorflow import keras

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.config import DEMAND_MODEL_DIR, DEMAND_PROCESSED_DIR, FIGURES_DIR, METRICS_DIR, ensure_dirs
from src.common.utils import save_metrics_json

FIG_DIR = os.path.join(FIGURES_DIR, "demand_forecasting")
MODEL_PATH = os.path.join(DEMAND_MODEL_DIR, "demand_lstm.keras")
SCALER_PATH = os.path.join(DEMAND_MODEL_DIR, "scaler.joblib")
METRICS_PATH = os.path.join(METRICS_DIR, "demand_forecasting_metrics.json")


def load_test_data():
    X_test = np.load(os.path.join(DEMAND_PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DEMAND_PROCESSED_DIR, "y_test.npy"))
    return X_test, y_test


def plot_actual_vs_predicted(y_true, y_pred):
    ensure_dirs(FIG_DIR)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_true, label="Actual demand", marker="o", markersize=3)
    ax.plot(y_pred, label="Predicted demand", marker="x", markersize=3)
    ax.set_title("Room Demand Forecasting — Actual vs Predicted (Test Set)")
    ax.set_xlabel("Day index (test period)")
    ax.set_ylabel("Number of bookings")
    ax.legend()
    fig.savefig(os.path.join(FIG_DIR, "actual_vs_predicted.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_evaluation():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run train.py first."
        )

    X_test, y_test_scaled = load_test_data()
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    y_pred_scaled = model.predict(X_test, verbose=0).ravel()

    # Invert scaling to get back to real "number of bookings" units
    y_test = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_pred = np.clip(y_pred, a_min=0, a_max=None)  # bookings can't be negative

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    # MAPE can blow up / divide-by-zero on days with 0 actual bookings; guard for that
    nonzero_mask = y_test != 0
    mape = (
        float(mean_absolute_percentage_error(y_test[nonzero_mask], y_pred[nonzero_mask]))
        if nonzero_mask.any()
        else None
    )

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "mape_excluding_zero_actual_days": mape,
        "test_set_size_days": int(len(y_test)),
        "mean_actual_demand": float(y_test.mean()),
    }

    print("[evaluate] Test set metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    plot_actual_vs_predicted(y_test, y_pred)
    save_metrics_json(metrics, METRICS_PATH)

    print("[evaluate] Done.")
    return metrics


if __name__ == "__main__":
    run_evaluation()
