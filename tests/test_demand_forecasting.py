"""
Tests — Module 3: Room Demand Forecasting
==============================================
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.demand_forecasting.model import build_lstm_model
from src.demand_forecasting.preprocess import build_daily_demand_series, create_windows


@pytest.fixture
def synthetic_dated_booking_df():
    """A synthetic booking DataFrame with real calendar dates, spanning
    enough days to build a meaningful demand series for testing."""
    np.random.seed(1)
    n = 1500
    dates = pd.date_range("2015-07-01", periods=365, freq="D")
    sampled_dates = pd.to_datetime(np.random.choice(dates, n))
    return pd.DataFrame({
        "is_canceled": np.random.choice([0, 1], n, p=[0.63, 0.37]),
        "arrival_date_year": sampled_dates.year,
        "arrival_date_month": sampled_dates.month_name(),
        "arrival_date_day_of_month": sampled_dates.day,
    })


class TestBuildDailyDemandSeries:
    def test_returns_expected_columns(self, synthetic_dated_booking_df):
        daily = build_daily_demand_series(synthetic_dated_booking_df)
        assert list(daily.columns) == ["arrival_date", "bookings"]

    def test_covers_full_calendar_range_with_no_gaps(self, synthetic_dated_booking_df):
        daily = build_daily_demand_series(synthetic_dated_booking_df)
        expected_days = (daily["arrival_date"].max() - daily["arrival_date"].min()).days + 1
        assert len(daily) == expected_days

    def test_bookings_are_non_negative_integers(self, synthetic_dated_booking_df):
        daily = build_daily_demand_series(synthetic_dated_booking_df)
        assert (daily["bookings"] >= 0).all()

    def test_excludes_canceled_bookings(self, synthetic_dated_booking_df):
        all_canceled = synthetic_dated_booking_df.copy()
        all_canceled["is_canceled"] = 1
        daily = build_daily_demand_series(all_canceled)
        assert (daily["bookings"] == 0).all()


class TestCreateWindows:
    def test_output_shapes(self):
        series = np.arange(20, dtype="float64")
        X, y = create_windows(series, window_size=5)
        assert X.shape == (15, 5)
        assert y.shape == (15,)

    def test_windows_align_correctly_with_targets(self):
        series = np.arange(10, dtype="float64")
        X, y = create_windows(series, window_size=3)
        assert list(X[0]) == [0.0, 1.0, 2.0]
        assert y[0] == 3.0
        assert list(X[-1]) == [6.0, 7.0, 8.0]
        assert y[-1] == 9.0


class TestBuildLstmModel:
    def test_model_compiles_and_has_correct_io_shape(self):
        model = build_lstm_model(window_size=14)
        assert model.input_shape == (None, 14, 1)
        assert model.output_shape == (None, 1)

    def test_model_predicts_finite_values(self):
        model = build_lstm_model(window_size=7)
        X_dummy = np.random.randn(4, 7, 1)
        preds = model.predict(X_dummy, verbose=0)
        assert preds.shape == (4, 1)
        assert np.isfinite(preds).all()


class TestPredictIntegration:
    def test_forecast_next_n_days_end_to_end(self):
        from src.demand_forecasting.predict import artifacts_available, forecast_next_n_days

        if not artifacts_available():
            pytest.skip("Module 3 model artifacts not found — train the model first to run this test.")

        forecast_df = forecast_next_n_days(n_days=5)

        assert len(forecast_df) == 5
        assert list(forecast_df.columns) == ["date", "predicted_bookings"]
        assert (forecast_df["predicted_bookings"] >= 0).all()
