"""
Tests — Module 1: Hotel Booking Prediction
===============================================
Unit tests cover preprocessing logic and model architecture using small
synthetic data (no real dataset required). An integration test exercises
the full predict.py inference path but is skipped automatically if the
module hasn't been trained yet.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.booking_prediction.model import build_ann_model
from src.booking_prediction.preprocess import (
    LEAKY_OR_IRRELEVANT_COLUMNS,
    build_feature_matrix,
    clean_data,
    engineer_features,
)


class TestCleanData:
    def test_drops_leaky_columns(self, synthetic_booking_df):
        cleaned = clean_data(synthetic_booking_df)
        for col in LEAKY_OR_IRRELEVANT_COLUMNS:
            assert col not in cleaned.columns

    def test_no_missing_children_or_country(self, synthetic_booking_df):
        cleaned = clean_data(synthetic_booking_df)
        assert cleaned["children"].isnull().sum() == 0
        assert cleaned["country"].isnull().sum() == 0

    def test_removes_zero_guest_rows(self, synthetic_booking_df):
        df = synthetic_booking_df.copy()
        df.loc[0:2, ["adults", "children", "babies"]] = 0
        cleaned = clean_data(df)
        zero_guest_rows = (cleaned["adults"] + cleaned["children"].fillna(0) + cleaned["babies"]) == 0
        assert zero_guest_rows.sum() == 0

    def test_output_has_fewer_or_equal_rows(self, synthetic_booking_df):
        cleaned = clean_data(synthetic_booking_df)
        assert len(cleaned) <= len(synthetic_booking_df)


class TestEngineerFeatures:
    def test_creates_total_nights(self, synthetic_booking_df):
        cleaned = clean_data(synthetic_booking_df)
        engineered = engineer_features(cleaned)
        assert "total_nights" in engineered.columns
        expected = engineered["stays_in_weekend_nights"] + engineered["stays_in_week_nights"]
        assert (engineered["total_nights"] == expected).all()

    def test_creates_total_guests(self, synthetic_booking_df):
        cleaned = clean_data(synthetic_booking_df)
        engineered = engineer_features(cleaned)
        assert "total_guests" in engineered.columns
        assert (engineered["total_guests"] >= 1).all()


class TestBuildFeatureMatrix:
    def test_output_shapes_match(self, synthetic_booking_df):
        cleaned = clean_data(synthetic_booking_df)
        engineered = engineer_features(cleaned)
        X, y, feature_names = build_feature_matrix(engineered)

        assert X.shape[0] == len(engineered)
        assert X.shape[1] == len(feature_names)
        assert len(y) == len(engineered)

    def test_target_is_binary(self, synthetic_booking_df):
        cleaned = clean_data(synthetic_booking_df)
        engineered = engineer_features(cleaned)
        _, y, _ = build_feature_matrix(engineered)
        assert set(np.unique(y)).issubset({0, 1})

    def test_no_categorical_columns_remain(self, synthetic_booking_df):
        cleaned = clean_data(synthetic_booking_df)
        engineered = engineer_features(cleaned)
        X, _, _ = build_feature_matrix(engineered)
        assert X.dtype == np.float64
        assert not np.isnan(X).any()


class TestBuildAnnModel:
    def test_model_compiles_and_has_correct_io_shape(self):
        model = build_ann_model(input_dim=20)
        assert model.input_shape == (None, 20)
        assert model.output_shape == (None, 1)

    def test_model_predicts_valid_probabilities(self):
        model = build_ann_model(input_dim=10)
        X_dummy = np.random.randn(5, 10)
        preds = model.predict(X_dummy, verbose=0)
        assert preds.shape == (5, 1)
        assert (preds >= 0).all() and (preds <= 1).all()


class TestPredictIntegration:
    """Integration test through the real predict.py inference helper.
    Skipped automatically if this module has not been trained yet."""

    def test_predict_cancellation_end_to_end(self):
        from src.booking_prediction.predict import artifacts_available, predict_cancellation

        if not artifacts_available():
            pytest.skip("Module 1 model artifacts not found — train the model first to run this test.")

        result = predict_cancellation({
            "hotel": "City Hotel", "lead_time": 45, "arrival_date_year": 2017,
            "arrival_date_month": "July", "arrival_date_week_number": 25,
            "arrival_date_day_of_month": 15, "stays_in_weekend_nights": 1,
            "stays_in_week_nights": 2, "adults": 2, "children": 0, "babies": 0,
            "meal": "BB", "country": "PRT", "market_segment": "Direct",
            "distribution_channel": "Direct", "is_repeated_guest": 0,
            "previous_cancellations": 0, "previous_bookings_not_canceled": 0,
            "reserved_room_type": "A", "booking_changes": 0, "deposit_type": "No Deposit",
            "days_in_waiting_list": 0, "customer_type": "Transient", "adr": 100.0,
            "required_car_parking_spaces": 0, "total_of_special_requests": 1,
        })

        assert "cancellation_probability" in result
        assert 0.0 <= result["cancellation_probability"] <= 1.0
        assert result["predicted_label"] in {"Canceled", "Not Canceled"}
