"""
Shared configuration for the Smart Hospitality Management System.

Every module (booking_prediction, sentiment_analysis, demand_forecasting,
food_classification) imports paths from here so that the whole project
uses ONE consistent folder layout instead of each module hardcoding its
own paths.
"""

import os

# Project root = two levels up from this file (src/common/config.py -> project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Top-level data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Raw dataset locations (populated by `kaggle datasets download ...`, see README §12)
BOOKING_RAW_DIR = os.path.join(RAW_DATA_DIR, "booking_demand")
REVIEWS_RAW_DIR = os.path.join(RAW_DATA_DIR, "hotel_reviews")
FOOD101_RAW_DIR = os.path.join(RAW_DATA_DIR, "food101")

# Processed dataset locations
BOOKING_PROCESSED_DIR = os.path.join(PROCESSED_DATA_DIR, "booking_demand")
REVIEWS_PROCESSED_DIR = os.path.join(PROCESSED_DATA_DIR, "hotel_reviews")
DEMAND_PROCESSED_DIR = os.path.join(PROCESSED_DATA_DIR, "room_demand")
FOOD101_PROCESSED_DIR = os.path.join(PROCESSED_DATA_DIR, "food101")

# Model artifact locations
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BOOKING_MODEL_DIR = os.path.join(MODELS_DIR, "booking_prediction")
SENTIMENT_MODEL_DIR = os.path.join(MODELS_DIR, "sentiment_analysis")
DEMAND_MODEL_DIR = os.path.join(MODELS_DIR, "demand_forecasting")
FOOD_MODEL_DIR = os.path.join(MODELS_DIR, "food_classification")

# Reports (figures + metrics)
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
METRICS_DIR = os.path.join(REPORTS_DIR, "metrics")

# Reproducibility
RANDOM_SEED = 42


def ensure_dirs(*dirs):
    """Create directories if they don't already exist. Safe to call repeatedly."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def ensure_project_dirs():
    """Create every directory this project expects to write to."""
    ensure_dirs(
        BOOKING_PROCESSED_DIR,
        REVIEWS_PROCESSED_DIR,
        DEMAND_PROCESSED_DIR,
        FOOD101_PROCESSED_DIR,
        BOOKING_MODEL_DIR,
        SENTIMENT_MODEL_DIR,
        DEMAND_MODEL_DIR,
        FOOD_MODEL_DIR,
        FIGURES_DIR,
        METRICS_DIR,
    )
