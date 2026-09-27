"""
Shared pytest fixtures for the Smart Hospitality Management System test suite.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(scope="session", autouse=True)
def _seed_everything():
    from src.common.utils import set_global_seed

    set_global_seed(42)


@pytest.fixture
def synthetic_booking_df():
    """A small synthetic DataFrame matching the Hotel Booking Demand schema,
    used to unit-test preprocessing logic without needing the real dataset."""
    np.random.seed(0)
    n = 200
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    df = pd.DataFrame({
        "hotel": np.random.choice(["Resort Hotel", "City Hotel"], n),
        "is_canceled": np.random.choice([0, 1], n, p=[0.63, 0.37]),
        "lead_time": np.random.randint(0, 500, n),
        "arrival_date_year": np.random.choice([2015, 2016, 2017], n),
        "arrival_date_month": np.random.choice(months, n),
        "arrival_date_week_number": np.random.randint(1, 53, n),
        "arrival_date_day_of_month": np.random.randint(1, 29, n),
        "stays_in_weekend_nights": np.random.randint(0, 4, n),
        "stays_in_week_nights": np.random.randint(0, 10, n),
        "adults": np.random.randint(1, 4, n),
        "children": np.random.choice([0, 1, 2, np.nan], n, p=[0.85, 0.08, 0.05, 0.02]),
        "babies": np.random.choice([0, 1], n, p=[0.95, 0.05]),
        "meal": np.random.choice(["BB", "HB", "FB", "SC"], n),
        "country": np.random.choice(["PRT", "GBR", "FRA", "ESP", "DEU"], n),
        "market_segment": np.random.choice(["Direct", "Corporate", "Online TA", "Offline TA/TO"], n),
        "distribution_channel": np.random.choice(["Direct", "Corporate", "TA/TO"], n),
        "is_repeated_guest": np.random.choice([0, 1], n, p=[0.9, 0.1]),
        "previous_cancellations": np.random.randint(0, 3, n),
        "previous_bookings_not_canceled": np.random.randint(0, 3, n),
        "reserved_room_type": np.random.choice(list("ABCDE"), n),
        "assigned_room_type": np.random.choice(list("ABCDE"), n),
        "booking_changes": np.random.randint(0, 3, n),
        "deposit_type": np.random.choice(["No Deposit", "Non Refund", "Refundable"], n),
        "agent": np.random.choice([1, 2, 3, np.nan], n),
        "company": np.random.choice([np.nan, 40, 223], n, p=[0.95, 0.03, 0.02]),
        "days_in_waiting_list": np.random.randint(0, 5, n),
        "customer_type": np.random.choice(["Transient", "Contract", "Group"], n),
        "adr": np.random.uniform(0, 300, n),
        "required_car_parking_spaces": np.random.randint(0, 2, n),
        "total_of_special_requests": np.random.randint(0, 3, n),
        "reservation_status": np.random.choice(["Check-Out", "Canceled", "No-Show"], n),
        "reservation_status_date": pd.date_range("2015-01-01", periods=n, freq="h").astype(str),
    })
    return df


@pytest.fixture
def synthetic_reviews_df():
    """A small synthetic DataFrame matching the 515K Hotel Reviews schema."""
    np.random.seed(2)
    n = 150
    pos_phrases = [
        "great location friendly staff", "amazing breakfast and clean room",
        "the bed was very comfortable", "wonderful view of the city",
    ]
    neg_phrases = [
        "the room was very noisy", "breakfast was cold and limited",
        "staff was rude and unhelpful", "bathroom was dirty and small",
    ]
    positive_review, negative_review = [], []
    for _ in range(n):
        positive_review.append(np.random.choice(pos_phrases) if np.random.rand() < 0.9 else "No Positive")
        negative_review.append(np.random.choice(neg_phrases) if np.random.rand() < 0.9 else "No Negative")

    return pd.DataFrame({
        "Hotel_Address": "Some Street, City, Country",
        "Additional_Number_of_Scoring": np.random.randint(1, 500, n),
        "Review_Date": "1/1/2017",
        "Average_Score": np.random.uniform(6, 9, n),
        "Hotel_Name": np.random.choice(["Hotel A", "Hotel B"], n),
        "Reviewer_Nationality": " United Kingdom ",
        "Negative_Review": negative_review,
        "Review_Total_Negative_Word_Counts": np.random.randint(0, 50, n),
        "Total_Number_of_Reviews": np.random.randint(50, 2000, n),
        "Positive_Review": positive_review,
        "Review_Total_Positive_Word_Counts": np.random.randint(0, 50, n),
        "Total_Number_of_Reviews_Reviewer_Has_Given": np.random.randint(1, 20, n),
        "Reviewer_Score": np.random.uniform(2.5, 10, n),
        "Tags": "[' Leisure trip ']",
        "days_since_review": np.random.randint(1, 700, n),
        "lat": 48.2,
        "lng": 16.3,
    })
