"""
Streamlit page — Module 1: Hotel Booking Prediction
========================================================
"""

import os
import sys

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.booking_prediction.predict import artifacts_available, predict_cancellation

st.set_page_config(page_title="Hotel Booking Prediction", page_icon="🏨", layout="wide")
st.title("🏨 Hotel Booking Prediction")
st.caption("ANN (Artificial Neural Network) trained on the Hotel Booking Demand dataset")

if not artifacts_available():
    st.warning(
        "⚠️ Model artifacts not found. Run the following before using this page:\n\n"
        "```bash\npython src/booking_prediction/preprocess.py\n"
        "python src/booking_prediction/train.py\n```"
    )
    st.stop()

st.markdown("Enter the booking details below to predict whether this booking is likely to be **canceled**.")

with st.form("booking_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        hotel = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
        lead_time = st.number_input("Lead Time (days before arrival)", min_value=0, max_value=800, value=45)
        arrival_date_year = st.selectbox("Arrival Year", [2015, 2016, 2017], index=2)
        arrival_date_month = st.selectbox(
            "Arrival Month",
            ["January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December"],
        )
        arrival_date_week_number = st.number_input("Arrival Week Number", min_value=1, max_value=53, value=25)
        arrival_date_day_of_month = st.number_input("Arrival Day of Month", min_value=1, max_value=31, value=15)
        stays_in_weekend_nights = st.number_input("Weekend Nights", min_value=0, max_value=10, value=1)
        stays_in_week_nights = st.number_input("Week Nights", min_value=0, max_value=20, value=2)

    with col2:
        adults = st.number_input("Adults", min_value=1, max_value=10, value=2)
        children = st.number_input("Children", min_value=0, max_value=10, value=0)
        babies = st.number_input("Babies", min_value=0, max_value=5, value=0)
        meal = st.selectbox("Meal Plan", ["BB", "HB", "FB", "SC", "Undefined"])
        country = st.text_input("Country Code (e.g. PRT, GBR, USA)", value="PRT")
        market_segment = st.selectbox(
            "Market Segment",
            ["Direct", "Corporate", "Online TA", "Offline TA/TO", "Complementary", "Groups", "Aviation"],
        )
        distribution_channel = st.selectbox("Distribution Channel", ["Direct", "Corporate", "TA/TO", "GDS"])
        deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])

    with col3:
        customer_type = st.selectbox("Customer Type", ["Transient", "Contract", "Transient-Party", "Group"])
        reserved_room_type = st.selectbox("Reserved Room Type", list("ABCDEFGH"))
        is_repeated_guest = st.selectbox("Repeated Guest?", ["No", "Yes"])
        previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=30, value=0)
        previous_bookings_not_canceled = st.number_input(
            "Previous Bookings Not Canceled", min_value=0, max_value=100, value=0
        )
        booking_changes = st.number_input("Booking Changes", min_value=0, max_value=20, value=0)
        days_in_waiting_list = st.number_input("Days in Waiting List", min_value=0, max_value=400, value=0)
        adr = st.number_input("Average Daily Rate (ADR)", min_value=0.0, max_value=1000.0, value=100.0)
        required_car_parking_spaces = st.number_input("Car Parking Spaces Required", min_value=0, max_value=5, value=0)
        total_of_special_requests = st.number_input("Total Special Requests", min_value=0, max_value=10, value=0)

    submitted = st.form_submit_button("Predict Cancellation Risk", type="primary")

if submitted:
    input_dict = {
        "hotel": hotel,
        "lead_time": lead_time,
        "arrival_date_year": arrival_date_year,
        "arrival_date_month": arrival_date_month,
        "arrival_date_week_number": arrival_date_week_number,
        "arrival_date_day_of_month": arrival_date_day_of_month,
        "stays_in_weekend_nights": stays_in_weekend_nights,
        "stays_in_week_nights": stays_in_week_nights,
        "adults": adults,
        "children": children,
        "babies": babies,
        "meal": meal,
        "country": country.strip().upper() if country.strip() else "Unknown",
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "is_repeated_guest": 1 if is_repeated_guest == "Yes" else 0,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "reserved_room_type": reserved_room_type,
        "booking_changes": booking_changes,
        "deposit_type": deposit_type,
        "days_in_waiting_list": days_in_waiting_list,
        "customer_type": customer_type,
        "adr": adr,
        "required_car_parking_spaces": required_car_parking_spaces,
        "total_of_special_requests": total_of_special_requests,
    }

    try:
        result = predict_cancellation(input_dict)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.divider()
    proba = result["cancellation_probability"]
    label = result["predicted_label"]

    if label == "Canceled":
        st.error(f"⚠️ Predicted outcome: **{label}**")
    else:
        st.success(f"✅ Predicted outcome: **{label}**")

    st.metric("Cancellation Probability", f"{proba:.1%}")
    st.progress(min(max(proba, 0.0), 1.0))
