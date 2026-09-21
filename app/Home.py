"""
Smart Hospitality Management System — Unified Dashboard (Home)
====================================================================
Run:
    streamlit run app/Home.py

This is the entry point. The four modules are separate pages under
app/pages/, which Streamlit's multipage app feature automatically lists
in the sidebar.
"""

import os
import sys

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.booking_prediction.predict import artifacts_available as booking_ready
from src.demand_forecasting.predict import artifacts_available as demand_ready
from src.food_classification.predict import artifacts_available as food_ready
from src.sentiment_analysis.predict import artifacts_available as sentiment_ready

st.set_page_config(
    page_title="Smart Hospitality Management System",
    page_icon="🏨",
    layout="wide",
)

st.title("🏨 Smart Hospitality Management System")
st.caption("A Unified Deep Learning Platform for the Hospitality Industry — Final Year Project")

st.markdown(
    """
    Welcome. This dashboard brings together **four Deep Learning modules** built on
    real, publicly sourced hospitality data, unified under one system rather than
    treated as separate projects. Use the sidebar to navigate between modules.
    """
)

st.divider()

col1, col2, col3, col4 = st.columns(4)

modules = [
    ("🏨 Hotel Booking Prediction", "ANN", booking_ready(), col1,
     "Predicts whether an incoming booking is likely to be canceled, from booking details."),
    ("💬 Review Sentiment Analysis", "BiLSTM", sentiment_ready(), col2,
     "Classifies a guest review as positive or negative sentiment."),
    ("📈 Room Demand Forecasting", "LSTM", demand_ready(), col3,
     "Forecasts upcoming daily room demand from historical booking patterns."),
    ("🍽️ Food Image Classification", "CNN (Transfer Learning)", food_ready(), col4,
     "Classifies an uploaded food photo into a hotel-menu food category."),
]

for title, arch, ready, col, desc in modules:
    with col:
        st.subheader(title)
        st.caption(f"Architecture: {arch}")
        st.write(desc)
        if ready:
            st.success("Model ready ✅")
        else:
            st.warning("Model not trained yet ⚠️")

st.divider()

st.markdown(
    """
    ### About this system
    - **Module 1 & 3** share the same underlying dataset — the *Hotel Booking Demand*
      dataset (Kaggle) — used both for per-booking cancellation prediction and,
      aggregated into a daily time series, for demand forecasting.
    - **Module 2** uses the *515K Hotel Reviews Data in Europe* dataset (Kaggle).
    - **Module 4** uses a documented 12-class subset of *Food-101* relevant to a
      hotel menu / breakfast buffet.

    If a module above shows **"Model not trained yet"**, run that module's
    `preprocess.py` and `train.py` scripts first (see the project README for
    exact commands and dataset download instructions).
    """
)
