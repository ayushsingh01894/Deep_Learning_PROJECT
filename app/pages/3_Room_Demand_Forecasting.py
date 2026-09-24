"""
Streamlit page — Module 3: Room Demand Forecasting
========================================================
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.demand_forecasting.predict import artifacts_available, forecast_next_n_days, load_daily_demand_history

st.set_page_config(page_title="Room Demand Forecasting", page_icon="📈", layout="wide")
st.title("📈 Room Demand Forecasting")
st.caption("LSTM trained on daily-aggregated bookings from the Hotel Booking Demand dataset")

if not artifacts_available():
    st.warning(
        "⚠️ Model artifacts not found. Run the following before using this page:\n\n"
        "```bash\npython src/demand_forecasting/preprocess.py\n"
        "python src/demand_forecasting/train.py\n```"
    )
    st.stop()

history = load_daily_demand_history()

st.markdown("### Historical Daily Room Demand")
st.line_chart(history.set_index("arrival_date")["bookings"])

st.divider()

st.markdown("### Forecast Future Demand")
n_days = st.slider("Number of days to forecast", min_value=1, max_value=30, value=7)

if st.button("Generate Forecast", type="primary"):
    try:
        forecast_df = forecast_next_n_days(n_days)
    except Exception as e:
        st.error(f"Forecasting failed: {e}")
        st.stop()

    st.divider()

    # Combine the tail of history with the forecast for a continuous chart
    recent_history = history.tail(30)[["arrival_date", "bookings"]].rename(
        columns={"arrival_date": "date", "bookings": "value"}
    )
    recent_history["type"] = "Actual"
    forecast_display = forecast_df.rename(columns={"predicted_bookings": "value"})
    forecast_display["type"] = "Forecast"

    combined = pd.concat([recent_history, forecast_display], ignore_index=True)
    chart_df = combined.pivot(index="date", columns="type", values="value")
    st.line_chart(chart_df)

    st.markdown("#### Forecasted values")
    st.dataframe(
        forecast_df.rename(columns={"date": "Date", "predicted_bookings": "Predicted Bookings"}),
        use_container_width=True,
        hide_index=True,
    )

    avg_forecast = forecast_df["predicted_bookings"].mean()
    st.metric(f"Average predicted daily demand (next {n_days} days)", f"{avg_forecast:.1f} bookings")
