"""
Streamlit page — Module 2: Hotel Review Sentiment Analysis
================================================================
"""

import os
import sys

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.sentiment_analysis.predict import artifacts_available, predict_sentiment

st.set_page_config(page_title="Hotel Review Sentiment Analysis", page_icon="💬", layout="wide")
st.title("💬 Hotel Review Sentiment Analysis")
st.caption("Bidirectional LSTM trained on the 515K Hotel Reviews Data in Europe dataset")

if not artifacts_available():
    st.warning(
        "⚠️ Model artifacts not found. Run the following before using this page:\n\n"
        "```bash\npython src/sentiment_analysis/preprocess.py\n"
        "python src/sentiment_analysis/train.py\n```"
    )
    st.stop()

st.markdown("Paste a guest review below to classify its sentiment.")

example_reviews = {
    "— choose an example —": "",
    "Positive example": "The staff were incredibly friendly and the room had a beautiful view of the city.",
    "Negative example": "The room was dirty and the air conditioning did not work at all during our stay.",
}
example_choice = st.selectbox("Or try an example:", list(example_reviews.keys()))

review_text = st.text_area(
    "Guest review text",
    value=example_reviews[example_choice],
    height=150,
    placeholder="e.g. The breakfast was excellent and the staff went above and beyond...",
)

if st.button("Analyze Sentiment", type="primary"):
    if not review_text.strip():
        st.error("Please enter some review text first.")
        st.stop()

    try:
        result = predict_sentiment(review_text)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.divider()
    sentiment = result["sentiment"]
    confidence = result["confidence"]

    if sentiment == "Positive":
        st.success(f"😊 Predicted sentiment: **{sentiment}**")
        display_confidence = confidence
    else:
        st.error(f"☹️ Predicted sentiment: **{sentiment}**")
        display_confidence = 1 - confidence

    st.metric("Confidence", f"{display_confidence:.1%}")
    st.progress(min(max(display_confidence, 0.0), 1.0))

    with st.expander("Raw model output"):
        st.write(f"Sigmoid output (P(positive)): `{confidence:.4f}`")
