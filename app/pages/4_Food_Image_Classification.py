"""
Streamlit page — Module 4: Food Image Classification
==========================================================
"""

import os
import sys

import pandas as pd
import streamlit as st
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.food_classification.predict import artifacts_available, predict_food_image

st.set_page_config(page_title="Food Image Classification", page_icon="🍽️", layout="wide")
st.title("🍽️ Food Image Classification")
st.caption("CNN (MobileNetV2 Transfer Learning) trained on a 12-class Food-101 subset")

if not artifacts_available():
    st.warning(
        "⚠️ Model artifacts not found. Run the following before using this page:\n\n"
        "```bash\npython src/food_classification/preprocess.py\n"
        "python src/food_classification/train.py\n```"
    )
    st.stop()

st.markdown(
    "Upload a photo of a food item to classify it. The model recognizes 12 categories "
    "relevant to a hotel menu / breakfast buffet: waffles, pancakes, omelette, french toast, "
    "pizza, sushi, hamburger, caesar salad, steak, ice cream, donuts, and club sandwich."
)

uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1.2])

    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    try:
        result = predict_food_image(image, top_k=3)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    with col2:
        predicted_class = result["predicted_class"].replace("_", " ").title()
        st.success(f"### Predicted: {predicted_class}")
        st.metric("Confidence", f"{result['confidence']:.1%}")

        st.markdown("#### Top 3 predictions")
        top_k_df = pd.DataFrame(
            [(c.replace("_", " ").title(), p) for c, p in result["top_k"]],
            columns=["Food Category", "Probability"],
        )
        st.dataframe(
            top_k_df.style.format({"Probability": "{:.1%}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(top_k_df.set_index("Food Category")["Probability"])
