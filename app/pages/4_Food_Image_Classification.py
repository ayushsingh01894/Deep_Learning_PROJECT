"""
Streamlit page — Module 4: Food Image Classification
==========================================================
"""

import json
import os
import sys

import pandas as pd
import streamlit as st
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.food_classification.predict import CLASS_NAMES_PATH, artifacts_available, predict_food_image

# Below this confidence the page warns the user instead of presenting the prediction as reliable.
LOW_CONFIDENCE_THRESHOLD = 0.50

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

with open(CLASS_NAMES_PATH, "r") as f:
    _class_names = json.load(f)
_pretty_classes = ", ".join(c.replace("_", " ") for c in _class_names)

st.markdown(
    f"Upload a photo of a food item to classify it. The model recognizes {len(_class_names)} categories: "
    f"{_pretty_classes}."
)
st.caption(
    "Tip: use a clear, well-lit photo with a single food item. Photos with many different dishes, "
    "or foods outside these categories, will give unreliable results."
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
        if result["confidence"] >= LOW_CONFIDENCE_THRESHOLD:
            st.success(f"### Predicted: {predicted_class}")
        else:
            st.warning(
                f"### Not sure — best guess: {predicted_class}\n\n"
                "The model's confidence is low. Try a clearer photo of a single item, "
                "or the food may not be one of the supported categories."
            )
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
