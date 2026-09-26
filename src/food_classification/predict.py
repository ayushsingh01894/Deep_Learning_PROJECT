"""
Inference helper — Module 4: Food Image Classification
===========================================================
Loads the trained CNN + class names and exposes `predict_food_image(image)`.
"""

import json
import os
import sys

import numpy as np
from PIL import Image
from tensorflow import keras

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.config import FOOD_MODEL_DIR
from src.food_classification.model import IMG_SIZE

MODEL_PATH = os.path.join(FOOD_MODEL_DIR, "food_cnn.keras")
CLASS_NAMES_PATH = os.path.join(FOOD_MODEL_DIR, "class_names.json")


def artifacts_available() -> bool:
    return all(os.path.exists(p) for p in (MODEL_PATH, CLASS_NAMES_PATH))


def load_artifacts():
    if not artifacts_available():
        raise FileNotFoundError(
            "Module 4 (Food Classification) model artifacts not found. "
            "Run preprocess.py and train.py for this module first."
        )
    model = keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    return model, class_names


def predict_food_image(image: Image.Image, top_k: int = 3) -> dict:
    """
    image: a PIL Image (as returned by st.file_uploader + Image.open)

    Returns: {
        "predicted_class": str,
        "confidence": float,
        "top_k": [(class_name, probability), ...]  sorted descending
    }
    """
    model, class_names = load_artifacts()

    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = keras.utils.img_to_array(image)
    arr = np.expand_dims(arr, axis=0)  # model applies its own preprocess_input internally

    proba = model.predict(arr, verbose=0).ravel()
    top_indices = np.argsort(proba)[::-1][:top_k]
    top_k_results = [(class_names[i], float(proba[i])) for i in top_indices]

    return {
        "predicted_class": top_k_results[0][0],
        "confidence": top_k_results[0][1],
        "top_k": top_k_results,
    }
