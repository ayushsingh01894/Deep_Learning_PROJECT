"""
Inference helper — Module 2: Hotel Review Sentiment Analysis
=================================================================
Loads the trained BiLSTM + tokenizer and exposes `predict_sentiment(text)`.
"""

import json
import os
import sys

import joblib
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.config import SENTIMENT_MODEL_DIR
from src.sentiment_analysis.preprocess import clean_text

MODEL_PATH = os.path.join(SENTIMENT_MODEL_DIR, "sentiment_bilstm.keras")
TOKENIZER_PATH = os.path.join(SENTIMENT_MODEL_DIR, "tokenizer.joblib")
CONFIG_PATH = os.path.join(SENTIMENT_MODEL_DIR, "preprocess_config.json")


def artifacts_available() -> bool:
    return all(os.path.exists(p) for p in (MODEL_PATH, TOKENIZER_PATH, CONFIG_PATH))


def load_artifacts():
    if not artifacts_available():
        raise FileNotFoundError(
            "Module 2 (Sentiment Analysis) model artifacts not found. "
            "Run preprocess.py and train.py for this module first."
        )
    model = keras.models.load_model(MODEL_PATH)
    tokenizer = joblib.load(TOKENIZER_PATH)
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    return model, tokenizer, config


def predict_sentiment(text: str) -> dict:
    """
    Returns: {"sentiment": "Positive"/"Negative", "confidence": float}
    `confidence` is the model's raw sigmoid output interpreted as
    P(positive); values near 0 mean confidently negative, near 1 mean
    confidently positive.
    """
    model, tokenizer, config = load_artifacts()

    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=config["max_len"], padding="post", truncating="post")

    proba = float(model.predict(padded, verbose=0).ravel()[0])
    sentiment = "Positive" if proba >= 0.5 else "Negative"

    return {"sentiment": sentiment, "confidence": proba}
