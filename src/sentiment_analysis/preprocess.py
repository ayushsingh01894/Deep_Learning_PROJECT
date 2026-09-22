"""
Preprocessing pipeline — Module 2: Hotel Review Sentiment Analysis
======================================================================
Dataset: 515K Hotel Reviews Data in Europe
(Kaggle: jiashenliu/515k-hotel-reviews-data-in-europe)

Expected raw file: data/raw/hotel_reviews/Hotel_Reviews.csv

Run:
    python src/sentiment_analysis/preprocess.py

Label construction
-------------------
This dataset stores each review as TWO separate free-text fields
(Positive_Review, Negative_Review) rather than one text + one label.
We convert this into a standard binary sentiment classification dataset by:
  1. Treating the Positive_Review text of every row as a "positive" (label=1)
     sample, UNLESS it is the placeholder "No Positive".
  2. Treating the Negative_Review text of every row as a "negative" (label=0)
     sample, UNLESS it is the placeholder "No Negative".
This roughly doubles the number of rows (up to 2 samples per original row)
and produces a clean, real, balanced-ish binary text classification dataset
without inventing any labels — the label comes directly from which field
the real reviewer text was written in.

By default we subsample the data (see MAX_SAMPLES) since 515K reviews
(~1M+ text samples after splitting) can be heavy to tokenize/train on
typical student hardware within a reasonable time. This is documented,
not hidden, per the project's honesty requirements.

Produces (under data/processed/hotel_reviews/):
    - X_train.npy, X_val.npy, X_test.npy   (padded token sequences)
    - y_train.npy, y_val.npy, y_test.npy   (0=negative, 1=positive)
Produces (under models/sentiment_analysis/):
    - tokenizer.joblib
    - preprocess_config.json   (max_len, vocab_size actually used)
"""

import json
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.config import RANDOM_SEED, REVIEWS_PROCESSED_DIR, REVIEWS_RAW_DIR, SENTIMENT_MODEL_DIR, ensure_dirs
from src.common.utils import set_global_seed

RAW_CSV = os.path.join(REVIEWS_RAW_DIR, "Hotel_Reviews.csv")

# Subsample cap — set to None to use the full dataset (515K rows -> up to ~1M text samples).
# 120,000 original rows keeps tokenizer/training time reasonable on a laptop/Colab while
# still being a large, real dataset.
MAX_SAMPLES = 120_000

VOCAB_SIZE = 20_000
MAX_LEN = 100  # tokens per review, truncate/pad to this length


def load_raw_data() -> pd.DataFrame:
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"Could not find {RAW_CSV}.\n"
            "Download the '515K Hotel Reviews Data in Europe' dataset from Kaggle "
            "(jiashenliu/515k-hotel-reviews-data-in-europe) and place Hotel_Reviews.csv "
            "at that path. See README.md section 12."
        )
    return pd.read_csv(RAW_CSV)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_labeled_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Split each row's Positive_Review / Negative_Review into separate
    labeled text samples, dropping the dataset's placeholder strings."""

    if MAX_SAMPLES is not None and len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)

    pos = df["Positive_Review"].astype(str).str.strip()
    neg = df["Negative_Review"].astype(str).str.strip()

    pos_mask = ~pos.str.lower().isin(["no positive", ""])
    neg_mask = ~neg.str.lower().isin(["no negative", ""])

    pos_samples = pd.DataFrame({"text": pos[pos_mask], "label": 1})
    neg_samples = pd.DataFrame({"text": neg[neg_mask], "label": 0})

    combined = pd.concat([pos_samples, neg_samples], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)  # shuffle
    return combined


def run_preprocessing():
    set_global_seed(RANDOM_SEED)
    ensure_dirs(REVIEWS_PROCESSED_DIR, SENTIMENT_MODEL_DIR)

    print("[preprocess] Loading raw data...")
    df = load_raw_data()
    print(f"[preprocess] Raw shape: {df.shape}")

    labeled = build_labeled_texts(df)
    print(f"[preprocess] Labeled text samples after split: {len(labeled)}")
    print(f"[preprocess] Label balance:\n{labeled['label'].value_counts(normalize=True)}")

    labeled["text"] = labeled["text"].apply(clean_text)
    labeled = labeled[labeled["text"].str.len() > 0].reset_index(drop=True)

    texts = labeled["text"].tolist()
    labels = labeled["label"].values

    # Split 70/15/15, stratified
    texts_train, texts_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.30, random_state=RANDOM_SEED, stratify=labels
    )
    texts_val, texts_test, y_val, y_test = train_test_split(
        texts_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )

    # Fit tokenizer ONLY on training text (avoid leakage)
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts_train)

    def to_padded(texts_list):
        seqs = tokenizer.texts_to_sequences(texts_list)
        return pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

    X_train = to_padded(texts_train)
    X_val = to_padded(texts_val)
    X_test = to_padded(texts_test)

    np.save(os.path.join(REVIEWS_PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(REVIEWS_PROCESSED_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(REVIEWS_PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(REVIEWS_PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(REVIEWS_PROCESSED_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(REVIEWS_PROCESSED_DIR, "y_test.npy"), y_test)

    joblib.dump(tokenizer, os.path.join(SENTIMENT_MODEL_DIR, "tokenizer.joblib"))
    with open(os.path.join(SENTIMENT_MODEL_DIR, "preprocess_config.json"), "w") as f:
        json.dump(
            {"vocab_size": VOCAB_SIZE, "max_len": MAX_LEN, "max_samples_used": MAX_SAMPLES},
            f,
            indent=2,
        )

    print("[preprocess] Done.")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Saved processed data to {REVIEWS_PROCESSED_DIR}")
    print(f"  Saved tokenizer + config to {SENTIMENT_MODEL_DIR}")


if __name__ == "__main__":
    run_preprocessing()
