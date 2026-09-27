"""
Tests — Module 2: Hotel Review Sentiment Analysis
======================================================
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sentiment_analysis.model import build_sentiment_model
from src.sentiment_analysis.preprocess import build_labeled_texts, clean_text


class TestCleanText:
    def test_lowercases_and_strips_punctuation(self):
        assert clean_text("Great Room!! Loved it.") == "great room loved it"

    def test_collapses_whitespace(self):
        assert clean_text("too   many    spaces") == "too many spaces"

    def test_handles_empty_string(self):
        assert clean_text("") == ""

    def test_preserves_apostrophes(self):
        assert "didn't" in clean_text("The room didn't have a view")


class TestBuildLabeledTexts:
    def test_produces_binary_labels(self, synthetic_reviews_df):
        labeled = build_labeled_texts(synthetic_reviews_df)
        assert set(labeled["label"].unique()).issubset({0, 1})

    def test_drops_placeholder_text(self, synthetic_reviews_df):
        labeled = build_labeled_texts(synthetic_reviews_df)
        texts_lower = labeled["text"].str.lower()
        assert not texts_lower.isin(["no positive", "no negative"]).any()

    def test_sample_count_is_at_most_double_input_rows(self, synthetic_reviews_df):
        labeled = build_labeled_texts(synthetic_reviews_df)
        assert len(labeled) <= 2 * len(synthetic_reviews_df)

    def test_positive_and_negative_both_present(self, synthetic_reviews_df):
        labeled = build_labeled_texts(synthetic_reviews_df)
        assert (labeled["label"] == 1).sum() > 0
        assert (labeled["label"] == 0).sum() > 0


class TestBuildSentimentModel:
    def test_model_compiles_and_has_correct_io_shape(self):
        model = build_sentiment_model(vocab_size=1000, max_len=50)
        assert model.input_shape == (None, 50)
        assert model.output_shape == (None, 1)

    def test_model_predicts_valid_probabilities(self):
        model = build_sentiment_model(vocab_size=500, max_len=20)
        X_dummy = np.random.randint(0, 500, size=(4, 20))
        preds = model.predict(X_dummy, verbose=0)
        assert preds.shape == (4, 1)
        assert (preds >= 0).all() and (preds <= 1).all()


class TestPredictIntegration:
    def test_predict_sentiment_end_to_end(self):
        from src.sentiment_analysis.predict import artifacts_available, predict_sentiment

        if not artifacts_available():
            pytest.skip("Module 2 model artifacts not found — train the model first to run this test.")

        result = predict_sentiment("The staff were incredibly friendly and helpful")

        assert result["sentiment"] in {"Positive", "Negative"}
        assert 0.0 <= result["confidence"] <= 1.0
