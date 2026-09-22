"""
Model definition — Module 2: Hotel Review Sentiment Analysis
=================================================================
An Embedding + BiLSTM binary text classifier.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_sentiment_model(
    vocab_size: int, max_len: int, embedding_dim: int = 128, learning_rate: float = 1e-3
) -> keras.Model:
    """
    Architecture:
        Input(max_len,)
        -> Embedding(vocab_size, embedding_dim)
        -> Bidirectional(LSTM(64, return_sequences=True)) -> Dropout(0.3)
        -> Bidirectional(LSTM(32))                         -> Dropout(0.3)
        -> Dense(32, ReLU)
        -> Dense(1, Sigmoid)
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(max_len,)),
            layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="hotel_review_sentiment_bilstm",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model
