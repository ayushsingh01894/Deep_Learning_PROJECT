"""
Model definition — Module 1: Hotel Booking Prediction
========================================================
A feed-forward ANN (DNN) for binary classification of booking cancellations.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_ann_model(input_dim: int, learning_rate: float = 1e-3) -> keras.Model:
    """
    Architecture:
        Input(input_dim)
        -> Dense(128, ReLU) -> BatchNorm -> Dropout(0.3)
        -> Dense(64, ReLU)  -> Dropout(0.2)
        -> Dense(32, ReLU)
        -> Dense(1, Sigmoid)
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="hotel_booking_ann",
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
