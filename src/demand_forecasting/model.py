"""
Model definition — Module 3: Room Demand Forecasting
========================================================
An LSTM-based sequence model that predicts next-day room demand from the
previous WINDOW_SIZE days of realized bookings.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_model(window_size: int, learning_rate: float = 1e-3) -> keras.Model:
    """
    Architecture:
        Input(window_size, 1)
        -> LSTM(64, return_sequences=True) -> Dropout(0.2)
        -> LSTM(32)                        -> Dropout(0.2)
        -> Dense(16, ReLU)
        -> Dense(1, linear)   [regression output: predicted bookings]
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(window_size, 1)),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="linear"),
        ],
        name="room_demand_lstm",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model
