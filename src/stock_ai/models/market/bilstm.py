"""
Market-Only Bidirectional LSTM Architecture.
"""

from typing import Tuple, Optional
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Bidirectional, LSTM, Dense, Dropout


def build_market_bilstm(
    sequence_length: int = 20,
    feature_dim: int = 16,
    lstm_units_1: int = 64,
    lstm_units_2: int = 32,
    dense_units: int = 32,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3
) -> tf.keras.Model:
    """Builds and compiles standard market-only BiLSTM model."""
    model = Sequential([
        Input(shape=(sequence_length, feature_dim)),
        Bidirectional(LSTM(lstm_units_1, return_sequences=True)),
        Dropout(dropout_rate),
        Bidirectional(LSTM(lstm_units_2)),
        Dropout(dropout_rate),
        Dense(dense_units, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
