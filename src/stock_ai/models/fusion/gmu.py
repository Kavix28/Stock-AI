"""
Gated Multimodal Unit (GMU) and Multimodal Fusion Architectures.

Implements dimensionally valid gated fusion:
z = sigmoid(Dense(64)(concatenate([h_market, h_news])))
h_fused = z * h_market_proj + (1 - z) * h_news_proj
"""

import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Bidirectional, LSTM, Dense, Dropout,
    LayerNormalization, Concatenate, Multiply, Add, Lambda
)


def build_gmu_fusion_model(
    seq_len: int = 20,
    market_feat_dim: int = 16,
    news_embed_dim: int = 768,
    proj_dim: int = 64,
    learning_rate: float = 1e-3
) -> Model:
    """Builds a Gated Multimodal Unit fusion model combining BiLSTM market branch with FinBERT news branch."""
    # 1. Market Branch
    in_market = Input(shape=(seq_len, market_feat_dim), name="market_input")
    x_m = Bidirectional(LSTM(64, return_sequences=True))(in_market)
    x_m = Dropout(0.2)(x_m)
    x_m = Bidirectional(LSTM(32))(x_m)
    x_m = Dropout(0.2)(x_m)
    h_m_proj = Dense(proj_dim, activation="relu", name="market_projection")(x_m)
    h_m_proj = LayerNormalization(name="market_ln")(h_m_proj)

    # 2. News Branch (Pre-extracted FinBERT representation)
    in_news = Input(shape=(news_embed_dim,), name="news_input")
    h_n_proj = Dense(proj_dim, activation="relu", name="news_projection")(in_news)
    h_n_proj = LayerNormalization(name="news_ln")(h_n_proj)

    # 3. Learned Gating Mechanism
    concat_repr = Concatenate(name="fusion_concat")([h_m_proj, h_n_proj])
    gate = Dense(proj_dim, activation="sigmoid", name="gmu_gate")(concat_repr)

    # 4. Gated Convex Combination
    market_gated = Multiply(name="market_gated")([gate, h_m_proj])
    one_minus_gate = Lambda(lambda g: 1.0 - g, name="one_minus_gate")(gate)
    news_gated = Multiply(name="news_gated")([one_minus_gate, h_n_proj])
    h_fused = Add(name="gmu_fused")([market_gated, news_gated])

    # 5. Classification Head
    out_dense = Dense(32, activation="relu")(h_fused)
    out_dense = Dropout(0.2)(out_dense)
    out = Dense(1, activation="sigmoid", name="prediction")(out_dense)

    model = Model(inputs=[in_market, in_news], outputs=out, name="GMU_Stock_Predictor")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model
