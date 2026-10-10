import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

from src.stock_ai.models.news.finbert import FinBertNewsEncoder


def test_encoder_init_and_metadata():
    encoder = FinBertNewsEncoder(max_length=64, pooling_method="cls", lazy_load=True)
    meta = encoder.get_metadata()

    assert meta["model_name"] == "ProsusAI/finbert"
    assert meta["embedding_dim"] == 768
    assert meta["max_length"] == 64
    assert meta["pooling_method"] == "cls"
    assert meta["is_loaded"] is False


def test_invalid_pooling_method_raises():
    with pytest.raises(ValueError, match="Unsupported pooling method"):
        FinBertNewsEncoder(pooling_method="invalid")


def test_zero_news_vector():
    encoder = FinBertNewsEncoder(lazy_load=True)
    zero_vec = encoder.get_zero_news_vector()

    assert zero_vec.shape == (768,)
    assert zero_vec.dtype == np.float32
    assert np.all(zero_vec == 0.0)


def test_empty_string_returns_zero_vector():
    encoder = FinBertNewsEncoder(lazy_load=True)
    assert np.array_equal(encoder.encode_text(""), encoder.get_zero_news_vector())
    assert np.array_equal(encoder.encode_text("   "), encoder.get_zero_news_vector())
    assert np.array_equal(encoder.encode_text(None), encoder.get_zero_news_vector())


def test_encode_batch_with_mocked_transformer():
    encoder = FinBertNewsEncoder(lazy_load=True)

    # Set up mock tokenizer and model
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    # Mock tokenizer output
    mock_inputs = {
        "input_ids": torch.tensor([[101, 200, 102], [101, 300, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]])
    }
    # Mock model output: last_hidden_state of shape (batch_size=2, seq_len=3, hidden_dim=768)
    hidden_states = torch.ones((2, 3, 768), dtype=torch.float32)
    # Give sample 1 distinct values
    hidden_states[1] = hidden_states[1] * 2.0

    mock_outputs = MagicMock()
    mock_outputs.last_hidden_state = hidden_states
    mock_model.return_value = mock_outputs

    encoder.tokenizer = mock_tokenizer
    encoder.model = mock_model
    encoder._is_loaded = True

    # Call encode_batch with 3 items (one empty)
    texts = ["Apple beats earnings expectations", "", "Fed cuts rates"]

    # When tokenizer is called on valid slice (len 2)
    def mock_tokenize_call(batch, **kwargs):
        mock_in = MagicMock()
        mock_in.to.return_value = {
            "input_ids": torch.tensor([[101, 200, 102], [101, 300, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]])
        }
        mock_in.__getitem__ = lambda self, key: mock_in.to.return_value[key]
        return mock_in

    mock_tokenizer.side_effect = mock_tokenize_call

    embeddings = encoder.encode_batch(texts, batch_size=2)

    assert embeddings.shape == (3, 768)
    assert embeddings.dtype == np.float32
    # Item 0 should have non-zero embeddings
    assert np.all(embeddings[0] == 1.0)
    # Item 1 (empty string) should be zero vector
    assert np.all(embeddings[1] == 0.0)
    # Item 2 should have non-zero embeddings
    assert np.all(embeddings[2] == 2.0)
