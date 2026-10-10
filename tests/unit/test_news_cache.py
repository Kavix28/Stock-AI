import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from src.stock_ai.data.news.cache import NewsEmbeddingCache


def test_cache_save_and_load(tmp_path):
    # Mock encoder
    mock_encoder = MagicMock()
    mock_encoder.embedding_dim = 768
    mock_encoder.encode_batch.return_value = np.ones((2, 768), dtype=np.float32)
    mock_encoder.get_zero_news_vector.return_value = np.zeros(768, dtype=np.float32)

    cache = NewsEmbeddingCache(cache_dir=str(tmp_path), encoder=mock_encoder)

    aligned_df = pd.DataFrame([
        {
            "target_date": "2022-01-05",
            "combined_text": "Apple quarterly profit beats Wall St targets",
            "has_news": True,
            "article_count": 1
        },
        {
            "target_date": "2022-01-06",
            "combined_text": "Tech stocks slip on rate hike fears",
            "has_news": True,
            "article_count": 1
        }
    ])

    manifest = cache.build_ticker_cache("AAPL", aligned_df)

    assert manifest["ticker"] == "AAPL"
    assert manifest["total_dates"] == 2
    assert manifest["dates_with_news"] == 2
    assert manifest["news_coverage_pct"] == 100.0

    # Load back
    embeddings, date_to_idx, has_news = cache.load_ticker_embeddings("AAPL")
    assert embeddings.shape == (2, 768)
    assert date_to_idx["2022-01-05"] == 0
    assert date_to_idx["2022-01-06"] == 1
    assert has_news[0] == True
    assert has_news[1] == True


def test_cache_handles_no_news_dates(tmp_path):
    mock_encoder = MagicMock()
    mock_encoder.embedding_dim = 768
    # 1 valid, 1 no-news
    mock_encoder.encode_batch.return_value = np.array([
        np.ones(768, dtype=np.float32),
        np.zeros(768, dtype=np.float32)
    ])
    mock_encoder.get_zero_news_vector.return_value = np.zeros(768, dtype=np.float32)

    cache = NewsEmbeddingCache(cache_dir=str(tmp_path), encoder=mock_encoder)

    aligned_df = pd.DataFrame([
        {
            "target_date": "2022-01-05",
            "combined_text": "Earnings jump",
            "has_news": True,
            "article_count": 1
        },
        {
            "target_date": "2022-01-06",
            "combined_text": "",
            "has_news": False,
            "article_count": 0
        }
    ])

    cache.build_ticker_cache("NVDA", aligned_df)

    emb_0, has_news_0 = cache.get_embedding_for_date("NVDA", "2022-01-05")
    assert has_news_0 == True
    assert np.all(emb_0 == 1.0)

    emb_1, has_news_1 = cache.get_embedding_for_date("NVDA", "2022-01-06")
    assert has_news_1 == False
    assert np.all(emb_1 == 0.0)

    # Missing date
    emb_miss, has_news_miss = cache.get_embedding_for_date("NVDA", "2099-01-01")
    assert has_news_miss == False
    assert np.all(emb_miss == 0.0)
