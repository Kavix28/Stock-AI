import pytest
import pandas as pd

from src.stock_ai.data.news.validator import NewsDataValidator, NewsValidationError
from src.stock_ai.data.news.schema import CanonicalNewsSchema


def test_validator_canonical_dataframe():
    raw_df = pd.DataFrame({
        "date": ["2022-01-05", "2022-01-06"],
        "Titles": ["Headline 1", "Headline 2"]
    })
    norm_df = CanonicalNewsSchema.normalize_dataframe(raw_df, symbol="AAPL")

    stats = NewsDataValidator.validate_canonical_dataframe(norm_df, symbol="AAPL")
    assert stats["status"] == "VALID"
    assert stats["total_articles"] == 2
    assert stats["symbol"] == "AAPL"


def test_validator_detects_future_timestamp_leakage():
    # Construct raw news where article date is in the future relative to target
    raw_news = pd.DataFrame({
        "article_id": ["art_100"],
        "symbol": ["AAPL"],
        "published_date": ["2022-01-10"],  # Published on Jan 10
        "published_at_utc": [None],
        "has_intraday_time": [False],
        "title": ["Future headline"],
        "source": ["Reuters"],
        "provider": ["test"],
        "url": [None]
    })

    # Erroneous alignment assigning Jan 10 news to Jan 05 target
    aligned_df = pd.DataFrame([{
        "symbol": "AAPL",
        "target_date": "2022-01-05",  # Target earlier than article!
        "article_ids": ["art_100"],
        "combined_text": "Future headline",
        "has_news": True
    }])

    with pytest.raises(NewsValidationError) as exc_info:
        NewsDataValidator.validate_no_timestamp_leakage(aligned_df, raw_news)
    assert "LEAKAGE VIOLATION" in str(exc_info.value)


def test_validator_chronological_splits_valid():
    train_dates = ["2021-01-05", "2021-09-30"]
    val_dates = ["2021-10-01", "2021-12-31"]
    test_dates = ["2022-01-03", "2022-09-20"]

    # Should pass cleanly
    NewsDataValidator.validate_chronological_splits(train_dates, val_dates, test_dates)


def test_validator_chronological_splits_contamination():
    train_dates = ["2021-01-05", "2021-10-15"]  # Overlaps with val
    val_dates = ["2021-10-01", "2021-12-31"]
    test_dates = ["2022-01-03", "2022-09-20"]

    with pytest.raises(NewsValidationError) as exc_info:
        NewsDataValidator.validate_chronological_splits(train_dates, val_dates, test_dates)
    assert "SPLIT CONTAMINATION" in str(exc_info.value)
