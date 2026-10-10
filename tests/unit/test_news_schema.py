import pytest
import pandas as pd
from src.stock_ai.data.news.schema import CanonicalNewsSchema, CANONICAL_NEWS_COLUMNS


def test_clean_headline_text():
    assert CanonicalNewsSchema.clean_headline_text("  Apple   launches new  MacBook \n ") == "Apple launches new MacBook"
    assert CanonicalNewsSchema.clean_headline_text(None) == ""
    assert CanonicalNewsSchema.clean_headline_text(float("nan")) == ""


def test_generate_article_id():
    id1 = CanonicalNewsSchema.generate_article_id("AAPL", "2022-01-05", "Apple reports record Q1 earnings")
    id2 = CanonicalNewsSchema.generate_article_id("AAPL", "2022-01-05", "Apple reports record Q1 earnings")
    id3 = CanonicalNewsSchema.generate_article_id("AAPL", "2022-01-05", "Different title")

    assert id1 == id2
    assert id1 != id3
    assert len(id1) == 16


def test_normalize_raw_news_dataframe():
    raw_df = pd.DataFrame({
        "date": ["2022-01-05", "2022-01-06", "2022-01-06"],
        "Titles": ["Apple event announced", "   ", "Tesla cuts prices"]
    })

    norm_df = CanonicalNewsSchema.normalize_dataframe(raw_df, symbol="AAPL")

    assert list(norm_df.columns) == CANONICAL_NEWS_COLUMNS
    assert len(norm_df) == 2  # Empty title row dropped
    assert norm_df["symbol"].iloc[0] == "AAPL"
    assert norm_df["published_date"].iloc[0] == "2022-01-05"
    assert norm_df["has_intraday_time"].iloc[0] == False


def test_normalize_with_intraday_timestamps():
    raw_df = pd.DataFrame({
        "published_at": ["2024-08-01 14:30:00", "2024-08-01 16:45:00"],
        "headline": ["Fed cuts interest rates", "Tech rally surges"]
    })

    norm_df = CanonicalNewsSchema.normalize_dataframe(raw_df, symbol="NVDA")

    assert len(norm_df) == 2
    assert norm_df["has_intraday_time"].all()
    assert norm_df["published_at_utc"].iloc[0].endswith("Z")
    assert norm_df["published_date"].iloc[0] == "2024-08-01"
