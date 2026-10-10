import pytest
import pandas as pd
from datetime import datetime
import pytz

from src.stock_ai.data.news.alignment import NewsMarketAligner, CausalAlignmentError
from src.stock_ai.data.news.schema import CanonicalNewsSchema


def test_market_session_bounds_utc():
    # Test during Standard Time (EST = UTC-5)
    start_utc, end_utc = NewsMarketAligner.get_market_session_bounds_utc("2022-01-05", "2022-01-06")
    # Jan 5 16:00 EST -> Jan 5 21:00 UTC
    assert start_utc.strftime("%Y-%m-%d %H:%M:%S") == "2022-01-05 21:00:00"
    # Jan 6 09:30 EST -> Jan 6 14:30 UTC
    assert end_utc.strftime("%Y-%m-%d %H:%M:%S") == "2022-01-06 14:30:00"

    # Test during Daylight Saving Time (EDT = UTC-4)
    start_dst, end_dst = NewsMarketAligner.get_market_session_bounds_utc("2022-06-01", "2022-06-02")
    # Jun 1 16:00 EDT -> Jun 1 20:00 UTC
    assert start_dst.strftime("%Y-%m-%d %H:%M:%S") == "2022-06-01 20:00:00"
    # Jun 2 09:30 EDT -> Jun 2 13:30 UTC
    assert end_dst.strftime("%Y-%m-%d %H:%M:%S") == "2022-06-02 13:30:00"


def test_align_missing_news_handling():
    aligner = NewsMarketAligner(alignment_mode="DAILY_DATE_CONVENTIONAL")
    empty_news = pd.DataFrame(columns=["article_id", "symbol", "published_date", "published_at_utc", "has_intraday_time", "title", "source", "provider", "url"])
    trading_dates = ["2022-01-03", "2022-01-04", "2022-01-05"]

    aligned = aligner.align_ticker_news(empty_news, trading_dates, symbol="AAPL")

    assert len(aligned) == 2  # indices 1 and 2
    assert (aligned["has_news"] == False).all()
    assert (aligned["article_count"] == 0).all()
    assert (aligned["combined_text"] == "").all()


def test_align_deduplication():
    aligner = NewsMarketAligner(alignment_mode="DAILY_DATE_CONVENTIONAL")
    raw_df = pd.DataFrame({
        "date": ["2022-01-04", "2022-01-04", "2022-01-04"],
        "Titles": [
            "Apple reports record quarterly revenue",
            "Apple reports record quarterly revenue",  # Exact duplicate
            "Apple launches new services"
        ]
    })
    news_df = CanonicalNewsSchema.normalize_dataframe(raw_df, symbol="AAPL")
    trading_dates = ["2022-01-03", "2022-01-04"]

    aligned = aligner.align_ticker_news(news_df, trading_dates, symbol="AAPL")

    assert len(aligned) == 1
    row = aligned.iloc[0]
    assert row["has_news"] == True
    assert row["article_count"] == 2  # Deduplicated from 3 to 2
    assert row["duplicate_count"] == 1
    assert "Apple reports record quarterly revenue" in row["combined_text"]
    assert "Apple launches new services" in row["combined_text"]


def test_align_unsorted_dates_raises_error():
    aligner = NewsMarketAligner(alignment_mode="DAILY_DATE_CONVENTIONAL")
    unsorted_dates = ["2022-01-05", "2022-01-03", "2022-01-04"]

    with pytest.raises(CausalAlignmentError):
        aligner.align_ticker_news(pd.DataFrame(), unsorted_dates, symbol="AAPL")
