import pytest
from unittest.mock import MagicMock, patch
import pandas as pd

from stock_ai.data.market.yahooquery_provider import YahooQueryMarketDataProvider


def test_provider_initialization():
    provider = YahooQueryMarketDataProvider(retry_attempts=2, timeout=10, rate_limit_delay=0.1)
    assert provider.get_provider_name() == "YahooQuery"
    assert provider.retry_attempts == 2


@patch("stock_ai.data.market.yahooquery_provider.Ticker")
def test_provider_fetch_success(mock_ticker_cls, sample_raw_yahooquery_df):
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = sample_raw_yahooquery_df
    mock_ticker_cls.return_value = mock_ticker_instance

    provider = YahooQueryMarketDataProvider(retry_attempts=1, rate_limit_delay=0)
    df = provider.fetch_historical_data("AAPL", "2021-07-01", "2021-07-05")

    assert not df.empty
    assert len(df) == 5
    mock_ticker_instance.history.assert_called_once_with(start="2021-07-01", end="2021-07-05")


@patch("stock_ai.data.market.yahooquery_provider.Ticker")
def test_provider_retry_on_failure(mock_ticker_cls):
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.side_effect = [Exception("Network error"), Exception("Timeout")]
    mock_ticker_cls.return_value = mock_ticker_instance

    provider = YahooQueryMarketDataProvider(retry_attempts=2, retry_backoff_factor=1.0, rate_limit_delay=0)
    with pytest.raises(RuntimeError, match="Failed to fetch market data"):
        provider.fetch_historical_data("INVALID", "2021-07-01", "2021-07-05")

    assert mock_ticker_instance.history.call_count == 2
