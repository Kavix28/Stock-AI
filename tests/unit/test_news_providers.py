import pytest
import os
import pandas as pd

try:
    from stock_ai.data.news.local_provider import LocalCanonicalNewsProvider
    from stock_ai.data.news.polygon_provider import PolygonNewsProvider
    from stock_ai.data.news.tiingo_provider import TiingoNewsProvider
    from stock_ai.data.news.sec_edgar_provider import SecEdgarNewsProvider
    from stock_ai.data.news.base import ProviderConfigurationError, ProviderDataError
except ImportError:
    from src.stock_ai.data.news.local_provider import LocalCanonicalNewsProvider
    from src.stock_ai.data.news.polygon_provider import PolygonNewsProvider
    from src.stock_ai.data.news.tiingo_provider import TiingoNewsProvider
    from src.stock_ai.data.news.sec_edgar_provider import SecEdgarNewsProvider
    from src.stock_ai.data.news.base import ProviderConfigurationError, ProviderDataError


def test_local_canonical_provider_metadata():
    provider = LocalCanonicalNewsProvider()
    meta = provider.get_provider_metadata()
    assert meta["provider_name"] == "local_canonical_archive"
    assert "TIMING-LIMITED" in meta["timestamp_quality"]
    assert meta["intraday_precision"] is False


def test_local_canonical_provider_fetch_real_file():
    provider = LocalCanonicalNewsProvider()
    if provider.is_configured():
        df = provider.fetch_news("AAPL", "2021-01-01", "2021-01-31")
        assert not df.empty
        assert "symbol" in df.columns
        assert (df["symbol"] == "AAPL").all()
        assert df.attrs.get("file_sha256") is not None


def test_local_canonical_provider_missing_ticker():
    provider = LocalCanonicalNewsProvider()
    with pytest.raises(ProviderDataError):
        provider.fetch_news("NONEXISTENT_TICKER_XYZ", "2021-01-01", "2021-01-31")


def test_polygon_provider_unconfigured():
    provider = PolygonNewsProvider(api_key=None)
    assert provider.is_configured() is False
    assert provider.get_required_credentials() == ["POLYGON_API_KEY"]
    with pytest.raises(ProviderConfigurationError) as exc_info:
        provider.fetch_news("AAPL", "2024-01-01", "2024-01-10")
    assert "POLYGON_API_KEY" in str(exc_info.value)


def test_tiingo_provider_unconfigured():
    provider = TiingoNewsProvider(api_key=None)
    assert provider.is_configured() is False
    assert provider.get_required_credentials() == ["TIINGO_API_KEY"]
    with pytest.raises(ProviderConfigurationError) as exc_info:
        provider.fetch_news("AAPL", "2024-01-01", "2024-01-10")
    assert "TIINGO_API_KEY" in str(exc_info.value)


def test_sec_edgar_provider_metadata():
    provider = SecEdgarNewsProvider()
    assert provider.is_configured() is True
    meta = provider.get_provider_metadata()
    assert meta["provider_name"] == "sec_edgar_disclosures"
    assert meta["intraday_precision"] is True
    assert "POINT-IN-TIME CERTIFIED" in meta["timestamp_quality"]
