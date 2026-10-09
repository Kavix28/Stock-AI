import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

from stock_ai.data.market.base import MarketDataProvider
from stock_ai.data.market.fallback_provider import FallbackMarketDataProvider, LocalCsvMarketDataProvider
from stock_ai.data.market.service import MarketDataService
from stock_ai.config.settings import load_config


class FailingProvider(MarketDataProvider):
    def get_provider_name(self) -> str:
        return "FailingProvider"

    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        raise RuntimeError(f"FailingProvider deliberately failed for symbol '{symbol}'")


class DummySecondaryProvider(MarketDataProvider):
    def get_provider_name(self) -> str:
        return "DummySecondary"

    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return pd.DataFrame({
            "symbol": [symbol],
            "date": ["2021-07-01"],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [104.0],
            "adjclose": [103.5],
            "volume": [10000]
        })


def test_fallback_provider_primary_success():
    primary = DummySecondaryProvider()
    fallback_prov = FallbackMarketDataProvider(primary_provider=primary, secondary_provider=None)
    df = fallback_prov.fetch_historical_data("AAPL", "2021-07-01", "2021-07-05")

    assert not df.empty
    assert fallback_prov.get_provider_for_symbol("AAPL") == "DummySecondary"


def test_fallback_provider_primary_fails_secondary_succeeds():
    primary = FailingProvider()
    secondary = DummySecondaryProvider()
    fallback_prov = FallbackMarketDataProvider(primary_provider=primary, secondary_provider=secondary)
    df = fallback_prov.fetch_historical_data("ATVI", "2021-07-01", "2021-07-05")

    assert not df.empty
    assert fallback_prov.get_provider_for_symbol("ATVI") == "DummySecondary"


def test_local_csv_provider(tmp_path):
    csv_file = tmp_path / "ATVI.csv"
    df_raw = pd.DataFrame({
        "symbol": ["ATVI"] * 2,
        "date": ["2021-07-01", "2021-07-02"],
        "open": [90.0, 91.0],
        "high": [92.0, 93.0],
        "low": [89.0, 90.0],
        "close": [91.5, 92.5],
        "adjclose": [91.0, 92.0],
        "volume": [50000, 60000]
    })
    df_raw.to_csv(csv_file, index=False)

    local_prov = LocalCsvMarketDataProvider(data_dir=tmp_path)
    assert local_prov.get_provider_name() == "LocalCSVArchive"
    
    df_res = local_prov.fetch_historical_data("ATVI", "2021-07-01", "2021-07-02")
    assert len(df_res) == 2
    assert df_res["symbol"].iloc[0] == "ATVI"


def test_manifest_status_when_stock_fails(tmp_path):
    config = load_config()
    config.storage.raw_market_dir = str(tmp_path / "raw")
    config.storage.processed_market_dir = str(tmp_path / "processed")
    config.storage.reports_dir = str(tmp_path / "reports")
    config.storage.manifest_file = str(tmp_path / "processed" / "manifest.json")
    config.storage.quality_report_file = str(tmp_path / "reports" / "report.md")

    failing_service = MarketDataService(config=config, provider=FailingProvider())
    res = failing_service.fetch_all(symbols=["ATVI"], start_date="2021-07-01", end_date="2021-07-05")

    assert res["success"] is False
    assert res["results"]["ATVI"]["status"] == "FAILED"
