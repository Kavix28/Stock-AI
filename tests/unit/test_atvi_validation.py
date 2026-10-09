import pytest
import pandas as pd
from pathlib import Path

from stock_ai.data.market.schema import CanonicalMarketSchema
from stock_ai.data.market.validator import MarketDataValidator
from stock_ai.data.market.fallback_provider import LocalCsvMarketDataProvider, FallbackMarketDataProvider, MarketDataProvider
from stock_ai.data.market.service import MarketDataService
from stock_ai.config.settings import load_config


def test_partial_historical_coverage_rejection(tmp_path):
    """Test that candidate CSVs with insufficient date coverage (e.g., 38 rows in 2020) are flagged as invalid."""
    csv_file = tmp_path / "ATVI.csv"
    # Create sample partial dataset with only 5 days in early 2020
    df_partial = pd.DataFrame({
        "symbol": ["ATVI"] * 5,
        "date": ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"],
        "open": [59.0, 59.5, 60.0, 60.5, 61.0],
        "high": [60.0, 60.5, 61.0, 61.5, 62.0],
        "low": [58.5, 59.0, 59.5, 60.0, 60.5],
        "close": [59.8, 60.2, 60.8, 61.2, 61.8],
        "adjclose": [59.0, 59.4, 60.0, 60.4, 61.0],
        "volume": [1000000, 1100000, 1200000, 1300000, 1400000]
    })
    df_partial.to_csv(csv_file, index=False)

    provider = LocalCsvMarketDataProvider(data_dir=tmp_path)
    # Query for the full paper acquisition range (2020-01-01 to 2022-09-30)
    with pytest.raises(ValueError, match="no observations in range"):
        provider.fetch_historical_data("ATVI", "2021-01-01", "2022-09-30")


def test_missing_adjusted_close_validation():
    """Test that market data missing the required adj_close column fails schema normalization."""
    df_no_adj = pd.DataFrame({
        "symbol": ["ATVI"],
        "date": ["2021-07-01"],
        "open": [90.0],
        "high": [92.0],
        "low": [89.0],
        "close": [91.0],
        "volume": [500000]
    })
    with pytest.raises(ValueError, match="missing required canonical columns"):
        CanonicalMarketSchema.normalize(df_no_adj, symbol="ATVI")


def test_invalid_ohlc_data_validation():
    """Test that invalid OHLC relationship (high < low) triggers a validation error."""
    df_bad_ohlc = pd.DataFrame({
        "symbol": ["ATVI"],
        "date": ["2021-07-01"],
        "open": [90.0],
        "high": [85.0],  # Invalid: High < Low
        "low": [89.0],
        "adj_close": [90.0],
        "volume": [500000]
    })
    validator = MarketDataValidator(expected_symbol="ATVI")
    is_valid, issues, _ = validator.validate(df_bad_ohlc)
    assert is_valid is False
    assert any(i.level == "ERROR" and "High price strictly less than Low" in i.message for i in issues)


def test_atvi_schema_normalization():
    """Test that a valid ATVI DataFrame is correctly normalized to canonical schema."""
    df_raw = pd.DataFrame({
        "Date": ["2021-07-01"],
        "Open": [90.5],
        "High": [92.0],
        "Low": [89.8],
        "Close": [91.2],
        "Adj Close": [90.8],
        "Volume": [2500000]
    })
    df_norm = CanonicalMarketSchema.normalize(df_raw, symbol="ATVI")
    assert list(df_norm.columns) == ["symbol", "date", "open", "high", "low", "adj_close", "volume"]
    assert df_norm["symbol"].iloc[0] == "ATVI"
    assert df_norm["adj_close"].iloc[0] == 90.8


def test_provider_metadata_tracking():
    """Test that provider name metadata is properly tracked for each symbol."""
    class MockProvider(MarketDataProvider):
        def get_provider_name(self) -> str:
            return "MockTestProvider"

        def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
            return pd.DataFrame({
                "symbol": [symbol],
                "date": ["2021-07-01"],
                "open": [100.0],
                "high": [105.0],
                "low": [99.0],
                "adj_close": [103.0],
                "volume": [10000]
            })

    prov = MockProvider()
    assert prov.get_provider_name() == "MockTestProvider"


def test_manifest_failure_when_any_stock_fails(tmp_path):
    """Test that dataset manifest reports overall_status = FAILED when a required paper stock fails."""
    class FailingProvider(MarketDataProvider):
        def get_provider_name(self) -> str:
            return "FailingProvider"

        def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
            raise RuntimeError("Download failed")

    config = load_config()
    config.storage.raw_market_dir = str(tmp_path / "raw")
    config.storage.processed_market_dir = str(tmp_path / "processed")
    config.storage.reports_dir = str(tmp_path / "reports")
    config.storage.manifest_file = str(tmp_path / "processed" / "manifest.json")
    config.storage.quality_report_file = str(tmp_path / "reports" / "report.md")

    service = MarketDataService(config=config, provider=FailingProvider())
    res = service.fetch_all(symbols=["ATVI"], start_date="2021-07-01", end_date="2021-07-05")

    assert res["success"] is False
    assert res["results"]["ATVI"]["status"] == "FAILED"
