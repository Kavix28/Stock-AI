import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock
import pandas as pd

from stock_ai.config.settings import load_config
from stock_ai.data.market.service import MarketDataService
from stock_ai.data.market.base import MarketDataProvider


class DummyMarketDataProvider(MarketDataProvider):
    """Deterministic fixture provider for integration testing."""

    def get_provider_name(self) -> str:
        return "DummyTestProvider"

    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        dates = ["2021-07-01", "2021-07-02", "2021-07-06"]
        data = {
            "symbol": [symbol] * 3,
            "date": dates,
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [104.0, 105.0, 106.0],
            "adjclose": [103.5, 104.5, 105.5],
            "volume": [10000, 12000, 11000]
        }
        return pd.DataFrame(data)


def test_integration_market_pipeline(tmp_path):
    # Setup temporary directory structure
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    manifest_file = processed_dir / "manifest.json"
    quality_report_file = reports_dir / "quality_report.md"

    config = load_config()
    config.storage.raw_market_dir = str(raw_dir)
    config.storage.processed_market_dir = str(processed_dir)
    config.storage.reports_dir = str(reports_dir)
    config.storage.manifest_file = str(manifest_file)
    config.storage.quality_report_file = str(quality_report_file)

    dummy_provider = DummyMarketDataProvider()
    service = MarketDataService(config=config, provider=dummy_provider)

    # Execute Pipeline for AAPL
    result = service.fetch_all(
        symbols=["AAPL"],
        start_date="2021-07-01",
        end_date="2021-07-06",
        force=True
    )

    assert result["success"] is True
    assert "AAPL" in result["results"]
    assert result["results"]["AAPL"]["status"] == "PASS"

    # Verify Files Created
    raw_file = raw_dir / "AAPL.csv"
    processed_parquet = processed_dir / "AAPL.parquet"
    
    assert raw_file.exists()
    assert processed_parquet.exists()
    assert manifest_file.exists()
    assert quality_report_file.exists()

    # Verify Manifest Contents
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["dataset"] == "paper_market_data"
    assert manifest["overall_status"] == "PASS"
    assert manifest["symbols"]["AAPL"]["rows"] == 3

    # Verify Quality Report Contents
    report_text = quality_report_file.read_text(encoding="utf-8")
    assert "AAPL" in report_text
    assert "PASS" in report_text
