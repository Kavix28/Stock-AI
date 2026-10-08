from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
from pydantic import BaseModel, Field


class StockConfig(BaseModel):
    symbol: str
    name: str
    sector: str
    industry: str


class EvaluationPeriod(BaseModel):
    start_date: str
    end_date: str


class MarketDataConfig(BaseModel):
    provider: str = "yahooquery"
    frequency: str = "daily"
    acquisition_range: EvaluationPeriod
    evaluation_periods: Dict[str, EvaluationPeriod]
    required_columns: List[str]
    provider_column_mapping: Dict[str, str]
    retry_attempts: int = 3
    retry_backoff_factor: float = 1.5
    timeout: int = 30
    rate_limit_delay: float = 0.5


class StorageConfig(BaseModel):
    raw_market_dir: str = "data/raw/market"
    processed_market_dir: str = "data/processed/market"
    reports_dir: str = "reports"
    manifest_file: str = "data/processed/market/manifest.json"
    quality_report_file: str = "reports/market_data_quality_report.md"


class AppConfig(BaseModel):
    project_name: str = Field(default="multimodal-stock-ai", alias="project.name")
    market_data: MarketDataConfig
    storage: StorageConfig
    stocks: List[StockConfig] = []


def load_config(
    base_config_path: Path = Path("configs/base.yaml"),
    stocks_config_path: Path = Path("configs/stocks.yaml")
) -> AppConfig:
    """Load and merge project YAML configurations into a structured AppConfig object."""
    base_path = Path(base_config_path)
    stocks_path = Path(stocks_config_path)

    if not base_path.exists():
        raise FileNotFoundError(f"Base configuration file not found at {base_path}")
    if not stocks_path.exists():
        raise FileNotFoundError(f"Stocks configuration file not found at {stocks_path}")

    with open(base_path, "r", encoding="utf-8") as f:
        base_dict = yaml.safe_load(f)

    with open(stocks_path, "r", encoding="utf-8") as f:
        stocks_dict = yaml.safe_load(f)

    merged = {**base_dict, "stocks": stocks_dict.get("stocks", [])}
    return AppConfig(**merged)
