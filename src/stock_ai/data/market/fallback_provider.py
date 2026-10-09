import os
from pathlib import Path
import pandas as pd
from typing import Dict, Optional

from stock_ai.data.market.base import MarketDataProvider
from stock_ai.logger import get_logger

logger = get_logger(__name__)


class LocalCsvMarketDataProvider(MarketDataProvider):
    """
    Market data provider that loads historical data from local CSV archives.
    Used for delisted assets or offline historical data sources.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def get_provider_name(self) -> str:
        return "LocalCSVArchive"

    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        file_path = self.data_dir / f"{symbol}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Local market data file not found for symbol '{symbol}' at '{file_path}'")

        logger.info(f"Loading local market data for '{symbol}' from '{file_path}'...")
        df = pd.read_csv(file_path)
        
        # Ensure date column is present and filter date range
        date_col = None
        for c in ["date", "Date", "DATE"]:
            if c in df.columns:
                date_col = c
                break
        
        if not date_col:
            raise ValueError(f"No date column found in local file '{file_path}'")

        df[date_col] = pd.to_datetime(df[date_col])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        filtered_df = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)].copy()
        if filtered_df.empty:
            raise ValueError(f"Local file '{file_path}' has no observations in range {start_date} to {end_date}")

        filtered_df[date_col] = filtered_df[date_col].dt.strftime("%Y-%m-%d")
        return filtered_df


class FallbackMarketDataProvider(MarketDataProvider):
    """
    Composite provider that tries a primary provider first (e.g. YahooQuery),
    and falls back to secondary/local provider if primary provider fails.
    """

    def __init__(self, primary_provider: MarketDataProvider, secondary_provider: Optional[MarketDataProvider] = None):
        self.primary_provider = primary_provider
        self.secondary_provider = secondary_provider
        self.last_provider_used: Dict[str, str] = {}

    def get_provider_name(self) -> str:
        return f"{self.primary_provider.get_provider_name()} (with Fallback)"

    def get_provider_for_symbol(self, symbol: str) -> str:
        return self.last_provider_used.get(symbol, self.primary_provider.get_provider_name())

    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            df = self.primary_provider.fetch_historical_data(symbol, start_date, end_date)
            if not df.empty:
                self.last_provider_used[symbol] = self.primary_provider.get_provider_name()
                return df
            raise ValueError(f"Primary provider returned empty data for '{symbol}'")
        except Exception as e:
            logger.warning(f"Primary provider '{self.primary_provider.get_provider_name()}' failed for '{symbol}': {e}")
            if self.secondary_provider:
                logger.info(f"Attempting secondary provider '{self.secondary_provider.get_provider_name()}' for '{symbol}'...")
                df = self.secondary_provider.fetch_historical_data(symbol, start_date, end_date)
                self.last_provider_used[symbol] = self.secondary_provider.get_provider_name()
                return df
            raise e
