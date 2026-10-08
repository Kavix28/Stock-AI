from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class MarketDataProvider(ABC):
    """Abstract base class for financial market data providers."""

    @abstractmethod
    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch raw historical daily market data for a given symbol within date range.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)

        Returns:
            pd.DataFrame: Un-normalized raw DataFrame returned by provider.
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the market data provider."""
        pass
