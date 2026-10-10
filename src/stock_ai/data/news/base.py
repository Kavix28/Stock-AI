"""
Base interface and exception hierarchy for financial news data providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import pandas as pd


class NewsProviderError(Exception):
    """Base exception for news provider operations."""
    pass


class ProviderConfigurationError(NewsProviderError):
    """Raised when required API credentials or configurations are missing."""
    pass


class ProviderRateLimitError(NewsProviderError):
    """Raised when provider rate limits are exceeded."""
    pass


class ProviderDataError(NewsProviderError):
    """Raised when returned news data fails integrity or schema checks."""
    pass


class NewsDataProvider(ABC):
    """Abstract base class for financial news data providers."""

    @abstractmethod
    def fetch_news(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch news articles for a given symbol within a date range.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)

        Returns:
            pd.DataFrame: DataFrame normalized to CanonicalNewsSchema.
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return human-readable identifier of the provider."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check whether provider has valid credentials/paths to operate."""
        pass

    @abstractmethod
    def get_required_credentials(self) -> List[str]:
        """List of environment variable names or credentials needed."""
        pass

    @abstractmethod
    def get_provider_metadata(self) -> Dict[str, Any]:
        """
        Return metadata regarding licensing, timestamp precision,
        and legal usability.
        """
        pass
