"""
Tiingo news provider implementation.
"""

import os
from typing import Dict, List, Any
import pandas as pd
import requests

from src.stock_ai.data.news.base import (
    NewsDataProvider,
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderDataError
)
from src.stock_ai.data.news.schema import CanonicalNewsSchema


class TiingoNewsProvider(NewsDataProvider):
    """
    Fetches news from Tiingo News REST API (/tiingo/news).
    
    Provides ISO-8601 UTC publication timestamps.
    Requires TIINGO_API_KEY environment variable.
    """

    BASE_URL = "https://api.tiingo.com/tiingo/news"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("TIINGO_API_KEY")

    def get_provider_name(self) -> str:
        return "tiingo_news"

    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def get_required_credentials(self) -> List[str]:
        return ["TIINGO_API_KEY"]

    def get_provider_metadata(self) -> Dict[str, Any]:
        return {
            "provider_name": self.get_provider_name(),
            "data_source": "Tiingo Financial News Feed",
            "licensing": "Commercial / Individual API Key Subscription",
            "timestamp_quality": "POINT-IN-TIME CERTIFIED (Second-level UTC)",
            "intraday_precision": True,
            "historical_range": "Multi-year historical coverage",
            "legal_usability": "Requires valid Tiingo API token; terms govern data redistribution"
        }

    def fetch_news(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        if not self.is_configured():
            raise ProviderConfigurationError(
                "TiingoNewsProvider is not configured. "
                "Please set the 'TIINGO_API_KEY' environment variable. "
                "Example: export TIINGO_API_KEY='your_api_key'"
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {self.api_key}"
        }
        params = {
            "tickers": symbol.upper(),
            "startDate": start_date,
            "endDate": end_date,
            "limit": 1000
        }

        try:
            resp = requests.get(self.BASE_URL, headers=headers, params=params, timeout=15)
        except requests.RequestException as e:
            raise ProviderDataError(f"Tiingo network request failed: {e}")

        if resp.status_code == 429:
            raise ProviderRateLimitError("Tiingo API rate limit exceeded (HTTP 429).")
        if resp.status_code in [401, 403]:
            raise ProviderConfigurationError("Tiingo API key is invalid or unauthorized (HTTP 401/403).")
        if resp.status_code != 200:
            raise ProviderDataError(f"Tiingo API returned HTTP {resp.status_code}: {resp.text}")

        results = resp.json()
        if not results:
            return pd.DataFrame()

        rows = []
        for item in results:
            rows.append({
                "title": item.get("title", ""),
                "date": item.get("publishedDate", ""),
                "source": item.get("source", "Tiingo"),
                "url": item.get("url", "")
            })

        raw_df = pd.DataFrame(rows)
        return CanonicalNewsSchema.normalize_dataframe(
            raw_df,
            symbol=symbol,
            provider=self.get_provider_name(),
            default_source="Tiingo News"
        )
