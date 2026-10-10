"""
Polygon.io news provider implementation.
"""

import os
from typing import Dict, List, Any
import pandas as pd
import requests

from .base import (
    NewsDataProvider,
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderDataError
)
from .schema import CanonicalNewsSchema


class PolygonNewsProvider(NewsDataProvider):
    """
    Fetches news from Polygon.io Ticker News REST API (/v2/reference/news).
    
    Provides second-level ISO-8601 UTC timestamps (e.g. '2024-08-01T14:32:00Z').
    Requires POLYGON_API_KEY environment variable.
    """

    BASE_URL = "https://api.polygon.io/v2/reference/news"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")

    def get_provider_name(self) -> str:
        return "polygon_io"

    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def get_required_credentials(self) -> List[str]:
        return ["POLYGON_API_KEY"]

    def get_provider_metadata(self) -> Dict[str, Any]:
        return {
            "provider_name": self.get_provider_name(),
            "data_source": "Polygon.io Financial News API",
            "licensing": "Commercial Developer API Subscription",
            "timestamp_quality": "POINT-IN-TIME CERTIFIED (Second-level UTC)",
            "intraday_precision": True,
            "historical_range": "2020-present (Tier-dependent)",
            "legal_usability": "Requires paid/developer API subscription; terms forbid public redistribution"
        }

    def fetch_news(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        if not self.is_configured():
            raise ProviderConfigurationError(
                "PolygonNewsProvider is not configured. "
                "Please set the 'POLYGON_API_KEY' environment variable. "
                "Example: export POLYGON_API_KEY='your_api_key'"
            )

        params = {
            "ticker": symbol.upper(),
            "published_utc.gte": f"{start_date}T00:00:00Z",
            "published_utc.lte": f"{end_date}T23:59:59Z",
            "limit": 1000,
            "apiKey": self.api_key
        }

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=15)
        except requests.RequestException as e:
            raise ProviderDataError(f"Polygon network request failed: {e}")

        if resp.status_code == 429:
            raise ProviderRateLimitError("Polygon API rate limit exceeded (HTTP 429).")
        if resp.status_code == 401 or resp.status_code == 403:
            raise ProviderConfigurationError("Polygon API key is invalid or unauthorized (HTTP 401/403).")
        if resp.status_code != 200:
            raise ProviderDataError(f"Polygon API returned HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return pd.DataFrame()

        rows = []
        for item in results:
            rows.append({
                "title": item.get("title", ""),
                "date": item.get("published_utc", ""),
                "source": item.get("publisher", {}).get("name", "Polygon"),
                "url": item.get("article_url", "")
            })

        raw_df = pd.DataFrame(rows)
        return CanonicalNewsSchema.normalize_dataframe(
            raw_df,
            symbol=symbol,
            provider=self.get_provider_name(),
            default_source="Polygon.io"
        )
