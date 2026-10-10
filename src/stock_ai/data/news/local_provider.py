"""
Local news data provider for loading verified research news archives.
"""

import os
import hashlib
from typing import Dict, List, Any
import pandas as pd

from .base import NewsDataProvider, ProviderDataError
from .schema import CanonicalNewsSchema


class LocalCanonicalNewsProvider(NewsDataProvider):
    """
    Loads verified historical news headlines from local disk (e.g. data/processed/news/).
    
    Preserves exact source filenames, hashes, and dates. Explicitly flags
    that daily-aggregated headlines lack intraday timestamps (TIMING-LIMITED).
    """

    def __init__(self, news_dir: str = "data/processed/news"):
        self.news_dir = news_dir

    def get_provider_name(self) -> str:
        return "local_canonical_archive"

    def is_configured(self) -> bool:
        return os.path.exists(self.news_dir) and len(os.listdir(self.news_dir)) > 0

    def get_required_credentials(self) -> List[str]:
        return []

    def get_provider_metadata(self) -> Dict[str, Any]:
        return {
            "provider_name": self.get_provider_name(),
            "data_source": "Chen et al. (2024) Capstone Research Archive",
            "licensing": "Academic Research Use",
            "timestamp_quality": "TIMING-LIMITED (Calendar Date YYYY-MM-DD Only)",
            "intraday_precision": False,
            "historical_range": "2021-01-05 to 2022-09-20",
            "legal_usability": "Defensible for historical replication; requires intraday timestamps for live trading"
        }

    def fetch_news(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Loads local news CSV for symbol, filters by date range, and returns CanonicalNewsSchema.
        """
        symbol_clean = symbol.upper().strip()
        fpath = os.path.join(self.news_dir, f"{symbol_clean}.csv")
        if not os.path.exists(fpath):
            raise ProviderDataError(f"Local news file not found for symbol '{symbol_clean}' at: {fpath}")

        try:
            raw_df = pd.read_csv(fpath)
        except Exception as e:
            raise ProviderDataError(f"Failed to read news file for '{symbol_clean}': {e}")

        # Compute checksum for provenance tracking
        with open(fpath, "rb") as f:
            file_sha256 = hashlib.sha256(f.read()).hexdigest()

        # Normalize via CanonicalNewsSchema
        normalized_df = CanonicalNewsSchema.normalize_dataframe(
            raw_df,
            symbol=symbol_clean,
            provider=self.get_provider_name(),
            default_source="Chen et al. 2024 Archive"
        )

        # Date filtering
        mask = (normalized_df["published_date"] >= start_date) & (normalized_df["published_date"] <= end_date)
        filtered_df = normalized_df[mask].copy().reset_index(drop=True)
        filtered_df.attrs["file_sha256"] = file_sha256
        filtered_df.attrs["source_path"] = fpath
        return filtered_df
