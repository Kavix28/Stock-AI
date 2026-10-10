"""
Canonical schema for financial news data.

Provides:
- CanonicalNewsArticle: Individual news item with verified timestamps and source metadata.
- DailyNewsFeature: Aggregated daily news feature aligned to market prediction targets.
- CanonicalNewsSchema: Utility to validate and normalize raw news DataFrames.
"""

import hashlib
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
from pydantic import BaseModel, Field


CANONICAL_NEWS_COLUMNS = [
    "article_id",
    "symbol",
    "published_date",
    "published_at_utc",
    "has_intraday_time",
    "title",
    "source",
    "provider",
    "url"
]


class CanonicalNewsArticle(BaseModel):
    """Normalized single news article representation."""
    article_id: str
    symbol: str
    published_date: str
    published_at_utc: Optional[str] = None
    has_intraday_time: bool = False
    title: str
    source: str = "Unknown"
    provider: str = "canonical_local"
    url: Optional[str] = None


class DailyNewsFeature(BaseModel):
    """Daily news representation aligned to a market prediction session."""
    symbol: str
    target_date: str
    combined_text: str
    article_count: int
    has_news: bool
    timing_quality: str = Field(
        ...,
        description="'POINT_IN_TIME_CERTIFIED' if UTC intraday timestamps verified, 'TIMING_LIMITED_DAILY_ONLY' if calendar date only."
    )
    article_ids: List[str] = Field(default_factory=list)


class CanonicalNewsSchema:
    """Normalizes and validates raw news DataFrames."""

    @staticmethod
    def clean_headline_text(text: Any) -> str:
        """Sanitizes headline text: trims whitespace, removes non-printable chars."""
        if pd.isna(text) or text is None:
            return ""
        s = str(text).strip()
        # Collapse multiple spaces and newlines
        s = re.sub(r"\s+", " ", s)
        return s

    @classmethod
    def generate_article_id(cls, symbol: str, date_str: str, title: str) -> str:
        """Deterministic hash for article deduplication."""
        raw_key = f"{symbol.upper()}|{date_str}|{title.lower().strip()}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def normalize_dataframe(
        cls,
        raw_df: pd.DataFrame,
        symbol: str,
        provider: str = "canonical_local",
        default_source: str = "Unknown"
    ) -> pd.DataFrame:
        """
        Converts arbitrary news DataFrame into CanonicalNewsSchema format.
        
        Handles various column naming conventions (e.g. 'Titles', 'headline', 'title', 'date').
        """
        if raw_df.empty:
            return pd.DataFrame(columns=CANONICAL_NEWS_COLUMNS)

        df = raw_df.copy()
        col_lower = {str(c).lower().strip(): c for c in df.columns}

        # 1. Resolve Title column
        title_col = None
        for candidate in ["titles", "title", "headline", "news_title", "text"]:
            if candidate in col_lower:
                title_col = col_lower[candidate]
                break
        if title_col is None:
            raise ValueError(f"Raw news DataFrame for '{symbol}' missing title/headline column. Available: {list(df.columns)}")

        # 2. Resolve Date column
        date_col = None
        for candidate in ["date", "published_date", "datetime", "timestamp", "published_at"]:
            if candidate in col_lower:
                date_col = col_lower[candidate]
                break
        if date_col is None:
            raise ValueError(f"Raw news DataFrame for '{symbol}' missing date column. Available: {list(df.columns)}")

        # Clean text
        df["title"] = df[title_col].apply(cls.clean_headline_text)
        # Drop empty headlines
        df = df[df["title"] != ""].copy()

        # Parse date and check for intraday timestamps
        raw_dates = df[date_col].astype(str)
        has_intraday = raw_dates.str.contains(r":\d{2}").any()

        df["symbol"] = symbol.upper()
        df["provider"] = provider
        df["source"] = df.get("source", default_source)
        df["url"] = df.get("url", None)

        if has_intraday:
            parsed_utc = pd.to_datetime(raw_dates, errors="coerce", utc=True)
            df["published_at_utc"] = parsed_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            df["published_date"] = parsed_utc.dt.strftime("%Y-%m-%d")
            df["has_intraday_time"] = True
        else:
            parsed_date = pd.to_datetime(raw_dates, errors="coerce")
            df["published_at_utc"] = None
            df["published_date"] = parsed_date.dt.strftime("%Y-%m-%d")
            df["has_intraday_time"] = False

        # Drop invalid dates
        df = df.dropna(subset=["published_date"]).copy()

        # Generate deterministic article_id
        df["article_id"] = [
            cls.generate_article_id(symbol, d, t)
            for d, t in zip(df["published_date"], df["title"])
        ]

        # Reorder to canonical columns
        result_df = df[CANONICAL_NEWS_COLUMNS].copy()
        result_df = result_df.sort_values(by=["published_date", "article_id"]).reset_index(drop=True)
        return result_df
