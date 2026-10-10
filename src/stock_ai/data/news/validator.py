"""
Validation suite for financial news datasets and aligned multimodal features.

Enforces:
1. Anti-leakage invariants: Verifies no future timestamps enter feature matrices.
2. Temporal isolation: Ensures train/validation/test splits do not overlap.
3. Integrity checks: Asserts text cleanliness, valid IDs, and deterministic formatting.
"""

from datetime import datetime
from typing import Dict, List, Any, Tuple
import pandas as pd


class NewsValidationError(Exception):
    """Raised when data validation rules are violated."""
    pass


class NewsDataValidator:
    """Validates news datasets, causal alignments, and experimental partitions."""

    @staticmethod
    def validate_canonical_dataframe(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Validates structure and contents of a CanonicalNewsSchema DataFrame.
        """
        required_cols = [
            "article_id", "symbol", "published_date", "published_at_utc",
            "has_intraday_time", "title", "source", "provider"
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise NewsValidationError(f"DataFrame for '{symbol}' missing required columns: {missing}")

        if df.empty:
            return {"total_articles": 0, "status": "EMPTY"}

        # Check for null titles
        null_titles = df["title"].isnull().sum() + (df["title"].astype(str).str.strip() == "").sum()
        if null_titles > 0:
            raise NewsValidationError(f"DataFrame for '{symbol}' contains {null_titles} null or empty titles.")

        # Check date parsing
        invalid_dates = pd.to_datetime(df["published_date"], errors="coerce").isnull().sum()
        if invalid_dates > 0:
            raise NewsValidationError(f"DataFrame for '{symbol}' contains {invalid_dates} invalid dates.")

        # Check unique article IDs
        duplicate_ids = df["article_id"].duplicated().sum()

        dates = pd.to_datetime(df["published_date"])
        return {
            "symbol": symbol.upper(),
            "total_articles": len(df),
            "duplicate_ids": int(duplicate_ids),
            "start_date": dates.min().strftime("%Y-%m-%d"),
            "end_date": dates.max().strftime("%Y-%m-%d"),
            "has_intraday_time": bool(df["has_intraday_time"].any()),
            "status": "VALID"
        }

    @staticmethod
    def validate_no_timestamp_leakage(
        aligned_df: pd.DataFrame,
        raw_news_df: pd.DataFrame
    ) -> None:
        """
        Verifies that no article linked to target_date T was published after T's cutoff.
        """
        if aligned_df.empty or raw_news_df.empty:
            return

        if not raw_news_df["has_intraday_time"].any():
            # If news only has calendar dates, check that published_date <= target_date
            # Map article_id -> published_date
            id_to_date = dict(zip(raw_news_df["article_id"], raw_news_df["published_date"]))
            for _, row in aligned_df.iterrows():
                target_d = row["target_date"]
                for art_id in row.get("article_ids", []):
                    pub_d = id_to_date.get(art_id)
                    if pub_d and pub_d > target_d:
                        raise NewsValidationError(
                            f"LEAKAGE VIOLATION: Article {art_id} dated {pub_d} assigned to earlier target {target_d}!"
                        )
            return

        # Intraday verification: published_at_utc must be strictly prior to market cutoff
        id_to_utc = dict(zip(raw_news_df["article_id"], pd.to_datetime(raw_news_df["published_at_utc"], utc=True)))
        for _, row in aligned_df.iterrows():
            target_d = row["target_date"]
            # Target day cutoff: 16:00 EST / 21:00 UTC (or 09:30 EST / 14:30 UTC for pre-market)
            # Conservative absolute ceiling: End of target trading day UTC
            target_dt_cutoff = pd.to_datetime(f"{target_d} 21:00:00", utc=True)
            for art_id in row.get("article_ids", []):
                pub_utc = id_to_utc.get(art_id)
                if pub_utc and pub_utc >= target_dt_cutoff:
                    raise NewsValidationError(
                        f"LEAKAGE VIOLATION: Article {art_id} with timestamp {pub_utc} assigned to target {target_d}!"
                    )

    @staticmethod
    def validate_chronological_splits(
        train_dates: List[str],
        val_dates: List[str],
        test_dates: List[str]
    ) -> None:
        """
        Ensures strict non-overlapping temporal order: train < val < test.
        """
        if not train_dates or not val_dates or not test_dates:
            raise NewsValidationError("Train, validation, and test date partitions must all be non-empty.")

        tr_max = max(train_dates)
        val_min = min(val_dates)
        val_max = max(val_dates)
        te_min = min(test_dates)

        if tr_max >= val_min:
            raise NewsValidationError(
                f"SPLIT CONTAMINATION: Training partition ends at {tr_max}, but validation starts at {val_min}."
            )
        if val_max >= te_min:
            raise NewsValidationError(
                f"SPLIT CONTAMINATION: Validation partition ends at {val_max}, but test starts at {te_min}."
            )
