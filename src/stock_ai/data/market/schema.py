from typing import Dict, List, Any
import pandas as pd
from pydantic import BaseModel, Field


CANONICAL_COLUMNS = ["symbol", "date", "open", "high", "low", "adj_close", "volume"]


class CanonicalMarketSchema:
    """Normalizes provider response dataframes into the canonical schema."""

    DEFAULT_MAPPING = {
        "symbol": "symbol",
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adjclose": "adj_close",
        "adj_close": "adj_close",
        "volume": "volume"
    }

    @classmethod
    def normalize(cls, raw_df: pd.DataFrame, symbol: str, column_mapping: Dict[str, str] = None) -> pd.DataFrame:
        """
        Convert raw provider dataframe to canonical schema.
        
        Canonical Columns:
            symbol (str)
            date (YYYY-MM-DD)
            open (float)
            high (float)
            low (float)
            adj_close (float)
            volume (float/int)
        """
        df = raw_df.copy()

        # Handle index if symbol/date are in index (common in YahooQuery)
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        elif "date" not in df.columns and "symbol" not in df.columns and df.index.name == "date":
            df = df.reset_index()

        # Normalize column names to lowercase
        df.columns = [str(col).lower().strip() for col in df.columns]

        mapping = {**cls.DEFAULT_MAPPING, **(column_mapping or {})}
        df = df.rename(columns=mapping)

        if "symbol" not in df.columns:
            df["symbol"] = symbol
        else:
            df["symbol"] = df["symbol"].fillna(symbol).astype(str)

        # Ensure date is string YYYY-MM-DD
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        # Required columns check
        missing = [col for col in CANONICAL_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Normalized DataFrame for '{symbol}' missing required canonical columns: {missing}")

        # Reorder and cast
        df = df[CANONICAL_COLUMNS].copy()
        
        numeric_cols = ["open", "high", "low", "adj_close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values(by="date").reset_index(drop=True)
        return df
