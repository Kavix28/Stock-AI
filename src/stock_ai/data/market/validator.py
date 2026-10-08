from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

from stock_ai.data.market.utils import TemporalNormalizer
from stock_ai.logger import get_logger

logger = get_logger(__name__)


class ValidationIssue:
    def __init__(self, level: str, category: str, message: str, count: int = 1):
        self.level = level  # INFO, WARNING, ERROR
        self.category = category  # Structural, Temporal, Numerical, Financial
        self.message = message
        self.count = count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "category": self.category,
            "message": self.message,
            "count": self.count,
        }


class MarketDataValidator:
    """Performs comprehensive data quality checks on normalized market data."""

    def __init__(self, expected_symbol: str = None):
        self.expected_symbol = expected_symbol

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[ValidationIssue], pd.DataFrame]:
        """
        Validate dataset against structural, temporal, numerical, and financial rules.

        Returns:
            (is_valid: bool, issues: List[ValidationIssue], cleaned_df: pd.DataFrame)
        """
        issues: List[ValidationIssue] = []
        cleaned_df = df.copy()

        # 1. Structural Checks
        required_cols = ["symbol", "date", "open", "high", "low", "adj_close", "volume"]
        missing_cols = [c for c in required_cols if c not in cleaned_df.columns]
        if missing_cols:
            issues.append(ValidationIssue("ERROR", "Structural", f"Missing columns: {missing_cols}"))
            return False, issues, cleaned_df

        if self.expected_symbol:
            inconsistent_symbols = cleaned_df[cleaned_df["symbol"] != self.expected_symbol]
            if not inconsistent_symbols.empty:
                issues.append(
                    ValidationIssue(
                        "ERROR",
                        "Structural",
                        f"Inconsistent symbols found. Expected '{self.expected_symbol}', found {inconsistent_symbols['symbol'].unique().tolist()}",
                        count=len(inconsistent_symbols),
                    )
                )

        # 2. Temporal Checks & Duplicate Detection
        dup_mask = cleaned_df.duplicated(subset=["symbol", "date"], keep=False)
        if dup_mask.any():
            dup_rows = cleaned_df[dup_mask]
            # Check if duplicates are identical
            unique_dups = dup_rows.drop_duplicates()
            if len(unique_dups) == len(dup_rows.groupby(["symbol", "date"])):
                issues.append(
                    ValidationIssue(
                        "WARNING",
                        "Temporal",
                        "Identical duplicate (symbol, date) rows detected. Deduplicating automatically.",
                        count=len(dup_rows),
                    )
                )
                cleaned_df = cleaned_df.drop_duplicates(subset=["symbol", "date"], keep="first").reset_index(drop=True)
            else:
                issues.append(
                    ValidationIssue(
                        "ERROR",
                        "Temporal",
                        "Conflicting duplicate (symbol, date) rows detected.",
                        count=len(dup_rows),
                    )
                )

        # Check date sorting
        if not cleaned_df["date"].is_monotonic_increasing:
            issues.append(ValidationIssue("WARNING", "Temporal", "Dates were not in chronological order. Sorting automatically."))
            cleaned_df = cleaned_df.sort_values(by="date").reset_index(drop=True)

        # Trading Gaps Analysis
        gap_report = TemporalNormalizer.analyze_trading_gaps(cleaned_df)
        if gap_report["unexpected_gap_count"] > 0:
            issues.append(
                ValidationIssue(
                    "WARNING",
                    "Temporal",
                    f"Found {gap_report['unexpected_gap_count']} unexpected trading day gap(s): {gap_report['unexpected_gaps'][:3]}",
                    count=gap_report["unexpected_gap_count"],
                )
            )
        else:
            issues.append(
                ValidationIssue(
                    "INFO",
                    "Temporal",
                    f"All date gaps align with expected weekend non-trading days ({gap_report['weekend_gaps']} weekends).",
                )
            )

        # 3. Numerical & Missing Value Checks
        null_counts = cleaned_df.isnull().sum()
        total_nulls = null_counts.sum()
        if total_nulls > 0:
            null_cols = null_counts[null_counts > 0].to_dict()
            issues.append(ValidationIssue("ERROR", "Numerical", f"Missing/NaN values detected: {null_cols}", count=int(total_nulls)))

        # Inf values check
        numeric_cols = ["open", "high", "low", "adj_close", "volume"]
        inf_mask = np.isinf(cleaned_df[numeric_cols].to_numpy())
        if np.any(inf_mask):
            issues.append(ValidationIssue("ERROR", "Numerical", "Infinite values detected in numerical columns.", count=int(np.sum(inf_mask))))

        # Negative price/volume check
        price_cols = ["open", "high", "low", "adj_close"]
        negative_prices = (cleaned_df[price_cols] < 0).sum().sum()
        if negative_prices > 0:
            issues.append(ValidationIssue("ERROR", "Numerical", "Negative price values detected.", count=int(negative_prices)))

        negative_volume = (cleaned_df["volume"] < 0).sum()
        if negative_volume > 0:
            issues.append(ValidationIssue("ERROR", "Numerical", "Negative volume values detected.", count=int(negative_volume)))

        # 4. Financial Semantics / OHLC Bounds
        invalid_hl = (cleaned_df["high"] < cleaned_df["low"]).sum()
        if invalid_hl > 0:
            issues.append(ValidationIssue("ERROR", "Financial", "High price strictly less than Low price detected.", count=int(invalid_hl)))

        invalid_open_high = (cleaned_df["open"] > cleaned_df["high"]).sum()
        if invalid_open_high > 0:
            issues.append(ValidationIssue("WARNING", "Financial", "Open price greater than High price detected.", count=int(invalid_open_high)))

        invalid_open_low = (cleaned_df["open"] < cleaned_df["low"]).sum()
        if invalid_open_low > 0:
            issues.append(ValidationIssue("WARNING", "Financial", "Open price less than Low price detected.", count=int(invalid_open_low)))

        # High/Low vs Adj Close check
        # Note: Adjusted Close modifies historical prices for splits/dividends. If all columns are adjusted, High >= Adj Close >= Low.
        # We flag as WARNING if Adj Close severely violates bounds (e.g. > 10x High or < 0.1x Low).
        severe_adj_high = (cleaned_df["adj_close"] > cleaned_df["high"] * 5.0).sum()
        severe_adj_low = (cleaned_df["adj_close"] < cleaned_df["low"] * 0.2).sum()
        if severe_adj_high + severe_adj_low > 0:
            issues.append(ValidationIssue("WARNING", "Financial", "Adjusted Close significantly deviates from raw OHLC range.", count=int(severe_adj_high + severe_adj_low)))

        has_errors = any(issue.level == "ERROR" for issue in issues)
        is_valid = not has_errors

        return is_valid, issues, cleaned_df
