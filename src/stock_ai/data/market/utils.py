from typing import List, Dict, Tuple, Any
import pandas as pd
import numpy as np


class TemporalNormalizer:
    """Utility for financial time-series date normalization and gap analysis."""

    @staticmethod
    def normalize_dates(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
        """
        Normalize date values in dataframe to naive YYYY-MM-DD format,
        ensuring proper chronological sorting and single observation per symbol/date.
        """
        df = df.copy()

        # Convert to datetime and strip timezone info
        df[date_column] = pd.to_datetime(df[date_column]).dt.tz_localize(None).dt.floor("D")
        
        # Sort chronologically
        df = df.sort_values(by=date_column).reset_index(drop=True)
        return df

    @staticmethod
    def analyze_trading_gaps(
        df: pd.DataFrame, date_column: str = "date"
    ) -> Dict[str, Any]:
        """
        Analyze gaps between consecutive date observations.
        Distinguishes EXPECTED NON-TRADING DAYS (weekends and market holidays like Thanksgiving, MLK Day, Good Friday)
        from UNEXPECTED DATA GAPS.

        Returns metadata summary.
        """
        if df.empty or len(df) < 2:
            return {
                "total_gaps": 0,
                "weekend_gaps": 0,
                "unexpected_gaps": [],
                "unexpected_gap_count": 0
            }

        dates = pd.to_datetime(df[date_column]).sort_values().reset_index(drop=True)
        date_diffs = dates.diff().dt.days

        unexpected_gaps = []
        weekend_gaps_count = 0

        for i in range(1, len(dates)):
            gap_days = int(date_diffs.iloc[i])
            if gap_days > 1:
                prev_date = dates.iloc[i - 1]
                curr_date = dates.iloc[i]
                
                # Check if gap is a weekend or mid-week holiday (e.g. Wednesday to Friday for Thanksgiving)
                is_expected_gap = (
                    (prev_date.dayofweek == 4 and curr_date.dayofweek == 0 and gap_days == 3) or  # Friday to Monday
                    (prev_date.dayofweek == 4 and curr_date.dayofweek == 1 and gap_days == 4) or  # Friday to Tuesday (Monday holiday)
                    (prev_date.dayofweek == 3 and curr_date.dayofweek == 0 and gap_days == 4) or  # Thursday to Monday (Good Friday)
                    (prev_date.dayofweek == 3 and curr_date.dayofweek == 1 and gap_days == 5) or  # Thursday to Tuesday
                    (prev_date.dayofweek == 2 and curr_date.dayofweek == 4 and gap_days == 2)     # Wednesday to Friday (Thanksgiving Thursday)
                )

                if is_expected_gap:
                    weekend_gaps_count += 1
                else:
                    unexpected_gaps.append({
                        "from": prev_date.strftime("%Y-%m-%d"),
                        "to": curr_date.strftime("%Y-%m-%d"),
                        "days_gap": gap_days
                    })

        return {
            "total_gaps": len(unexpected_gaps) + weekend_gaps_count,
            "weekend_gaps": weekend_gaps_count,
            "unexpected_gaps": unexpected_gaps,
            "unexpected_gap_count": len(unexpected_gaps)
        }
