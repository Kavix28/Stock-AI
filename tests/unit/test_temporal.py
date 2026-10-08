import pytest
import pandas as pd
from stock_ai.data.market.utils import TemporalNormalizer


def test_trading_gap_analysis_weekends():
    # Friday 2021-07-02 to Monday 2021-07-05 (3 days gap)
    df = pd.DataFrame({
        "date": ["2021-07-02", "2021-07-05"]
    })
    res = TemporalNormalizer.analyze_trading_gaps(df)
    assert res["unexpected_gap_count"] == 0
    assert res["weekend_gaps"] == 1


def test_trading_gap_analysis_unexpected():
    # Tuesday 2021-07-06 to Friday 2021-07-09 (missing Wed and Thu -> 3 days gap on weekdays)
    df = pd.DataFrame({
        "date": ["2021-07-06", "2021-07-09"]
    })
    res = TemporalNormalizer.analyze_trading_gaps(df)
    assert res["unexpected_gap_count"] == 1
    assert res["weekend_gaps"] == 0
