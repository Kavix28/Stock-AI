import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_raw_yahooquery_df():
    """Return a raw MultiIndex DataFrame structure matching YahooQuery history output."""
    dates = pd.date_range("2021-07-01", periods=5, freq="B")
    data = {
        "open": [136.60, 137.90, 140.00, 141.58, 142.50],
        "high": [137.97, 140.00, 143.00, 143.15, 144.50],
        "low": [135.80, 137.50, 139.50, 140.00, 142.00],
        "close": [137.27, 139.96, 142.02, 143.24, 144.30],
        "adjclose": [135.50, 138.10, 140.15, 141.35, 142.40],
        "volume": [96471800, 78857100, 105501300, 105494200, 91520000]
    }
    index = pd.MultiIndex.from_tuples(
        [("AAPL", d) for d in dates],
        names=["symbol", "date"]
    )
    return pd.DataFrame(data, index=index)


@pytest.fixture
def sample_canonical_df():
    """Return a clean normalized canonical DataFrame."""
    return pd.DataFrame({
        "symbol": ["AAPL"] * 5,
        "date": ["2021-07-01", "2021-07-02", "2021-07-06", "2021-07-07", "2021-07-08"],
        "open": [136.60, 137.90, 140.00, 141.58, 142.50],
        "high": [137.97, 140.00, 143.00, 143.15, 144.50],
        "low": [135.80, 137.50, 139.50, 140.00, 142.00],
        "adj_close": [135.50, 138.10, 140.15, 141.35, 142.40],
        "volume": [96471800, 78857100, 105501300, 105494200, 91520000]
    })
