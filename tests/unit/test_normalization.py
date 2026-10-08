import pytest
import pandas as pd
from stock_ai.data.market.schema import CanonicalMarketSchema, CANONICAL_COLUMNS


def test_schema_normalization(sample_raw_yahooquery_df):
    df_norm = CanonicalMarketSchema.normalize(sample_raw_yahooquery_df, symbol="AAPL")

    assert list(df_norm.columns) == CANONICAL_COLUMNS
    assert len(df_norm) == 5
    assert (df_norm["symbol"] == "AAPL").all()
    assert "adj_close" in df_norm.columns
    assert df_norm["adj_close"].iloc[0] == 135.50


def test_schema_normalization_missing_column():
    bad_df = pd.DataFrame({"symbol": ["AAPL"], "date": ["2021-07-01"], "open": [100.0]})
    with pytest.raises(ValueError, match="missing required canonical columns"):
        CanonicalMarketSchema.normalize(bad_df, symbol="AAPL")
