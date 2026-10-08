import pytest
import pandas as pd
from stock_ai.data.market.validator import MarketDataValidator


def test_validation_clean_data(sample_canonical_df):
    validator = MarketDataValidator(expected_symbol="AAPL")
    is_valid, issues, cleaned_df = validator.validate(sample_canonical_df)

    assert is_valid is True
    assert not any(i.level == "ERROR" for i in issues)
    assert len(cleaned_df) == len(sample_canonical_df)


def test_validation_high_less_than_low(sample_canonical_df):
    bad_df = sample_canonical_df.copy()
    bad_df.loc[0, "high"] = 100.0
    bad_df.loc[0, "low"] = 150.0  # High < Low

    validator = MarketDataValidator(expected_symbol="AAPL")
    is_valid, issues, _ = validator.validate(bad_df)

    assert is_valid is False
    assert any(i.level == "ERROR" and "High price strictly less than Low" in i.message for i in issues)


def test_validation_duplicate_dates(sample_canonical_df):
    dup_df = pd.concat([sample_canonical_df, sample_canonical_df.iloc[[0]]]).reset_index(drop=True)
    validator = MarketDataValidator(expected_symbol="AAPL")
    is_valid, issues, cleaned_df = validator.validate(dup_df)

    assert is_valid is True  # Identical duplicates deduplicated automatically
    assert len(cleaned_df) == len(sample_canonical_df)
    assert any(i.level == "WARNING" and "Identical duplicate" in i.message for i in issues)


def test_validation_missing_values(sample_canonical_df):
    null_df = sample_canonical_df.copy()
    null_df.loc[0, "adj_close"] = None

    validator = MarketDataValidator(expected_symbol="AAPL")
    is_valid, issues, _ = validator.validate(null_df)

    assert is_valid is False
    assert any(i.level == "ERROR" and "Missing/NaN values" in i.message for i in issues)
