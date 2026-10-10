"""Unit tests for technical indicator engineering."""
import unittest
import numpy as np
import pandas as pd
from stock_ai.features.indicators import compute_indicators


class TestIndicators(unittest.TestCase):
    def test_indicator_computation(self):
        dates = pd.date_range("2021-01-01", periods=100, freq="B")
        df = pd.DataFrame({
            "date": dates,
            "open": np.linspace(100, 150, 100),
            "high": np.linspace(102, 155, 100),
            "low": np.linspace(98, 148, 100),
            "close": np.linspace(101, 152, 100),
            "adjclose": np.linspace(101, 152, 100),
            "volume": np.random.randint(1000, 50000, 100)
        })
        res = compute_indicators(df)
        self.assertIn("12-Day EMA", res.columns)
        self.assertIn("26-Day EMA", res.columns)
        self.assertIn("MACD", res.columns)
        self.assertIn("rsi", res.columns)
        self.assertIn("Upper Band", res.columns)
        self.assertIn("Middle Band", res.columns)
        self.assertIn("Lower Band", res.columns)
        self.assertIn("SAR", res.columns)
        self.assertIn("Slow k", res.columns)
        self.assertIn("Slow d", res.columns)
        self.assertIn("Diff", res.columns)
        self.assertEqual(res.isnull().sum().sum(), 0)


if __name__ == "__main__":
    unittest.main()
