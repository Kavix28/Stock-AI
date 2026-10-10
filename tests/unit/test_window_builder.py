"""Unit tests for chronological rolling window builder."""
import unittest
import numpy as np
import pandas as pd
from stock_ai.data.window_builder import build_chronological_windows, CORE_MARKET_FEATURES


class TestWindowBuilder(unittest.TestCase):
    def setUp(self):
        # Create synthetic stock data
        dates = pd.date_range("2022-01-01", periods=100, freq="B")
        data = {feat: np.random.uniform(10, 100, size=100) for feat in CORE_MARKET_FEATURES}
        data["date"] = dates
        data["close"] = np.linspace(50, 100, 100)
        data["PrevDayClose"] = np.roll(data["close"], 1)
        data["Diff"] = (data["close"] > data["PrevDayClose"]).astype(int)
        self.df = pd.DataFrame(data)

    def test_window_construction_and_lookback(self):
        w = build_chronological_windows(
            self.df,
            lookback=10,
            train_cutoff="2022-03-01",
            val_cutoff="2022-04-01"
        )
        self.assertIn("train", w)
        self.assertIn("val", w)
        self.assertIn("test", w)
        self.assertIn("scaler", w)
        
        # Test shape
        if len(w["test"]["X_seq"]) > 0:
            self.assertEqual(w["test"]["X_seq"].shape[1], 10)
            self.assertEqual(w["test"]["X_seq"].shape[2], 16)

    def test_scaler_isolation(self):
        w = build_chronological_windows(
            self.df,
            lookback=10,
            train_cutoff="2022-03-01",
            val_cutoff="2022-04-01"
        )
        # Scaler must have fit on 16 features
        self.assertEqual(len(w["scaler"].data_min_), 16)
        self.assertEqual(len(w["scaler"].data_max_), 16)


if __name__ == "__main__":
    unittest.main()
