"""Regression tests confirming benchmark manifest integrity and consistency."""
import os
import unittest
import pandas as pd


class TestBenchmarkManifests(unittest.TestCase):
    def setUp(self):
        self.results_dir = "results/latest"

    def test_metrics_by_ticker_present(self):
        fpath = os.path.join(self.results_dir, "metrics_by_ticker.csv")
        self.assertTrue(os.path.exists(fpath), f"Missing {fpath}")
        df = pd.read_csv(fpath)
        self.assertGreater(len(df), 20)
        self.assertIn("accuracy", df.columns)
        self.assertIn("macro_f1", df.columns)
        self.assertIn("ticker", df.columns)

    def test_pooled_metrics_present(self):
        fpath = os.path.join(self.results_dir, "pooled_metrics.csv")
        self.assertTrue(os.path.exists(fpath), f"Missing {fpath}")
        df = pd.read_csv(fpath)
        self.assertEqual(len(df), 1)
        self.assertGreater(df.iloc[0]["total_test_observations"], 6000)

    def test_all_12_ticker_predictions_exist(self):
        pred_dir = os.path.join(self.results_dir, "predictions")
        self.assertTrue(os.path.exists(pred_dir))
        for ticker in ["AAPL", "AMT", "ATVI", "BIO", "JNJ", "NDAQ", "NFLX", "NVDA", "PLD", "SBUX", "SCHW", "TSLA"]:
            pfile = os.path.join(pred_dir, f"{ticker}_predictions.csv")
            self.assertTrue(os.path.exists(pfile), f"Missing prediction file for {ticker}")


if __name__ == "__main__":
    unittest.main()
