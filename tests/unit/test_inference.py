"""Unit tests for inference engine."""
import unittest
from stock_ai.inference.engine import InferenceEngine


class TestInference(unittest.TestCase):
    def setUp(self):
        self.engine = InferenceEngine(
            data_dir="data/processed/market",
            results_dir="results/latest"
        )

    def test_supported_tickers(self):
        tickers = self.engine.get_supported_tickers()
        self.assertEqual(len(tickers), 12)
        self.assertIn("AAPL", tickers)
        self.assertIn("ATVI", tickers)

    def test_get_prediction_structure(self):
        pred = self.engine.get_prediction("AAPL")
        self.assertEqual(pred["ticker"], "AAPL")
        self.assertIn("predicted_class", pred)
        self.assertIn("probability", pred)
        self.assertIn("disclaimer", pred)


if __name__ == "__main__":
    unittest.main()
