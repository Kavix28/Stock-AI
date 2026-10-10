"""Unit tests for FastAPI backend endpoints."""
import os
import unittest
from fastapi.testclient import TestClient
from backend.main import app


class TestBackendAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")

    def test_metadata_endpoint(self):
        resp = self.client.get("/api/v1/metadata")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["project_name"], "Stock-AI")
        self.assertEqual(data["supported_tickers_count"], 12)

    def test_tickers_endpoint(self):
        resp = self.client.get("/api/v1/tickers")
        self.assertEqual(resp.status_code, 200)
        tickers = resp.json()
        self.assertEqual(len(tickers), 12)
        symbols = [t["symbol"] for t in tickers]
        self.assertIn("AAPL", symbols)
        self.assertIn("ATVI", symbols)

    def test_ticker_history_endpoint(self):
        resp = self.client.get("/api/v1/tickers/AAPL/history?limit=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["ticker"], "AAPL")
        self.assertEqual(len(data["data"]), 10)

    def test_metrics_endpoint(self):
        resp = self.client.get("/api/v1/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("model_summaries", data)
        self.assertIn("ticker_metrics", data)

    def test_predict_endpoint(self):
        resp = self.client.post("/api/v1/predict", json={"ticker": "AAPL"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["ticker"], "AAPL")
        self.assertIn(data["predicted_class"], ["Up", "Down"])
        self.assertIn("disclaimer", data)


if __name__ == "__main__":
    unittest.main()
