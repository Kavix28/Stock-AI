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
        self.assertIn("multimodal_architecture", data)
        self.assertIn("atvi_status", data)

    def test_tickers_endpoint(self):
        resp = self.client.get("/api/v1/tickers")
        self.assertEqual(resp.status_code, 200)
        tickers = resp.json()
        self.assertEqual(len(tickers), 12)
        symbols = [t["symbol"] for t in tickers]
        self.assertIn("AAPL", symbols)
        self.assertIn("ATVI", symbols)

        # Verify ATVI is explicitly flagged as delisted
        atvi = next(t for t in tickers if t["symbol"] == "ATVI")
        self.assertTrue(atvi["delisted"])

    def test_ticker_history_endpoint(self):
        resp = self.client.get("/api/v1/tickers/AAPL/history?limit=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["ticker"], "AAPL")
        self.assertEqual(len(data["data"]), 10)

    def test_ticker_news_endpoint(self):
        resp = self.client.get("/api/v1/tickers/AAPL/news?limit=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["ticker"], "AAPL")
        self.assertIn("timing_status", data)
        self.assertIn("news", data)

    def test_metrics_endpoint(self):
        resp = self.client.get("/api/v1/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("models", data)

    def test_multimodal_metrics_endpoint(self):
        resp = self.client.get("/api/v1/metrics/multimodal")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("models", data)
        self.assertGreater(len(data["models"]), 0)

    def test_predict_endpoint_gmu(self):
        resp = self.client.post("/api/v1/predict", json={"ticker": "AAPL", "model_type": "gmu_fusion"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["ticker"], "AAPL")
        self.assertIn(data["predicted_class"], ["Up", "Down"])
        self.assertIn("disclaimer", data)
        self.assertIn("market_gate_weight", data)
        self.assertIn("news_gate_weight", data)
        self.assertEqual(data["model_version"], "GMU-Fusion-v1.0-PyTorch")

    def test_predict_endpoint_bilstm(self):
        resp = self.client.post("/api/v1/predict", json={"ticker": "AAPL", "model_type": "market_bilstm"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["ticker"], "AAPL")
        self.assertIn(data["predicted_class"], ["Up", "Down"])
        self.assertEqual(data["model_version"], "Market-BiLSTM-v1.0-PyTorch")
        self.assertEqual(data["market_gate_weight"], 1.0)
        self.assertEqual(data["news_gate_weight"], 0.0)


if __name__ == "__main__":
    unittest.main()
