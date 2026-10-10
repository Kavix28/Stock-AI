"""
Comprehensive standalone verification of FastAPI backend endpoints.
Validates status codes, schemas, error handling, unknown tickers, and prediction integrity.
"""
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "Stock-AI Backend"
    assert "version" in data

def test_metadata_endpoint():
    res = client.get("/api/v1/metadata")
    assert res.status_code == 200
    data = res.json()
    assert data["project_name"] == "Stock-AI"
    assert data["supported_tickers_count"] == 12
    assert data["features_count"] == 16
    assert data["lookback_window"] == 20
    assert data["pooled_test_observations"] == 6292

def test_tickers_endpoint():
    res = client.get("/api/v1/tickers")
    assert res.status_code == 200
    tickers = res.json()
    assert len(tickers) == 12
    symbols = [t["symbol"] for t in tickers]
    assert "AAPL" in symbols
    assert "NVDA" in symbols
    assert "ATVI" in symbols
    # ATVI should be marked delisted
    atvi_entry = [t for t in tickers if t["symbol"] == "ATVI"][0]
    assert atvi_entry["delisted"] is True

def test_history_endpoint():
    res = client.get("/api/v1/tickers/AAPL/history?limit=30")
    assert res.status_code == 200
    data = res.json()
    assert data["ticker"] == "AAPL"
    assert len(data["data"]) == 30
    assert "close" in data["data"][0]
    assert "date" in data["data"][0]

def test_history_unknown_ticker():
    res = client.get("/api/v1/tickers/FAKE_TICKER/history")
    assert res.status_code == 404

def test_news_endpoint():
    res = client.get("/api/v1/tickers/AAPL/news?limit=20")
    assert res.status_code == 200
    data = res.json()
    assert data["ticker"] == "AAPL"
    assert "TIMING-LIMITED" in data["timing_status"]

def test_metrics_endpoint():
    res = client.get("/api/v1/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "model_summaries" in data
    assert len(data["model_summaries"]) > 0

def test_predict_latest():
    res = client.post("/api/v1/predict", json={"ticker": "AAPL"})
    assert res.status_code == 200
    data = res.json()
    assert data["ticker"] == "AAPL"
    assert data["predicted_class"] in ["Up", "Down"]
    assert 0.0 <= data["probability"] <= 1.0
    assert "disclaimer" in data
    assert data["evaluation_scope"] == "Out-of-Sample Forward Test Horizon"

def test_predict_specific_date():
    res = client.post("/api/v1/predict", json={"ticker": "AAPL", "target_date": "2024-07-01"})
    assert res.status_code == 200
    data = res.json()
    assert data["target_date"] == "2024-07-01"
    assert data["ground_truth_class"] in ["Up", "Down"]

def test_predict_unknown_ticker():
    res = client.post("/api/v1/predict", json={"ticker": "INVALID_XYZ"})
    assert res.status_code == 400
    assert "Unsupported ticker" in res.json()["detail"]

def test_predict_out_of_range_date():
    res = client.post("/api/v1/predict", json={"ticker": "AAPL", "target_date": "1999-01-01"})
    assert res.status_code == 404
    assert "not found in held-out test evaluation window" in res.json()["detail"]
