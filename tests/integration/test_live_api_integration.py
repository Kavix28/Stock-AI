"""
End-to-End API Integration Tests for Stock-AI FastAPI Service.

Tests the actual running FastAPI app using TestClient, verifying response schema,
GMU gate weights, BiLSTM predictions, ATVI delisted handling, and error codes.
"""
import pytest
from fastapi.testclient import TestClient

import sys
import os
project_root = r"C:\PROJECTS\multimodal-stock-ai"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_api_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "healthy"]
    assert "service" in data
    assert "version" in data


def test_api_models_endpoint(client):
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    models = response.json()
    model_ids = [m["id"] for m in models]
    assert "gmu_fusion" in model_ids
    assert "market_bilstm" in model_ids


def test_api_multimodal_metrics_endpoint(client):
    response = client.get("/api/v1/metrics/multimodal")
    assert response.status_code == 200
    data = response.json()
    assert data["benchmark_type"] == "Multimodal Research Benchmark"
    assert len(data["models"]) >= 4


def test_api_predict_gmu_fusion_aapl(client):
    payload = {
        "ticker": "AAPL",
        "model_type": "gmu_fusion"
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    res = response.json()

    assert res["ticker"] == "AAPL"
    assert res["model_version"] == "GMU-Fusion-v1.0-PyTorch"
    assert res["predicted_class"] in ["Up", "Down"]
    assert 0.0 <= res["probability"] <= 1.0
    assert 0.0 <= res["market_gate_weight"] <= 1.0
    assert 0.0 <= res["news_gate_weight"] <= 1.0
    assert pytest.approx(res["market_gate_weight"] + res["news_gate_weight"], abs=1e-3) == 1.0


def test_api_predict_market_bilstm_nvda(client):
    payload = {
        "ticker": "NVDA",
        "model_type": "market_bilstm"
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    res = response.json()

    assert res["ticker"] == "NVDA"
    assert res["model_version"] == "Market-BiLSTM-v1.0-PyTorch"
    assert res["market_gate_weight"] == 1.0
    assert res["news_gate_weight"] == 0.0


def test_api_predict_delisted_atvi(client):
    payload = {
        "ticker": "ATVI",
        "model_type": "gmu_fusion"
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    res = response.json()

    assert res["ticker"] == "ATVI"
    assert "Delisted Oct 2023" in res["ticker_status"]


def test_api_predict_unsupported_ticker(client):
    payload = {
        "ticker": "INVALID_TICKER",
        "model_type": "gmu_fusion"
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 400
    assert "Unsupported ticker" in response.json()["detail"]


def test_api_predict_nonexistent_date(client):
    payload = {
        "ticker": "AAPL",
        "target_date": "1990-01-01",
        "model_type": "gmu_fusion"
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 404
