"""
Stock-AI FastAPI Backend Service.

Exposes RESTful endpoints for ticker metadata, historical prices,
technical indicators, news headlines, multimodal benchmark metrics, and inference.
"""

import os
import sys
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

# Add src and project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src.stock_ai.inference.engine import InferenceEngine

app = FastAPI(
    title="Stock-AI Research API",
    description="Multimodal AI for Stock Trend Prediction RESTful API",
    version="0.3.0"
)

# CORS Middleware
origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(project_root, "data", "processed", "market")
NEWS_DIR = os.path.join(project_root, "data", "processed", "news")
RESULTS_DIR = os.path.join(project_root, "results")
MODELS_DIR = os.path.join(project_root, "models")

inference_engine = InferenceEngine(
    data_dir=DATA_DIR,
    models_dir=MODELS_DIR,
    results_dir=RESULTS_DIR
)

TICKER_INFO = {
    "AAPL": {"name": "Apple Inc.", "sector": "Information Technology", "delisted": False},
    "AMT": {"name": "American Tower Corporation", "sector": "Real Estate", "delisted": False},
    "ATVI": {"name": "Activision Blizzard, Inc.", "sector": "Communication Services", "delisted": True, "notes": "Delisted Oct 2023; Historical Research Archive Only"},
    "BIO": {"name": "Bio-Rad Laboratories, Inc.", "sector": "Health Care", "delisted": False},
    "JNJ": {"name": "Johnson & Johnson", "sector": "Health Care", "delisted": False},
    "NDAQ": {"name": "Nasdaq, Inc.", "sector": "Financials", "delisted": False},
    "NFLX": {"name": "Netflix, Inc.", "sector": "Communication Services", "delisted": False},
    "NVDA": {"name": "NVIDIA Corporation", "sector": "Information Technology", "delisted": False},
    "PLD": {"name": "Prologis, Inc.", "sector": "Real Estate", "delisted": False},
    "SBUX": {"name": "Starbucks Corporation", "sector": "Consumer Discretionary", "delisted": False},
    "SCHW": {"name": "The Charles Schwab Corporation", "sector": "Financials", "delisted": False},
    "TSLA": {"name": "Tesla, Inc.", "sector": "Consumer Discretionary", "delisted": False}
}


class PredictionRequest(BaseModel):
    ticker: str
    target_date: Optional[str] = None
    model_type: Optional[str] = "gmu_fusion"


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "Stock-AI Backend",
        "version": "0.3.0",
        "environment": os.getenv("ENVIRONMENT", "development")
    }


@app.get("/api/v1/metadata")
def get_metadata() -> Dict[str, Any]:
    return {
        "project_name": "Stock-AI",
        "research_title": "Multimodal AI for Stock Trend Prediction",
        "paper_doi": "10.1007/s00521-024-10303-1",
        "supported_tickers_count": 12,
        "features_count": 16,
        "lookback_window": 20,
        "pooled_test_observations": 6292,
        "multimodal_architecture": "Gated Multimodal Unit (GMU)",
        "news_encoder": "ProsusAI/finbert (768-dim CLS)",
        "market_encoder": "Two-Layer Bidirectional LSTM (64-dim)",
        "evaluation_period": "2021-01-05 to 2022-09-20 (Verified historical news archive)",
        "atvi_status": "ISOLATED: Delisted Oct 2023; Historical 2021-2022 Archive Only"
    }


@app.get("/api/v1/tickers")
def list_tickers() -> List[Dict[str, Any]]:
    tickers = []
    for symbol, info in TICKER_INFO.items():
        tickers.append({
            "symbol": symbol,
            "name": info["name"],
            "sector": info["sector"],
            "delisted": info["delisted"],
            "notes": info.get("notes", "")
        })
    return tickers


@app.get("/api/v1/tickers/{ticker}/history")
def get_ticker_history(ticker: str, limit: int = Query(100, ge=10, le=1000)) -> Dict[str, Any]:
    ticker = ticker.upper()
    if ticker not in TICKER_INFO:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found.")

    fpath = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail=f"Historical dataset for '{ticker}' missing.")

    df = pd.read_csv(fpath)
    records = df.tail(limit).to_dict(orient="records")
    return {
        "ticker": ticker,
        "name": TICKER_INFO[ticker]["name"],
        "count": len(records),
        "data": records
    }


@app.get("/api/v1/tickers/{ticker}/news")
def get_ticker_news(ticker: str, limit: int = Query(50, ge=5, le=200)) -> Dict[str, Any]:
    ticker = ticker.upper()
    if ticker not in TICKER_INFO:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found.")

    fpath = os.path.join(NEWS_DIR, f"{ticker}.csv")
    if not os.path.exists(fpath):
        fpath = os.path.join(NEWS_DIR, f"{ticker}_news.csv")

    if not os.path.exists(fpath):
        return {"ticker": ticker, "count": 0, "news": [], "timing_status": "NO-DATA"}

    df = pd.read_csv(fpath)
    records = df.tail(limit).to_dict(orient="records")
    return {
        "ticker": ticker,
        "count": len(records),
        "timing_status": "TIMING-LIMITED (Calendar Date YYYY-MM-DD Alignment)",
        "news": records
    }


@app.get("/api/v1/metrics")
def get_metrics() -> Dict[str, Any]:
    # Multimodal metrics path
    multi_path = os.path.join(RESULTS_DIR, "multimodal", "multimodal_metrics_summary.csv")
    legacy_path = os.path.join(RESULTS_DIR, "latest", "metrics_by_ticker.csv")

    if os.path.exists(multi_path):
        m_df = pd.read_csv(multi_path)
        records = m_df.to_dict(orient="records")
        return {
            "benchmark_type": "Multimodal Research Benchmark (PyTorch GMU & FinBERT)",
            "evaluation_period": "2021-01-05 to 2022-09-20",
            "models": records,
            "model_summaries": records,
            "ticker_metrics": records
        }

    if os.path.exists(legacy_path):
        l_df = pd.read_csv(legacy_path)
        records = l_df.to_dict(orient="records")
        return {
            "benchmark_type": "Legacy Market-Only Benchmark",
            "models": records,
            "model_summaries": records,
            "ticker_metrics": records
        }

    raise HTTPException(status_code=404, detail="Benchmark metrics not found.")


@app.get("/api/v1/metrics/multimodal")
def get_multimodal_metrics() -> Dict[str, Any]:
    multi_path = os.path.join(RESULTS_DIR, "multimodal", "multimodal_metrics_summary.csv")
    if not os.path.exists(multi_path):
        raise HTTPException(status_code=404, detail="Multimodal metrics summary not found.")

    m_df = pd.read_csv(multi_path)
    return {
        "benchmark_type": "Multimodal Research Benchmark",
        "models": m_df.to_dict(orient="records")
    }


@app.get("/api/v1/models")
def list_models() -> List[Dict[str, Any]]:
    return [
        {
            "id": "gmu_fusion",
            "name": "Gated Multimodal Unit (GMU Fusion)",
            "type": "multimodal_fusion",
            "description": "Learned convex gating combining Market BiLSTM (64-d) with FinBERT embeddings (768-d).",
            "timing_status": "TIMING-LIMITED (Daily Headline Alignment)"
        },
        {
            "id": "market_bilstm",
            "name": "Market-Only BiLSTM (PyTorch)",
            "type": "deep_learning",
            "description": "Two-layer Bidirectional LSTM on 20-session sequential market technical indicators.",
            "timing_status": "POINT-IN-TIME CERTIFIED"
        },
        {
            "id": "lr_endpoint",
            "name": "Logistic Regression (t-1 Endpoint)",
            "type": "linear",
            "description": "Linear classifier fitted on lag-1 technical indicator vector.",
            "timing_status": "POINT-IN-TIME CERTIFIED"
        },
        {
            "id": "direction_persistence",
            "name": "Direction Persistence (Lag-1)",
            "type": "heuristic",
            "description": "Predicts next-day direction matches previous trading day.",
            "timing_status": "POINT-IN-TIME CERTIFIED"
        },
        {
            "id": "majority_baseline",
            "name": "Majority Baseline",
            "type": "heuristic",
            "description": "Predicts historical training-majority direction.",
            "timing_status": "POINT-IN-TIME CERTIFIED"
        }
    ]


@app.post("/api/v1/predict")
def predict_trend(req: PredictionRequest) -> Dict[str, Any]:
    ticker = req.ticker.upper()
    if ticker not in TICKER_INFO:
        raise HTTPException(status_code=400, detail=f"Unsupported ticker: '{ticker}'")

    result = inference_engine.get_prediction(ticker, req.target_date, model_type=req.model_type or "gmu_fusion")
    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result.get("message"))
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
