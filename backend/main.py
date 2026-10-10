"""
Stock-AI FastAPI Backend Service.

Exposes RESTful endpoints for ticker metadata, historical prices,
technical indicators, news headlines, benchmark metrics, and inference.
"""

import os
import sys
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from stock_ai.inference.engine import InferenceEngine

app = FastAPI(
    title="Stock-AI Research API",
    description="Multimodal AI for Stock Trend Prediction RESTful API",
    version="0.2.0"
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

inference_engine = InferenceEngine(
    data_dir="data/processed/market",
    results_dir="results/latest"
)

DATA_DIR = "data/processed/market"
NEWS_DIR = "data/processed/news"
RESULTS_DIR = "results/latest"

TICKER_INFO = {
    "AAPL": {"name": "Apple Inc.", "sector": "Information Technology", "delisted": False},
    "AMT": {"name": "American Tower Corporation", "sector": "Real Estate", "delisted": False},
    "ATVI": {"name": "Activision Blizzard, Inc.", "sector": "Communication Services", "delisted": True, "notes": "Delisted Oct 2023; Historical Fallback"},
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


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "Stock-AI Backend",
        "version": "0.2.0",
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
        "test_horizon": "2024-07-01 to 2026-10-09",
        "pooled_test_observations": 6292
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

    fpath = os.path.join(NEWS_DIR, f"{ticker}_news.csv")
    if not os.path.exists(fpath):
        return {"ticker": ticker, "count": 0, "news": [], "timing_status": "TIMING-LIMITED"}

    df = pd.read_csv(fpath)
    records = df.tail(limit).to_dict(orient="records")
    return {
        "ticker": ticker,
        "count": len(records),
        "timing_status": "TIMING-LIMITED (Daily aggregated without intraday timestamps)",
        "news": records
    }


@app.get("/api/v1/metrics")
def get_metrics() -> Dict[str, Any]:
    metrics_path = os.path.join(RESULTS_DIR, "metrics_by_ticker.csv")
    pooled_path = os.path.join(RESULTS_DIR, "pooled_metrics.csv")

    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Benchmark metrics not found.")

    m_df = pd.read_csv(metrics_path)
    p_df = pd.read_csv(pooled_path) if os.path.exists(pooled_path) else None

    # Compute overall averages per model
    model_averages = []
    for model_name, grp in m_df.groupby("model_type"):
        model_averages.append({
            "model_type": model_name,
            "mean_accuracy": float(grp["accuracy"].mean()),
            "mean_macro_f1": float(grp["macro_f1"].mean()),
            "mean_balanced_accuracy": float(grp["balanced_accuracy"].mean()),
            "tickers_count": int(len(grp))
        })

    return {
        "model_summaries": model_averages,
        "ticker_metrics": m_df.to_dict(orient="records"),
        "pooled_metrics": p_df.to_dict(orient="records") if p_df is not None else []
    }


@app.get("/api/v1/models")
def list_models() -> List[Dict[str, Any]]:
    return [
        {
            "id": "majority_baseline",
            "name": "Majority Baseline",
            "type": "heuristic",
            "description": "Predicts historical training-majority direction."
        },
        {
            "id": "direction_persistence",
            "name": "Direction Persistence (Lag-1)",
            "type": "heuristic",
            "description": "Predicts next-day direction matches previous trading day."
        },
        {
            "id": "lr_endpoint",
            "name": "Logistic Regression (t-1 Endpoint)",
            "type": "linear",
            "description": "Linear classifier fitted on lag-1 technical indicator vector."
        },
        {
            "id": "lr_flat",
            "name": "Logistic Regression (20-Day Flat)",
            "type": "linear",
            "description": "Linear classifier fitted on flattened 20-session market window."
        },
        {
            "id": "individual_bilstm",
            "name": "Individual Market BiLSTM",
            "type": "deep_learning",
            "description": "Two-layer Bidirectional LSTM trained per ticker."
        },
        {
            "id": "shared_bilstm",
            "name": "Shared Multi-Ticker BiLSTM (Pooled)",
            "type": "deep_learning",
            "description": "Pooled Bidirectional LSTM trained across 11 active equities."
        },
        {
            "id": "gmu_fusion",
            "name": "Gated Multimodal Unit (GMU)",
            "type": "multimodal_fusion",
            "description": "Learned convex gating between market sequence and FinBERT embeddings."
        }
    ]


@app.post("/api/v1/predict")
def predict_trend(req: PredictionRequest) -> Dict[str, Any]:
    ticker = req.ticker.upper()
    if ticker not in TICKER_INFO:
        raise HTTPException(status_code=400, detail=f"Unsupported ticker: '{ticker}'")

    result = inference_engine.get_prediction(ticker, req.target_date)
    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result.get("message"))
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
