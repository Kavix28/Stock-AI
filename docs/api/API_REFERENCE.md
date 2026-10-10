# Stock-AI RESTful API Reference

The Stock-AI backend is implemented with FastAPI and provides high-performance, asynchronous endpoints for data exploration, model evaluation, and inference.

---

## Base URL & Interactive Docs

- **Base URL:** `http://localhost:8000`
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## Endpoints

### 1. Health Check

`GET /health`

- **Description:** Returns service operational status.
- **Response:**
  ```json
  {
    "status": "healthy",
    "service": "Stock-AI Backend",
    "version": "0.2.0",
    "environment": "development"
  }
  ```

---

### 2. Platform Metadata

`GET /api/v1/metadata`

- **Description:** Returns dataset scope, lookback configuration, and test horizon.
- **Response:**
  ```json
  {
    "project_name": "Stock-AI",
    "research_title": "Multimodal AI for Stock Trend Prediction",
    "paper_doi": "10.1007/s00521-024-10303-1",
    "supported_tickers_count": 12,
    "features_count": 16,
    "lookback_window": 20,
    "test_horizon": "2024-07-01 to 2026-10-09",
    "pooled_test_observations": 6292
  }
  ```

---

### 3. Universe Ticker List

`GET /api/v1/tickers`

- **Description:** Returns the 12 equities with company name, sector, and delisting status.

---

### 4. Historical Ticker Series

`GET /api/v1/tickers/{ticker}/history?limit=100`

- **Parameters:**
  - `ticker` (string, path): Stock symbol (e.g., `AAPL`, `NVDA`).
  - `limit` (integer, query, default: 100): Number of latest trading sessions to retrieve.
- **Response:** Array of records with date, open, high, low, close, volume, and 16 technical indicators.

---

### 5. News Headlines Archive

`GET /api/v1/tickers/{ticker}/news?limit=50`

- **Parameters:**
  - `ticker` (string, path): Stock symbol.
  - `limit` (integer, query, default: 50): Max headlines to retrieve.
- **Notes:** Includes `TIMING-LIMITED` status flag to reflect daily aggregation.

---

### 6. Benchmark Metrics Summary

`GET /api/v1/metrics`

- **Description:** Returns mean accuracy, Macro F1, and balanced accuracy across all evaluated models, along with per-ticker breakdowns.

---

### 7. Point-in-Time Prediction

`POST /api/v1/predict`

- **Request Body:**
  ```json
  {
    "ticker": "AAPL",
    "target_date": "2026-10-09"
  }
  ```
- **Response:**
  ```json
  {
    "ticker": "AAPL",
    "target_date": "2026-10-09",
    "predicted_class": "Up",
    "predicted_label": 1,
    "ground_truth_class": "Up",
    "ground_truth_label": 1,
    "probability": 0.534,
    "model_version": "Shared-BiLSTM-v2.1",
    "lookback_window": 20,
    "evaluation_scope": "Out-of-Sample Forward Test Horizon",
    "disclaimer": "This is an academic research prediction. Not financial advice."
  }
  ```
