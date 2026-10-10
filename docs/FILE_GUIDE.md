# Stock-AI Complete File Guide

This document provides a comprehensive, file-by-file inventory of every maintained first-party file in the Stock-AI repository.

---

## 1. Root Files

- **`README.md`**: Master repository documentation containing 28 structured sections covering research context, architecture, mathematical formulation, benchmark tables, setup instructions, and disclaimers.
- **`LICENSE`**: Open-source MIT License granting permissive usage and modification rights.
- **`.gitignore`**: Git exclusion rules protecting virtual environments (`.venv`), node modules (`node_modules`), Python caches, research paper PDFs (`*.pdf`), heavy video binaries (`*.mp4`), and environment secrets (`.env`).
- **`.env.example`**: Template environment variable file defining default host, port, directory paths, and CORS settings.
- **`pyproject.toml`**: Standard Python packaging specification defining package metadata, build system, dependencies, and pytest configuration.
- **`requirements.txt`**: Pinned pip requirements for the Python environment.

---

## 2. Configuration Directory (`configs/`)

- **`configs/base.yaml`**: Base platform settings including project metadata, logging levels, and environment presets.
- **`configs/stocks.yaml`**: Universe specification listing the 12 equities, sectors, and historical availability.
- **`configs/data.yaml`**: Centralized data configuration defining chronological cutoffs (train < 2023, val < 2024-07, test >= 2024-07), directory paths, 16 core feature allowlist, and forbidden target columns.
- **`configs/model.yaml`**: Hyperparameter specifications for baseline heuristics, Linear Models, Market BiLSTM, Shared Multi-Ticker BiLSTM, and Gated Multimodal Unit (GMU) fusion architectures.
- **`configs/experiment.yaml`**: Reproducibility parameters including random seed list (`[42, 123, 7]`), early stopping patience, validation threshold search grids, and reporting outputs.

---

## 3. Data Directory (`data/`)

- **`data/README.md`**: Overview of dataset structure, schema specifications, and historical provenance.
- **`data/raw/market/`**: Immutable raw market CSV downloads directly from YahooQuery/YFinance providers.
- **`data/processed/market/`**: Canonical processed market datasets for all 12 equities (`AAPL.csv` through `TSLA.csv`), containing 4,185 daily trading sessions per active stock with all 16 technical indicators and zero NaNs.
- **`data/processed/news/`**: Aggregated daily financial headlines for all 12 equities.
- **`data/manifests/market_data_manifest.csv`**: Comprehensive audit manifest tracking date ranges, observation counts, null counts, and delisting status.
- **`data/manifests/news_coverage_manifest.csv`**: News coverage inventory tracking headline counts and daily date coverage.
- **`data/manifests/train_validation_test_manifest.csv`**: Exact observation counts across train, validation, and test partitions for each stock.

---

## 4. Source Code (`src/stock_ai/`)

- **`src/stock_ai/__init__.py`**: Top-level package exports (`get_logger`, `build_chronological_windows`, `compute_indicators`, `InferenceEngine`).
- **`src/stock_ai/logger.py`**: Centralized structured logging utility.
- **`src/stock_ai/config/settings.py`**: Pydantic-based configuration loaders parsing YAML definitions into typed objects.
- **`src/stock_ai/data/market/base.py`**: Abstract base class defining provider interface (`MarketDataProvider`).
- **`src/stock_ai/data/market/download.py`**: CLI downloader orchestrating batch equity acquisition.
- **`src/stock_ai/data/market/fallback_provider.py`**: Provider fallback mechanism for delisted tickers (e.g., `ATVI`).
- **`src/stock_ai/data/market/schema.py`**: Pydantic schema validation for OHLCV tabular rows.
- **`src/stock_ai/data/market/validator.py`**: Strict financial validation enforcing `high >= low`, `high >= open`, `high >= close`, and date monotonicity.
- **`src/stock_ai/data/market/yahooquery_provider.py`**: Concrete YahooQuery acquisition provider with exponential backoff retries.
- **`src/stock_ai/data/window_builder.py`**: Core leakage-free chronological rolling window constructor enforcing 20-day lookback ending strictly on session $t-1$ and fitting MinMaxScaler solely on training sets.
- **`src/stock_ai/features/indicators.py`**: Leakage-free technical indicator calculator producing all 16 indicators (EMA 12/26, MACD, RSI, Bollinger Bands, Parabolic SAR, Stochastic %K/%D) and dropping the first 33 warmup rows.
- **`src/stock_ai/models/baselines.py`**: Heuristic and linear baselines (`MajorityBaseline`, `DirectionPersistenceBaseline`, `LogisticRegressionBaseline`).
- **`src/stock_ai/models/market/bilstm.py`**: Two-layer Bidirectional LSTM architecture with dropout regularization for sequential price processing.
- **`src/stock_ai/models/fusion/gmu.py`**: Gated Multimodal Unit (GMU) architecture computing learned convex combinations between sequential price representations and FinBERT embeddings.
- **`src/stock_ai/evaluation/metrics.py`**: Metrics computation (`accuracy`, `macro_f1`, `balanced_accuracy`, confusion matrix) and validation threshold calibration (`calibrate_threshold_on_val`).
- **`src/stock_ai/inference/engine.py`**: Point-in-time inference service serving predictions from trained models or validated benchmark series.

---

## 5. Execution Scripts (`scripts/`)

- **`scripts/refresh_data.py`**: End-to-end market data refresh script downloading active equities from YFinance and applying historical archive fallback for delisted ATVI.
- **`scripts/validate_data.py`**: Validation runner verifying schema compliance, boundary checks, and zero NaNs across all 12 equity datasets.
- **`scripts/evaluate.py`**: Master benchmark runner evaluating Baselines, Individual BiLSTM, and Shared Multi-Ticker BiLSTM across all tickers.

---

## 6. Test Suite (`tests/`)

- **`tests/conftest.py`**: Pytest fixtures for test datasets and mock providers.
- **`tests/unit/test_validation.py`**: Unit tests for schema validation and boundary checks.
- **`tests/unit/test_temporal.py`**: Unit tests for chronological sequencing.
- **`tests/unit/test_provider.py`**: Unit tests for YahooQuery provider.
- **`tests/unit/test_fallback_provider.py`**: Unit tests for fallback provider handling delisted tickers.
- **`tests/unit/test_atvi_validation.py`**: Validation tests for historical ATVI dataset.
- **`tests/unit/test_normalization.py`**: Tests for schema column standardization.
- **`tests/unit/test_cli.py`**: Tests for CLI flags and arguments.
- **`tests/unit/test_window_builder.py`**: Unit tests for 20-day rolling lookback ending strictly on $t-1$ and scaler isolation.
- **`tests/unit/test_indicators.py`**: Unit tests for mathematical correctness of all 16 technical indicators.
- **`tests/unit/test_baselines.py`**: Unit tests for Majority, Persistence, and Logistic Regression baselines.
- **`tests/unit/test_inference.py`**: Unit tests for inference engine and prediction formats.
- **`tests/unit/test_backend_api.py`**: Integration tests for FastAPI REST endpoints (`/health`, `/api/v1/tickers`, `/api/v1/predict`, `/api/v1/metrics`).
- **`tests/integration/test_market_pipeline.py`**: End-to-end integration test of acquisition and processing pipeline.
- **`tests/regression/test_benchmark_manifests.py`**: Regression assertions verifying benchmark metrics files and prediction artifacts.

---

## 7. Backend Application (`backend/`)

- **`backend/main.py`**: Production-ready FastAPI application exposing endpoints for ticker metadata, historical OHLCV series, technical indicators, news headlines, benchmark metrics, and inference.

---

## 8. Frontend Application (`frontend/`)

- **`frontend/package.json`**: Node dependencies including React 18, Vite, TypeScript, Lucide React, and Recharts.
- **`frontend/vite.config.ts`**: Vite configuration with proxy rules routing API calls to `localhost:8000`.
- **`frontend/tsconfig.json`**: TypeScript compiler options.
- **`frontend/index.html`**: Application entry HTML with Google Fonts (`Inter` and `JetBrains Mono`).
- **`frontend/src/main.tsx`**: React DOM root mounter.
- **`frontend/src/App.tsx`**: Main application container managing multi-tab navigation and layout.
- **`frontend/src/index.css`**: Professional dark glassmorphic design system tokens and responsive styles.
- **`frontend/src/api.ts`**: Typed REST client connecting to backend endpoints with client-side fallback resilience.
- **`frontend/src/components/Navbar.tsx`**: Header navigation bar with project logo, navigation tabs, and DOI reference badge.
- **`frontend/src/views/LandingView.tsx`**: Research landing view with problem statement, multimodal overview, and universe chips.
- **`frontend/src/views/DashboardView.tsx`**: Research dashboard displaying real dataset coverage, schema validation status, and multi-model benchmark metrics.
- **`frontend/src/views/ExplorerView.tsx`**: Interactive equity and technical indicator chart explorer with Bollinger Bands and EMAs.
- **`frontend/src/views/ModelLabView.tsx`**: Model laboratory comparing empirical performance, Macro F1 vs. Accuracy tradeoffs, and ablation metrics.
- **`frontend/src/views/PredictionDemoView.tsx`**: Model prediction demonstration with point-in-time ground truth verification and risk disclaimers.
- **`frontend/src/views/DocsView.tsx`**: Platform documentation and mathematical formulations view.

---

## 9. Benchmark Results (`results/`)

- **`results/latest/metrics_by_ticker.csv`**: Comprehensive out-of-sample metrics for every model across each of the 12 tickers.
- **`results/latest/pooled_metrics.csv`**: Consolidated metrics across 6,292 test observations for the Shared Multi-Ticker BiLSTM model.
- **`results/latest/predictions/*.csv`**: Exact target-by-target out-of-sample predictions and probabilities for each of the 12 tickers.
- **`results/reconciled_benchmark/`**: Historical 180-day 2022 evaluation benchmark artifacts.

---

## 10. Technical Video Assets (`video/`)

- **`video/output/Stock_AI_Project_Explained.mp4`**: Rendered 1920x1080 Full HD 9m 30s technical explanatory video covering all 7 chapters.
- **`video/script/NARRATION_SCRIPT.md`**: Complete verbatim voiceover narration script with exact timings and cues.
- **`video/storyboard/STORYBOARD.md`**: Visual storyboard detailing scene compositions, diagrams, and transitions.
- **`video/captions/Stock_AI_Project_Explained.srt`**: Synchronized SubRip (.srt) subtitle track.
- **`video/render_video.py`**: Automated video rendering script generating high-resolution slide frames and executing FFmpeg compilation.
- **`video/assets/`**: High-resolution generated slide frames (`slide_01.png` through `slide_07.png`).
