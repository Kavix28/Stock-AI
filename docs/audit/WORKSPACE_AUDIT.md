# Comprehensive Workspace Audit: Primary Candidate vs. Secondary Research Workspace

**Project**: Stock-AI  
**Research Title**: Multimodal AI for Stock Trend Prediction  
**Audit Timestamp**: 2026-10-10 19:35 UTC  
**Target Repository**: `https://github.com/Kavix28/Stock-AI.git` (`main` branch)

---

## 1. Executive Summary

This audit compares the two existing candidate workspaces to establish an authoritative component map for consolidating Stock-AI into a production-grade research platform:

1. **Primary Candidate Workspace**: `C:\PROJECTS\multimodal-stock-ai`
   - **Git Status**: Initialized Git repository on branch `main`, tracking `https://github.com/Kavix28/stock_ai.git` (commit `29f06fe`).
   - **Architecture**: Professional modular Python package layout (`src/stock_ai/`, `configs/`, `tests/unit/`, `pyproject.toml`).
   - **Capabilities**: Provider abstraction for market data downloading via YahooQuery; schema normalization; 23 passing tests.
   - **Limitations**: Stalled at Phase 2 due to delisting of ticker `ATVI`. Contains **no** machine learning models, **no** neural networks, **no** news processing, **no** benchmark predictions, **no** frontend, and **no** backend service.

2. **Secondary Research Workspace**: `C:\PROJECTS\Research paer\finbert-lstm`
   - **Git Status**: Fork of original research paper repository (`rcorizzo/finbert-lstm.git`).
   - **Architecture**: Flat research script layout evolved with dedicated modules (`src/data/window_builder.py`, `scripts/`).
   - **Capabilities**: Complete, verified quantitative machine learning and deep learning engine. Contains all 12 stock datasets with full 16 technical indicators, resolved ATVI fallback, news headline archives, leakage-free chronological windowing, 4 baselines, individual BiLSTM, shared multi-ticker BiLSTM, FinBERT-LSTM, Bottleneck fusion, GMU gated fusion, authoritative historical 180-day and modern 572-day benchmark outputs, and 32 passing tests.

**Consolidation Decision**: `C:\PROJECTS\multimodal-stock-ai` is the primary workspace. We migrate the verified research components from `finbert-lstm` into `multimodal-stock-ai`'s clean package hierarchy, preserving the secondary research workspace completely intact without deletion or modification.

---

## 2. Detailed Workspace Comparison Matrix

| Component                | Primary Workspace (`multimodal-stock-ai`)      | Secondary Workspace (`finbert-lstm`)                         | Authoritative Source & Consolidation Decision                                                                     |
| :----------------------- | :--------------------------------------------- | :----------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------- |
| **Git & Remote**         | Branch `main`, tracking `Kavix28/stock_ai.git` | Branch `main`, tracking `rcorizzo/finbert-lstm.git`          | **Primary**: Retain `multimodal-stock-ai` repository, update remote to `https://github.com/Kavix28/Stock-AI.git`. |
| **Package Layout**       | Modular `src/stock_ai` with `pyproject.toml`   | Scripts and `src/data/`                                      | **Primary**: Adopt `src/stock_ai` modular package structure.                                                      |
| **Configuration**        | `configs/base.yaml`, `configs/stocks.yaml`     | Hardcoded / argparse CLI flags                               | **Primary**: Standardize on YAML configuration files in `configs/`.                                               |
| **Stock Universe**       | 11 stocks downloaded; `ATVI` missing           | All 12 stocks present with full history                      | **Secondary**: Migrate 12-ticker data with resolved `ATVI` historical fallback.                                   |
| **Technical Indicators** | None                                           | 16 core indicators (EMA 12/26, MACD, SAR, BB, Stoch, RSI)    | **Secondary**: Integrate 16-indicator computation into `src/stock_ai/features/`.                                  |
| **News Processing**      | None                                           | Aggregated headlines for all 12 stocks; FinBERT embeddings   | **Secondary**: Migrate news loader and sentiment modules into `src/stock_ai/news/`.                               |
| **Temporal Windowing**   | Basic date check                               | `window_builder.py` with boundary-crossing lookbacks         | **Secondary**: Migrate leakage-free window builder into `src/stock_ai/data/`.                                     |
| **Model Baselines**      | None                                           | Majority, Persistence (Lag-1), Next-Day LR (Endpoint & Flat) | **Secondary**: Migrate into `src/stock_ai/models/baselines/`.                                                     |
| **Neural Architectures** | None                                           | Market-Only BiLSTM, Shared BiLSTM, FinBERT-LSTM, GMU         | **Secondary**: Migrate into `src/stock_ai/models/`.                                                               |
| **Benchmark Artifacts**  | None                                           | Reconciled 180-day 2022 & Modern 572-day 2024–2026 results   | **Secondary**: Retain all verified outputs in `results/`.                                                         |
| **Frontend UI**          | None                                           | None                                                         | **New**: Build modern dark AI research dashboard in `frontend/`.                                                  |
| **Backend API**          | None                                           | None                                                         | **New**: Build FastAPI service in `backend/`.                                                                     |
| **Explanatory Video**    | None                                           | None                                                         | **New**: Produce rendered MP4 video in `video/output/`.                                                           |
| **Automated Tests**      | 23 unit/integration tests                      | 32 unit/regression tests                                     | **Consolidated**: Combine both test suites for complete coverage.                                                 |

---

## 3. Data Provenance & ATVI Delisting Resolution

- **The Problem**: Activision Blizzard (`ATVI`) was delisted from Nasdaq in October 2023 following its acquisition by Microsoft. As a result, modern Yahoo Finance and YahooQuery API calls return 0 rows for live market downloads, causing acquisition pipelines to halt.
- **The Resolution**: In `finbert-lstm`, a provider fallback mechanism was established that loads the verified historical archive for `ATVI` (`3,079` trading sessions ending September 20, 2022), validates all 16 technical indicators, and records `Delisted Oct 2023; Historical Archive Fallback` in the dataset manifest.
- **Academic Distinction**: All experiments involving `ATVI` are explicitly labeled as using the historical evaluation horizon (180 targets in 2022), while active tickers share the modern out-of-sample forward test horizon (572 targets in 2024–2026).

---

## 4. Migration & Consolidation Plan

1. **Package Refactoring**:
   - Move `window_builder.py` and data loaders into `src/stock_ai/data/` and `src/stock_ai/features/`.
   - Implement model definitions in `src/stock_ai/models/` (`baselines.py`, `market_bilstm.py`, `fusion.py`, `gmu.py`).
   - Create unified training and evaluation runners in `src/stock_ai/training/` and `src/stock_ai/evaluation/`.
2. **Data & Results Preservation**:
   - Copy verified datasets (`data/stock_data_latest/`, `data/news_latest/`) to `data/`.
   - Copy historical and modern benchmark results (`results/reconciled_benchmark/`, `results/latest/`) to `results/`.
3. **Backend Service (`backend/`)**:
   - Implement FastAPI application exposing endpoints for metadata, tickers, price history, indicators, news, benchmark metrics, and model inference.
4. **Frontend Dashboard (`frontend/`)**:
   - Implement dark AI research application with Landing Page, Dashboard, Stock Explorer, Model Laboratory, Prediction Demo, and Documentation.
5. **Video Production (`video/`)**:
   - Render `Stock_AI_Project_Explained.mp4` with narration script, storyboard, and captions.
6. **Git Synchronization**:
   - Point remote `origin` to `https://github.com/Kavix28/Stock-AI.git`, verify clean working tree, and push to branch `main`.
