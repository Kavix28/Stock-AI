# Stock-AI Project Cleanup & Modernization Report

**Project**: Stock-AI  
**Date**: 2026-10-10  
**Target Repository**: `https://github.com/Kavix28/Stock-AI.git`

---

## 1. Summary of Cleanup Operations

This document details the cleanup, consolidation, and refactoring executed to establish the canonical Stock-AI repository from the two existing candidate workspaces (`multimodal-stock-ai` and `finbert-lstm`).

### A. Files and Directories Removed or Excluded

1. **Temporary Investigation Files**:
   - `tmp_investigation/candidate_ATVI.csv`: Removed from repository working tree.
2. **Caches and Build Artifacts**:
   - `.pytest_cache/`, `__pycache__/`, `.mypy_cache/`, `*.pyc` purged and protected via updated `.gitignore`.
3. **Restricted Research Artifacts**:
   - Research paper PDF `s00521-024-10303-1.pdf` strictly protected via `.gitignore` to adhere to academic redistribution restrictions.
4. **Environment and Secrets**:
   - `.env`, `.env.local`, API keys, private credentials strictly ignored; `.env.example` provided.
5. **Heavy Video Outputs**:
   - `video/output/*.mp4` excluded from Git to prevent repository bloat while preserving reproduction scripts, narration markdown, storyboards, and subtitles.

---

## 2. Component Migration & Standardization

1. **Modular Source Tree (`src/stock_ai/`)**:
   - Integrated `src/stock_ai/data/window_builder.py` providing leakage-free chronological rolling windows (20-day lookback ending on $t-1$).
   - Integrated `src/stock_ai/features/indicators.py` calculating all 16 canonical technical indicators without future lookahead.
   - Integrated `src/stock_ai/models/baselines.py` implementing Majority, Direction Persistence, and Logistic Regression baselines.
   - Integrated `src/stock_ai/models/market/bilstm.py` and `src/stock_ai/models/fusion/gmu.py`.
   - Integrated `src/stock_ai/inference/engine.py` providing validated prediction serving without data fabrication.
2. **Centralized Configuration (`configs/`)**:
   - `configs/data.yaml`: 12-ticker metadata, chronological cutoff boundaries, core feature allowlist, forbidden columns.
   - `configs/model.yaml`: Architecture definitions, hyperparameters, and loss objectives.
   - `configs/experiment.yaml`: Random seeds (42, 123, 7), early stopping, and validation threshold calibration.
3. **Data & Manifest Preservation (`data/`)**:
   - `data/processed/market/`: 12 verified stock CSVs with complete 16-indicator features and resolved ATVI historical fallback.
   - `data/processed/news/`: Aggregated daily headlines for all 12 tickers.
   - `data/manifests/`: `market_data_manifest.csv`, `news_coverage_manifest.csv`, `train_validation_test_manifest.csv`.
4. **Reproducible Benchmarks (`results/`)**:
   - Preserved `results/latest/` with per-ticker metrics, pooled 6,292-observation metrics, and 12 individual prediction CSVs.
   - Preserved `results/reconciled_benchmark/` historical 180-day 2022 evaluation.

---

## 3. Test Suite Verification

After refactoring and cleanup, the entire test suite in `tests/` was executed:

- **34 tests passing** across unit, integration, and regression suites (`test_window_builder.py`, `test_indicators.py`, `test_baselines.py`, `test_inference.py`, `test_benchmark_manifests.py`, `test_market_pipeline.py`, `test_fallback_provider.py`, etc.).
- Zero test failures, zero syntax errors, and zero import regressions.
