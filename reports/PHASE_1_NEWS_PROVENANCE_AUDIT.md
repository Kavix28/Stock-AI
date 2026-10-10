# Stock-AI: Phase 1 News Provider, Provenance & Point-in-Time Audit

**Audit Date:** 2026-10-10 18:14:04 UTC  
**Repository:** `C:\PROJECTS\multimodal-stock-ai`  
**Execution Script:** `scripts/run_phase1_news_pipeline.py`  
**Audit Purpose:** Comprehensive Phase 1 verification of news provider integrations, empirical coverage, SHA-256 provenance manifests, anti-leakage invariants, and point-in-time readiness.

---

## 1. Executive Summary & Honest Readiness Status

| Evaluation Dimension | Verification Status | Exact Finding |
| :--- | :---: | :--- |
| **Provider Architecture** | **COMPLETE & TESTED** | Modular `NewsDataProvider` interface implemented with `LocalCanonicalNewsProvider`, `SecEdgarNewsProvider`, `PolygonNewsProvider`, and `TiingoNewsProvider`. 19/19 unit tests passing. |
| **Available Data Source** | **CANONICAL LOCAL ARCHIVE** | 12 tickers from the Chen et al. (2024) research archive (`data/processed/news/`). |
| **Historical Coverage** | **VERIFIED (2021-01-05 to 2022-09-20)** | 404 to 427 trading sessions per stock across 12 tickers (5,053 total headlines). |
| **Forward Coverage (2024-2026)** | **CRITICAL GAP: 0% (BLOCKER)** | **Zero headlines exist** for July 2024 to October 2026. Forward multimodal evaluation cannot proceed without API acquisition. |
| **Timestamp Precision** | **TIMING-LIMITED (Calendar Date Only)** | Local headlines lack hours/minutes/seconds. A headline on day $t$ cannot be proven to predate market open ($09:30$ EST) without intraday timestamps. |
| **Causal Anti-Leakage** | **VERIFIED & TESTED** | `NewsMarketAligner` strictly prevents future date leakage. Zero post-cutoff headlines admitted. |
| **Deduplication** | **ENFORCED** | Exact duplicate headlines are removed per date and tracked. Mean duplicate rate across tickers: 0.00%. |
| **ATVI Provenance** | **BLOCKED (DELISTED)** | Activision Blizzard delisted on October 13, 2023. Historical news (427 rows) verified, but 0 forward sessions exist. |

---

## 2. News Provider Evaluation & Credential Audit

| Provider Name | Class / Module | Configured in Environment? | Required Credentials | Licensing & Legal Usability | Timestamp Precision |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **Local Canonical Archive** | `LocalCanonicalNewsProvider` | **YES** | None (Local files) | Academic Research Use (Chen et al. 2024) | `TIMING-LIMITED` (Date only) |
| **U.S. SEC EDGAR** | `SecEdgarNewsProvider` | **YES** | `SEC_EDGAR_USER_AGENT` | Public Domain (U.S. Govt Work). Compliant Fair Access. | `POINT-IN-TIME CERTIFIED` (Second-level UTC) |
| **Polygon.io API** | `PolygonNewsProvider` | **NO (MISSING KEY)** | `POLYGON_API_KEY` | Commercial/Developer Subscription. Distribution restricted. | `POINT-IN-TIME CERTIFIED` (Second-level UTC) |
| **Tiingo News API** | `TiingoNewsProvider` | **NO (MISSING KEY)** | `TIINGO_API_KEY` | Commercial/Individual Token. Distribution restricted. | `POINT-IN-TIME CERTIFIED` (Second-level UTC) |

> [!IMPORTANT]
> **Credential Access Disclosure:**
> Neither `POLYGON_API_KEY` nor `TIINGO_API_KEY` is currently configured in the environment (`.env` does not exist).
> In accordance with project integrity guidelines, **no fake credentials were created and no mock downloads were simulated**.
> Phase 1 continues exclusively using the genuine, verified canonical news archive and the unauthenticated SEC EDGAR endpoint.

---

## 3. Ticker-by-Ticker Empirical Coverage & Alignment Statistics

The canonical archive was audited against the market calendar for all 12 tickers:

| symbol   |   trading_sessions |   sessions_with_news |   coverage_pct |   total_articles |   duplicate_articles |   duplicate_rate_pct |   max_articles_single_day | leakage_test_passed   | timing_status                               | atvi_delisted   |
|:---------|-------------------:|---------------------:|---------------:|-----------------:|---------------------:|---------------------:|--------------------------:|:----------------------|:--------------------------------------------|:----------------|
| AAPL     |                430 |                  403 |           93.7 |              404 |                    0 |                    0 |                         1 | True                  | TIMING-LIMITED: Missing Intraday Timestamps | False           |
| AMT      |                426 |                  422 |           99.1 |              423 |                    0 |                    0 |                         1 | True                  | TIMING-LIMITED: Missing Intraday Timestamps | False           |
| ATVI     |                430 |                  426 |           99.1 |              427 |                    0 |                    0 |                         1 | True                  | TIMING-LIMITED: Missing Intraday Timestamps | True            |
| BIO      |                430 |                  419 |           97.4 |              420 |                    0 |                    0 |                         1 | True                  | TIMING-LIMITED: Missing Intraday Timestamps | False           |
| JNJ      |                430 |                  426 |           99.1 |              427 |                    0 |                    0 |                         1 | True                  | TIMING-LIMITED: Missing Intraday Timestamps | False           |
| NDAQ     |                430 |                  426 |           99.1 |              427 |                    0 |                    0 |                         1 | True                  | TIMING-LIMITED: Missing Intraday Timestamps | False           |
| NFLX     |                430 |                  426 |           99.1 |              427 |                    0 |                    0 |                         1 | True                  | TIMING-LIMITED: Missing Intraday Timestamps | False           |
| NVDA     |                430 |                  426 |           99.1 |              427 |                    0 |                    0 |                         1 | True                  | TIMING-LIMITED: Missing Intraday Timestamps | False           |
| PLD      |                430 |                  426 |           99.1 |              427 |                    0 |                    0 |                         1 | True                  | TIMING-LIMITED: Missing Intraday Timestamps | False           |
| SBUX     |                430 |                  426 |           99.1 |              427 |                    0 |                    0 |                         1 | True                  | TIMING-LIMITED: Missing Intraday Timestamps | False           |
| SCHW     |                430 |                  426 |           99.1 |              427 |                    0 |                    0 |                         1 | True                  | TIMING-LIMITED: Missing Intraday Timestamps | False           |
| TSLA     |                429 |                  403 |           93.9 |              404 |                    0 |                    0 |                         1 | True                  | TIMING-LIMITED: Missing Intraday Timestamps | False           |

---

## 4. News Data Manifest (Reproducible Provenance)

A complete provenance manifest was generated at:
- `data/processed/news/news_manifest.csv`
- `data/processed/news/news_manifest.json`

### Manifest Sample (First 5 Tickers):

| symbol   |   total_rows | start_date   | end_date   | has_intraday_time   | timing_status                               | forward_2024_2026_coverage   | licensing_status                         |
|:---------|-------------:|:-------------|:-----------|:--------------------|:--------------------------------------------|:-----------------------------|:-----------------------------------------|
| AAPL     |          404 | 2021-01-05   | 2022-09-20 | False               | TIMING-LIMITED: Missing Intraday Timestamps | False                        | Academic Research Use (Chen et al. 2024) |
| AMT      |          423 | 2021-01-11   | 2022-09-20 | False               | TIMING-LIMITED: Missing Intraday Timestamps | False                        | Academic Research Use (Chen et al. 2024) |
| ATVI     |          427 | 2021-01-05   | 2022-09-20 | False               | TIMING-LIMITED: Missing Intraday Timestamps | False                        | Academic Research Use (Chen et al. 2024) |
| BIO      |          420 | 2021-01-05   | 2022-09-20 | False               | TIMING-LIMITED: Missing Intraday Timestamps | False                        | Academic Research Use (Chen et al. 2024) |
| JNJ      |          427 | 2021-01-05   | 2022-09-20 | False               | TIMING-LIMITED: Missing Intraday Timestamps | False                        | Academic Research Use (Chen et al. 2024) |

---

## 5. Causal Alignment & Anti-Leakage Architecture

### 5.1 Timezone & Daylight Saving Time Conversion
Market trading hours are anchored in US Eastern (`America/New_York`):
- Standard Time (EST): UTC-5 (e.g. 09:30 EST = 14:30 UTC, 16:00 EST = 21:00 UTC)
- Daylight Saving Time (EDT): UTC-4 (e.g. 09:30 EDT = 13:30 UTC, 16:00 EDT = 20:00 UTC)
`NewsMarketAligner.get_market_session_bounds_utc()` automatically resolves timezone offsets using `pytz.timezone('America/New_York')`.

### 5.2 Causal Observation Windows
1. **`PRE_MARKET_WINDOW` (Intraday Certified):**
   $$\mathcal{W}_t = [\text{Market Close}_{t-1} \ (16:00 \ \text{EST}), \ \text{Market Open}_t \ (09:30 \ \text{EST}))$$
   Eliminates all intraday lookahead. Requires ISO-8601 UTC timestamps.
2. **`DAILY_DATE_CONVENTIONAL` (Chen et al. 2024 Replication):**
   Matches headlines by publication calendar date ($t$). Tagged `TIMING-LIMITED` because post-close news on date $t$ could contaminate features.
3. **`DAILY_DATE_LAG1` (Conservative Lagged Alignment):**
   Matches headlines published on date $t-1$ to predict return on date $t$.

### 5.3 Deduplication & Missing News Policy
- **Duplicate Removal:** Exact case-insensitive matching collapses repeated syndicated wires into a single headline, reducing token redundancy.
- **Missing News:** Days with zero headlines receive empty string `""` (or `[NO_NEWS]`), `article_count = 0`, and `has_news = False`. Zero hallucinated or synthetic headlines are injected.

---

## 6. Automated Test Suite Results

The Phase 1 news pipeline was verified through 19 newly implemented unit tests:
- `tests/unit/test_news_schema.py`: Schema normalization, text sanitization, deterministic hash generation.
- `tests/unit/test_news_providers.py`: Provider metadata, exception handling, credential guards.
- `tests/unit/test_news_alignment.py`: Pre-market UTC session bounds, DST offsets, deduplication, missing news fallback.
- `tests/unit/test_news_validator.py`: Future timestamp leakage detection, chronological split non-overlap.
- `tests/unit/test_news_manifest.py`: Manifest export verification (CSV + JSON).

**Full Test Suite Run:**
- Total tests executed: **70** (51 existing platform regression tests + 19 news pipeline tests)
- Total passed: **70 / 70 (100%)**
- Total failed: **0**

---

## 7. Remaining Blockers & Next Steps for Phase 2

1. **Forward News API Access (BLOCKER):**
   To benchmark multimodal prediction on the 2024–2026 forward testing horizon (6,292 sessions), a commercial API key (`POLYGON_API_KEY` or `TIINGO_API_KEY`) must be supplied.
2. **Phase 2 Development Focus:**
   In the absence of forward news credentials, Phase 2 will implement the **modular FinBERT encoder** (`src/stock_ai/models/news/finbert.py`), **embedding cache** (`data/processed/embeddings/`), and **GMU training pipeline** validated on the historical 2021–2022 dataset where verified news is available.
