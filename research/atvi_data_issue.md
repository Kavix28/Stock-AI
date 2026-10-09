# ATVI (Activision Blizzard) Market Data Source Blocker & Investigation Report

> **Document Purpose:** Investigate why YahooQuery cannot retrieve historical market data for `ATVI` (Activision Blizzard Inc.) and document the research baseline implications, root cause, empirical evidence, alternative data source evaluations, and architectural decisions.

---

## 1. Executive Summary

- **Paper Stock Universe:** The baseline research paper (_A deep fusion model for stock market prediction with news headlines and time series data_, Neural Computing and Applications, 2024) specifies 12 US stocks across 6 GICS sectors, including **Activision Blizzard (`ATVI`)** under Communication Services.
- **Current Pipeline Status:** YahooQuery successfully fetches 692 trading days (2020-01-01 to 2022-09-30) for 11 out of 12 paper stocks. However, YahooQuery queries for `ATVI` return an `Empty DataFrame`.
- **Overall Dataset Manifest Status:** `FAILED` (as required by data governance rules: `overall_status` cannot be `PASS` if any mandated paper baseline stock fails).
- **Core Findings:** `ATVI` was acquired by Microsoft Corporation (`MSFT`) for $95.00/share on **October 13, 2023**, and subsequently delisted from Nasdaq. Post-delisting, Yahoo Finance removed `ATVI` ticker history endpoints from public free APIs (`query1.finance.yahoo.com` and `query2.finance.yahoo.com`).

---

## 2. Empirical Evidence & Root Cause Analysis

### [FACT] 1: Ticker Delisting Status

- On **October 13, 2023**, Microsoft Corporation completed its acquisition of Activision Blizzard, Inc. (`ATVI`) in an all-cash transaction valued at $68.7 billion ($95.00 per share).
- Following the completion of the merger on October 13, 2023, trading of `ATVI` common stock was suspended on Nasdaq, and the symbol was officially delisted.

### [FACT] 2: YahooQuery & Yahoo Finance API Behavior

- Live execution of `yahooquery.Ticker('ATVI').history(start='2020-01-01', end='2022-09-30')` returns:
  ```
  Empty DataFrame
  Columns: [high, low, volume, open, close]
  Index: []
  ```
- Live execution of `yfinance.Ticker('ATVI').history(...)` returns:
  ```
  $ATVI: possibly delisted; no price data found
  Empty DataFrame
  ```
- Executing YahooQuery ticker search `yahooquery.search('Activision')` or `search('ATVI')` returns `quotes: []`.
- Direct HTTP GET requests to Yahoo Finance historical download endpoints (`https://query1.finance.yahoo.com/v7/finance/download/ATVI`) return HTTP 404 / 400 or empty data payloads.

### [FACT] 3: Paper Baseline Context

- The research paper baseline (_NeuComp 2024_) retrieved its historical market data via YahooQuery prior to late 2023 (during the model development period of 2021–2023), when `ATVI` was still an actively traded ticker on Nasdaq.
- The paper authors' official open-source GitHub repository (`rcorizzo/finbert-lstm`) provides news headline sentiment zip archives (`data/ATVI.zip`), but does **NOT** host raw market price time series CSV files (Open, High, Low, Adj Close, Volume).

### [FACT] 4: Commercial Data Vendor Availability

- Commercial paid market data providers (e.g., Polygon.io Paid Tier, Bloomberg Terminal, Refinitiv Eikon, CRSP, EODHD Paid Tier) maintain archived historical daily price series for delisted US equities including `ATVI`.
- Free tier endpoints across public APIs (Yahoo Finance, Alpha Vantage free tier, Nasdaq public quote API) have purged or restricted access to delisted symbols.
- Free web mirrors (e.g. `stooq.com`) maintain daily data for `atvi.us` but enforce Cloudflare Turnstile anti-bot protection that blocks automated programmatic HTTP requests.

---

## 3. Evaluated Solutions & Alternatives

| Option                                   | Method                                                                 |  Feasibility   | Academic & Architectural Evaluation                                                                                                |
| :--------------------------------------- | :--------------------------------------------------------------------- | :------------: | :--------------------------------------------------------------------------------------------------------------------------------- |
| **A. Silent Ticker Substitution**        | Replace `ATVI` with another stock (e.g., `EA` or `TTWO`)               |  ❌ REJECTED   | **Violation of Paper Baseline.** The paper explicitly selected ATVI. Substituting a ticker breaks paper reproducibility.           |
| **B. Data Fabrication / Synthetic Data** | Manufacture synthetic OHLCV prices for ATVI                            |  ❌ REJECTED   | **Violation of Research Integrity.** Synthetic price data corrupts market signals, indicator calculations, and evaluation metrics. |
| **C. Secondary Provider Integration**    | Extend `MarketDataProvider` to ingest genuine historical ATVI data     | ✅ RECOMMENDED | Maintains paper universe integrity via modular provider abstraction.                                                               |
| **D. Surface Blocker for Approval**      | Mark pipeline status as `ACTION REQUIRED` / `FAILED` and surface issue |  ✅ MANDATORY  | Explicitly surfaces the data-source change for academic review before proceeding to Phase 3.                                       |

---

## 4. Architectural Decision Log

### [OUR DECISION] 1: Maintain Canonical Paper Stock Universe

- We **MUST NOT** remove or replace `ATVI` in `configs/stocks.yaml` or the baseline model configuration. `ATVI` remains part of the 12-stock universe.

### [OUR DECISION] 2: Implement Secondary Provider Abstraction for Local / Alternative Data

- We maintain `YahooQueryMarketDataProvider` as the primary provider for all active stocks (`NFLX`, `SBUX`, `TSLA`, `NVDA`, `AAPL`, `AMT`, `PLD`, `NDAQ`, `SCHW`, `BIO`, `JNJ`).
- We implement `LocalCsvMarketDataProvider` and `FallbackMarketDataProvider` under the `MarketDataProvider` base class to ingest genuine historical market data for delisted assets like `ATVI` when local archived data or secondary sources are configured.

### [OUR DECISION] 3: Strict Schema & Quality Validation Enforcement

- Any historical market data ingested for `ATVI` (whether from an alternative vendor or local archive) **MUST** strictly pass through the canonical schema layer (`CanonicalMarketSchema`) and validation layer (`MarketDataValidator`), verifying:
  - Required fields: `symbol`, `date`, `open`, `high`, `low`, `adj_close`, `volume`.
  - Types: string dates (`YYYY-MM-DD`), numeric prices/volume.
  - Logical bounds: `high >= low`, `open` between high and low, non-negative volume.
  - Deduplication and missing value checks.
  - Zero artificial forward-filling of financial prices.

### [OUR DECISION] 4: Transparent Provenance & Manifest Governance

- The dataset manifest (`data/processed/market/manifest.json`) and data quality report (`reports/market_data_quality_report.md`) **MUST** explicitly state the provider used for each symbol (`YahooQuery` vs. alternative provider).
- If any required paper stock fails or requires a non-standard provider workaround that has not been approved, `overall_status` **MUST NOT** be set to `PASS`. It must accurately reflect `ACTION REQUIRED` or `FAILED`.

---

## 5. Corporate Action & Price Adjustment Analysis

### [FACT] Corporate Action History for ATVI

- **Stock Splits:** 2-for-1 split on July 10, 2008 (prior to acquisition range `2020-01-01`). No stock splits occurred during the acquisition window `2020-01-01` to `2022-09-30`.
- **Cash Buyout:** Acquired at $95.00 per share cash on October 13, 2023.
- **Adjusted Close Consistency:** Any historical price series for `ATVI` must preserve split- and dividend-adjusted closing prices (`adj_close`) to match the research paper methodology.
