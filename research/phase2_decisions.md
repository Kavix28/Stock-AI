# Phase 2 — Architecture & Implementation Decisions

This document tracks all decisions made during Phase 2 (Market Data Acquisition & Data Pipeline), distinguishing strictly between research paper specifications (`[FROM PAPER]`) and engineering design decisions (`[OUR DECISION]`).

---

## 1. Data Provider Selection

- **Decision:** Use `yahooquery` as the primary market data provider for reproduction.
- **Classification:** `[FROM PAPER]`
- **Reason:** The research paper explicitly specifies `yahooquery` (https://pypi.org/project/yahooquery/) for fetching historical stock data.
- **Paper Reference:** Section 3.1, Page 4.

- **Decision:** Implement a provider abstraction layer (`MarketDataProvider` base class) with `YahooQueryMarketDataProvider` implementation.
- **Classification:** `[OUR DECISION]`
- **Reason:** Decouples pipeline consumers from the specific vendor implementation, allowing seamless integration of alternative providers (e.g., Yahoo Finance, Polygon, Alpha Vantage) if needed in future iterations without changing pipeline logic.

---

## 2. Market Variables & Column Selection

- **Decision:** Retain Open, High, Low, Adjusted Close (`adj_close`), and Volume.
- **Classification:** `[FROM PAPER]`
- **Reason:** The research paper specifies these 5 core market variables.
- **Paper Reference:** Section 3.1, Page 4.

- **Decision:** Normalize field names to lowercase canonical names: `symbol`, `date`, `open`, `high`, `low`, `adj_close`, `volume`.
- **Classification:** `[OUR DECISION]`
- **Reason:** Ensures consistent, deterministic pandas and PyArrow schema handling across raw inputs, validation logic, and processed datasets.

- **Decision:** Explicitly preserve Adjusted Close (`adj_close`) instead of raw Close.
- **Classification:** `[FROM PAPER]`
- **Reason:** The paper specifies "adjusted closing prices", which account for stock splits and dividend distributions.

---

## 3. Stock Universe & Sector Mapping

- **Decision:** Use the paper's 12 stock tickers: ATVI, NFLX, SBUX, TSLA, NVDA, AAPL, AMT, PLD, NDAQ, SCHW, BIO, JNJ.
- **Classification:** `[FROM PAPER]`
- **Reason:** Required for accurate reproduction of the paper's baseline experiments.
- **Paper Reference:** Section 3.1 & Section 4.

- **Decision:** Store stock universe and sector metadata in `configs/stocks.yaml`.
- **Classification:** `[OUR DECISION]`
- **Reason:** Externalized YAML configuration ensures reproducibility, testability, and non-hardcoded pipeline setup.

---

## 4. Date Ranges & Historical Lookback

- **Decision:** Evaluation periods are Uptrend (2021-07-01 to 2021-12-31) and Downtrend (2022-01-01 to 2022-09-20).
- **Classification:** `[FROM PAPER]`
- **Reason:** Explicit evaluation split in the paper.

- **Decision:** Download historical market data starting from `2020-01-01` through `2022-09-30`.
- **Classification:** `[OUR DECISION]`
- **Reason:** Technical indicators (e.g., EMA26, MACD, Stochastic) and LSTM historical sequence lookback require sufficient trading history prior to the evaluation start date (`2021-07-01`).

---

## 5. Storage Format & Immutability

- **Decision:** Store downloaded raw CSV files immutably at `data/raw/market/{SYMBOL}.csv`.
- **Classification:** `[OUR DECISION]`
- **Reason:** Prevents accidental modification or destruction of raw downloaded data.

- **Decision:** Store cleaned, validated, and normalized data as Parquet files at `data/processed/market/{SYMBOL}.parquet` alongside CSV exports.
- **Classification:** `[OUR DECISION]`
- **Reason:** Parquet preserves rich column data types, date formats, compression, and fast I/O performance for downstream feature generation.

---

## 6. Financial Semantics & Quality Rules

- **Decision:** Do NOT fill missing trading days (weekends, holidays) with synthetic forward-filled prices.
- **Classification:** `[OUR DECISION]`
- **Reason:** Financial market series operate on trading-day calendars. Filling weekends with synthetic prices distorts volatility and indicator calculations (e.g. moving averages).

- **Decision:** Validate OHLC relations (`high >= low`, `high >= open`, `low <= open`, `high >= adj_close` accounting for split/dividend adjustments).
- **Classification:** `[OUR DECISION]`
- **Reason:** Prevents corrupted or anomalous data from leaking into feature engineering while properly recognizing split-adjusted historical pricing.
