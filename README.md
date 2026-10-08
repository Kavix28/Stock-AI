# Multimodal AI for Stock Trend Prediction

> Research reproduction baseline and implementation for deep multimodal fusion of stock market prices, technical indicators, and financial news sentiment.

**Research Paper:** _A deep fusion model for stock market prediction with news headlines and time series data_ (Neural Computing and Applications, 2024).  
**DOI:** [10.1007/s00521-024-10303-1](https://doi.org/10.1007/s00521-024-10303-1)

---

## Phase 2 — Market Data Acquisition & Quality Pipeline

Phase 2 builds a reproducible, testable, and robust market data pipeline for fetching, validating, and normalizing market data for the paper's stock universe.

### Key Architecture Features

- **Provider Abstraction:** Extensible `MarketDataProvider` architecture with `YahooQueryMarketDataProvider`.
- **Immutable Raw Storage:** Raw downloaded CSV files are preserved in `data/raw/market/{SYMBOL}.csv`.
- **Canonical Schema Normalization:** Lowercase standard schema (`symbol`, `date`, `open`, `high`, `low`, `adj_close`, `volume`).
- **Adjusted Close Preservation:** Strictly preserves Adjusted Close (`adj_close`) as specified by the paper.
- **Financial Quality Verification:** Validates OHLC relationships (`high >= low`, `open` bounds), duplicate dates, NaN values, and temporal sequence integrity.
- **Trading-Day Gap Analysis:** Distinguishes between expected weekend non-trading days and unexpected gaps.
- **Dataset Manifest & Quality Reporting:** Automatically writes `data/processed/market/manifest.json` and `reports/market_data_quality_report.md`.

---

### Paper Stock Universe (12 Stocks)

| Ticker   | Name                 | Sector                 |
| :------- | :------------------- | :--------------------- |
| **ATVI** | Activision Blizzard  | Communication Services |
| **NFLX** | Netflix              | Communication Services |
| **SBUX** | Starbucks            | Consumer Discretionary |
| **TSLA** | Tesla                | Consumer Discretionary |
| **NVDA** | NVIDIA               | Information Technology |
| **AAPL** | Apple                | Information Technology |
| **AMT**  | American Tower       | Real Estate            |
| **PLD**  | Prologis             | Real Estate            |
| **NDAQ** | Nasdaq               | Financials             |
| **SCHW** | Charles Schwab       | Financials             |
| **BIO**  | Bio-Rad Laboratories | Health Care            |
| **JNJ**  | Johnson & Johnson    | Health Care            |

---

### Configured Date Ranges

- **Acquisition Range (Historical Lookback Included):** `2020-01-01` to `2022-09-30`
- **Uptrend Evaluation Period:** `2021-07-01` to `2021-12-31`
- **Downtrend Evaluation Period:** `2022-01-01` to `2022-09-20`

---

### Installation & Environment Setup

```bash
# Clone repository and create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # On Windows PowerShell: .venv\Scripts\Activate.ps1

# Install package and dependencies
pip install -e .
```

---

### Usage & Pipeline Commands

```bash
# Run unit and integration tests
pytest

# Dry run market data acquisition
python -m stock_ai.data.market.download --dry-run

# Run full market data acquisition pipeline for all 12 stocks
python -m stock_ai.data.market.download

# Force re-download ignoring raw cache
python -m stock_ai.data.market.download --force
```

---

### Output Locations

- **Raw Data:** `data/raw/market/{SYMBOL}.csv`
- **Processed Parquet Data:** `data/processed/market/{SYMBOL}.parquet`
- **Processed CSV Data:** `data/processed/market/{SYMBOL}.csv`
- **Dataset Manifest:** `data/processed/market/manifest.json`
- **Quality Assurance Report:** `reports/market_data_quality_report.md`

---

### Troubleshooting

- **YahooQuery API Throttling / Rate Limits:** The provider includes configurable exponential backoff retries (`retry_attempts: 3`) and per-request rate-limiting delays (`rate_limit_delay: 0.5s`).
- **Unexpected Gaps Warning:** Weekends are expected non-trading days and are NOT filled with artificial prices. Unexpected weekday gaps trigger a warning in the quality report.
- **Cache Invalidation:** Use `--force` to purge raw caches and re-fetch directly from YahooQuery.
