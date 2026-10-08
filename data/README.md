# Market Dataset & Reproducibility Guide

This directory manages all raw and processed market data for the **Multimodal AI for Stock Trend Prediction** project, strictly adhering to the market data methodology described in:

> _A deep fusion model for stock market prediction with news headlines and time series data_  
> Neural Computing and Applications (2024) 36:21229–21271  
> DOI: 10.1007/s00521-024-10303-1

---

## Directory Structure

```
data/
├── raw/
│   └── market/               # Immutable raw downloaded CSV files from YahooQuery
│       ├── AAPL.csv
│       ├── NVDA.csv
│       └── ... (12 paper stocks)
└── processed/
    └── market/               # Cleaned, schema-normalized Parquet & CSV files
        ├── AAPL.parquet
        ├── AAPL.csv
        ├── ... (12 paper stocks)
        └── manifest.json     # Machine-readable dataset manifest
```

---

## Paper Stock Universe (12 Stocks)

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

## Canonical Schema

| Field       | Type        | Description                  |
| :---------- | :---------- | :--------------------------- |
| `symbol`    | `str`       | Stock ticker symbol          |
| `date`      | `str`       | Trading date (`YYYY-MM-DD`)  |
| `open`      | `float`     | Daily opening price          |
| `high`      | `float`     | Daily high price             |
| `low`       | `float`     | Daily low price              |
| `adj_close` | `float`     | Daily adjusted closing price |
| `volume`    | `float/int` | Daily trading volume         |

---

## How to Reproduce the Dataset

To re-download, validate, and generate the processed datasets and manifests for the paper's stock universe across the configured acquisition date range (`2020-01-01` to `2022-09-30`), run:

```bash
# Activate virtual environment
source .venv/Scripts/activate # Windows PowerShell: .venv\Scripts\Activate.ps1

# Perform a dry run
python -m stock_ai.data.market.download --dry-run

# Run full download & pipeline execution
python -m stock_ai.data.market.download

# Force re-download ignoring existing raw cache
python -m stock_ai.data.market.download --force
```

---

## Data Validation & Reports

- **Manifest:** Machine-readable metadata at `data/processed/market/manifest.json`.
- **Quality Report:** Human-readable quality report at `reports/market_data_quality_report.md`.
