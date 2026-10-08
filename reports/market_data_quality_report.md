# Market Data Quality Assurance Report

**Generated:** 2026-10-08 20:04:20 UTC
**Provider:** YahooQuery
**Target Range:** 2020-01-01 to 2022-09-30

## Summary Table

| Symbol | Status | Rows | First Date | Last Date | Missing | Dups | Unexpected Gaps |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ATVI** | FAILED | 0 | - | - | - | - | - |
| **NFLX** | PASS | 692 | 2020-01-02 | 2022-09-29 | 0 | 0 | 0 |
| **SBUX** | PASS | 692 | 2020-01-02 | 2022-09-29 | 0 | 0 | 0 |
| **TSLA** | PASS | 692 | 2020-01-02 | 2022-09-29 | 0 | 0 | 0 |
| **NVDA** | PASS | 692 | 2020-01-02 | 2022-09-29 | 0 | 0 | 0 |
| **AAPL** | PASS | 692 | 2020-01-02 | 2022-09-29 | 0 | 0 | 0 |
| **AMT** | PASS | 692 | 2020-01-02 | 2022-09-29 | 0 | 0 | 0 |
| **PLD** | PASS | 692 | 2020-01-02 | 2022-09-29 | 0 | 0 | 0 |
| **NDAQ** | PASS | 692 | 2020-01-02 | 2022-09-29 | 0 | 0 | 0 |
| **SCHW** | PASS | 692 | 2020-01-02 | 2022-09-29 | 0 | 0 | 0 |
| **BIO** | PASS | 692 | 2020-01-02 | 2022-09-29 | 0 | 0 | 0 |
| **JNJ** | PASS | 692 | 2020-01-02 | 2022-09-29 | 0 | 0 | 0 |

## Overall Quality Status

- **Total Stocks Audited:** 12
- **Passed Validation:** 11 / 12
- **Total Rows Ingested:** 7612
- **Pipeline Status:** ACTION REQUIRED

## Detailed Stock Quality Breakdown

### ATVI
- **Status:** FAILED
- **Error:** Failed to fetch market data for 'ATVI' after 3 attempts. Error: No market data returned by YahooQuery for ticker 'ATVI'.

### NFLX
- **Status:** PASS
- **Rows:** 692
- **Date Range:** 2020-01-02 to 2022-09-29
- **Missing Values:** 0
- **Duplicates Removed:** 0
- **Unexpected Trading Gaps:** 0
- **Raw File:** `data\raw\market\NFLX.csv`
- **Processed File:** `data\processed\market\NFLX.parquet`
- **Validation Findings:**
  - [INFO] (Temporal) All date gaps align with expected weekend non-trading days (145 weekends).

### SBUX
- **Status:** PASS
- **Rows:** 692
- **Date Range:** 2020-01-02 to 2022-09-29
- **Missing Values:** 0
- **Duplicates Removed:** 0
- **Unexpected Trading Gaps:** 0
- **Raw File:** `data\raw\market\SBUX.csv`
- **Processed File:** `data\processed\market\SBUX.parquet`
- **Validation Findings:**
  - [INFO] (Temporal) All date gaps align with expected weekend non-trading days (145 weekends).

### TSLA
- **Status:** PASS
- **Rows:** 692
- **Date Range:** 2020-01-02 to 2022-09-29
- **Missing Values:** 0
- **Duplicates Removed:** 0
- **Unexpected Trading Gaps:** 0
- **Raw File:** `data\raw\market\TSLA.csv`
- **Processed File:** `data\processed\market\TSLA.parquet`
- **Validation Findings:**
  - [INFO] (Temporal) All date gaps align with expected weekend non-trading days (145 weekends).

### NVDA
- **Status:** PASS
- **Rows:** 692
- **Date Range:** 2020-01-02 to 2022-09-29
- **Missing Values:** 0
- **Duplicates Removed:** 0
- **Unexpected Trading Gaps:** 0
- **Raw File:** `data\raw\market\NVDA.csv`
- **Processed File:** `data\processed\market\NVDA.parquet`
- **Validation Findings:**
  - [INFO] (Temporal) All date gaps align with expected weekend non-trading days (145 weekends).

### AAPL
- **Status:** PASS
- **Rows:** 692
- **Date Range:** 2020-01-02 to 2022-09-29
- **Missing Values:** 0
- **Duplicates Removed:** 0
- **Unexpected Trading Gaps:** 0
- **Raw File:** `data\raw\market\AAPL.csv`
- **Processed File:** `data\processed\market\AAPL.parquet`
- **Validation Findings:**
  - [INFO] (Temporal) All date gaps align with expected weekend non-trading days (145 weekends).

### AMT
- **Status:** PASS
- **Rows:** 692
- **Date Range:** 2020-01-02 to 2022-09-29
- **Missing Values:** 0
- **Duplicates Removed:** 0
- **Unexpected Trading Gaps:** 0
- **Raw File:** `data\raw\market\AMT.csv`
- **Processed File:** `data\processed\market\AMT.parquet`
- **Validation Findings:**
  - [INFO] (Temporal) All date gaps align with expected weekend non-trading days (145 weekends).

### PLD
- **Status:** PASS
- **Rows:** 692
- **Date Range:** 2020-01-02 to 2022-09-29
- **Missing Values:** 0
- **Duplicates Removed:** 0
- **Unexpected Trading Gaps:** 0
- **Raw File:** `data\raw\market\PLD.csv`
- **Processed File:** `data\processed\market\PLD.parquet`
- **Validation Findings:**
  - [INFO] (Temporal) All date gaps align with expected weekend non-trading days (145 weekends).

### NDAQ
- **Status:** PASS
- **Rows:** 692
- **Date Range:** 2020-01-02 to 2022-09-29
- **Missing Values:** 0
- **Duplicates Removed:** 0
- **Unexpected Trading Gaps:** 0
- **Raw File:** `data\raw\market\NDAQ.csv`
- **Processed File:** `data\processed\market\NDAQ.parquet`
- **Validation Findings:**
  - [INFO] (Temporal) All date gaps align with expected weekend non-trading days (145 weekends).

### SCHW
- **Status:** PASS
- **Rows:** 692
- **Date Range:** 2020-01-02 to 2022-09-29
- **Missing Values:** 0
- **Duplicates Removed:** 0
- **Unexpected Trading Gaps:** 0
- **Raw File:** `data\raw\market\SCHW.csv`
- **Processed File:** `data\processed\market\SCHW.parquet`
- **Validation Findings:**
  - [INFO] (Temporal) All date gaps align with expected weekend non-trading days (145 weekends).

### BIO
- **Status:** PASS
- **Rows:** 692
- **Date Range:** 2020-01-02 to 2022-09-29
- **Missing Values:** 0
- **Duplicates Removed:** 0
- **Unexpected Trading Gaps:** 0
- **Raw File:** `data\raw\market\BIO.csv`
- **Processed File:** `data\processed\market\BIO.parquet`
- **Validation Findings:**
  - [INFO] (Temporal) All date gaps align with expected weekend non-trading days (145 weekends).

### JNJ
- **Status:** PASS
- **Rows:** 692
- **Date Range:** 2020-01-02 to 2022-09-29
- **Missing Values:** 0
- **Duplicates Removed:** 0
- **Unexpected Trading Gaps:** 0
- **Raw File:** `data\raw\market\JNJ.csv`
- **Processed File:** `data\processed\market\JNJ.parquet`
- **Validation Findings:**
  - [INFO] (Temporal) All date gaps align with expected weekend non-trading days (145 weekends).
