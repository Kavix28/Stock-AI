# Market Data Quality Assurance Report

**Generated:** 2026-10-09 14:35:33 UTC  
**Target Range:** 2020-01-01 to 2022-09-30  

## Summary Table

| Symbol | Provider | Status | Rows | First Date | Last Date | Missing | Dups | Unexpected Gaps |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ATVI** | YahooQuery | FAILED | 0 | - | - | - | 0 | - |
| **NFLX** | YahooQuery | PASS | 692 | 2020-01-02 | 2022-09-29 | 0 | 0 | 0 |
| **SBUX** | YahooQuery | PASS | 692 | 2020-01-02 | 2022-09-29 | 0 | 0 | 0 |
| **TSLA** | YahooQuery | PASS | 692 | 2020-01-02 | 2022-09-29 | 0 | 0 | 0 |
| **NVDA** | YahooQuery | PASS | 692 | 2020-01-02 | 2022-09-29 | 0 | 0 | 0 |
| **AAPL** | YahooQuery | PASS | 692 | 2020-01-02 | 2022-09-29 | 0 | 0 | 0 |
| **AMT** | YahooQuery | PASS | 692 | 2020-01-02 | 2022-09-29 | 0 | 0 | 0 |
| **PLD** | YahooQuery | PASS | 692 | 2020-01-02 | 2022-09-29 | 0 | 0 | 0 |
| **NDAQ** | YahooQuery | PASS | 692 | 2020-01-02 | 2022-09-29 | 0 | 0 | 0 |
| **SCHW** | YahooQuery | PASS | 692 | 2020-01-02 | 2022-09-29 | 0 | 0 | 0 |
| **BIO** | YahooQuery | PASS | 692 | 2020-01-02 | 2022-09-29 | 0 | 0 | 0 |
| **JNJ** | YahooQuery | PASS | 692 | 2020-01-02 | 2022-09-29 | 0 | 0 | 0 |

## Overall Quality Status

- **Total Stocks Audited:** 12
- **Passed Validation:** 11 / 12
- **Total Rows Ingested:** 7612
- **Pipeline Status:** ACTION REQUIRED

## Detailed Stock Quality Breakdown

### ATVI
- **Provider:** YahooQuery
- **Status:** FAILED
- **Error:** Failed to fetch market data for 'ATVI' after 3 attempts. Error: No market data returned by YahooQuery for ticker 'ATVI'.
- **Validation Findings:**
  - [ERROR] (FETCH) Failed to fetch market data for 'ATVI' after 3 attempts. Error: No market data returned by YahooQuery for ticker 'ATVI'.

### NFLX
- **Provider:** YahooQuery
- **Status:** PASS
- **Rows:** 692
- **Date Range:** 2020-01-02 to 2022-09-29
- **Missing Values:** 0
- **Unexpected Trading Gaps:** 0
- **Raw File:** `data\raw\market\NFLX.csv`
- **Processed File:** `data\processed\market\NFLX.parquet`
- **Validation Findings:**
  - [INFO] (Temporal) All date gaps align with expected weekend non-trading days (145 weekends).

### SBUX
- **Provider:** YahooQuery
- **Status:** PASS
- **Rows:** 692
- **Date Range:** 2020-01-02 to 2022-09-29
- **Missing Values:** 0
- **Unexpected Trading Gaps:** 0
- **Raw File:** `data\raw\market\SBUX.csv`
- **Processed File:** `data\processed\market\SBUX.parquet`
- **Validation Findings:**
  - [INFO] (Temporal) All date gaps align with expected weekend non-trading days (145 weekends).

### TSLA
- **Provider:** YahooQuery
- **Status:** PASS
- **Rows:** 692
- **Date Range:** 2020-01-02 to 2022-09-29
- **Missing Values:** 0
- **Unexpected Trading Gaps:** 0
- **Raw File:** `data\raw\market\TSLA.csv`
- **Processed File:** `data\processed\market\TSLA.parquet`
- **Validation Findings:**
  - [INFO] (Temporal) All date gaps align with expected weekend non-trading days (145 weekends).

### NVDA
- **Provider:** YahooQuery
- **Status:** PASS
- **Rows:** 692
- **Date Range:** 2020-01-02 to 2022-09-29
- **Missing Values:** 0
- **Unexpected Trading Gaps:** 0
- **Raw File:** `data\raw\market\NVDA.csv`
- **Processed File:** `data\processed\market\NVDA.parquet`
- **Validation Findings:**
  - [INFO] (Temporal) All date gaps align with expected weekend non-trading days (145 weekends).

### AAPL
- **Provider:** YahooQuery
- **Status:** PASS
- **Rows:** 692
- **Date Range:** 2020-01-02 to 2022-09-29
- **Missing Values:** 0
- **Unexpected Trading Gaps:** 0
- **Raw File:** `data\raw\market\AAPL.csv`
- **Processed File:** `data\processed\market\AAPL.parquet`
- **Validation Findings:**
  - [INFO] (Temporal) All date gaps align with expected weekend non-trading days (145 weekends).

### AMT
- **Provider:** YahooQuery
- **Status:** PASS
- **Rows:** 692
- **Date Range:** 2020-01-02 to 2022-09-29
- **Missing Values:** 0
- **Unexpected Trading Gaps:** 0
- **Raw File:** `data\raw\market\AMT.csv`
- **Processed File:** `data\processed\market\AMT.parquet`
- **Validation Findings:**
  - [INFO] (Temporal) All date gaps align with expected weekend non-trading days (145 weekends).

### PLD
- **Provider:** YahooQuery
- **Status:** PASS
- **Rows:** 692
- **Date Range:** 2020-01-02 to 2022-09-29
- **Missing Values:** 0
- **Unexpected Trading Gaps:** 0
- **Raw File:** `data\raw\market\PLD.csv`
- **Processed File:** `data\processed\market\PLD.parquet`
- **Validation Findings:**
  - [INFO] (Temporal) All date gaps align with expected weekend non-trading days (145 weekends).

### NDAQ
- **Provider:** YahooQuery
- **Status:** PASS
- **Rows:** 692
- **Date Range:** 2020-01-02 to 2022-09-29
- **Missing Values:** 0
- **Unexpected Trading Gaps:** 0
- **Raw File:** `data\raw\market\NDAQ.csv`
- **Processed File:** `data\processed\market\NDAQ.parquet`
- **Validation Findings:**
  - [INFO] (Temporal) All date gaps align with expected weekend non-trading days (145 weekends).

### SCHW
- **Provider:** YahooQuery
- **Status:** PASS
- **Rows:** 692
- **Date Range:** 2020-01-02 to 2022-09-29
- **Missing Values:** 0
- **Unexpected Trading Gaps:** 0
- **Raw File:** `data\raw\market\SCHW.csv`
- **Processed File:** `data\processed\market\SCHW.parquet`
- **Validation Findings:**
  - [INFO] (Temporal) All date gaps align with expected weekend non-trading days (145 weekends).

### BIO
- **Provider:** YahooQuery
- **Status:** PASS
- **Rows:** 692
- **Date Range:** 2020-01-02 to 2022-09-29
- **Missing Values:** 0
- **Unexpected Trading Gaps:** 0
- **Raw File:** `data\raw\market\BIO.csv`
- **Processed File:** `data\processed\market\BIO.parquet`
- **Validation Findings:**
  - [INFO] (Temporal) All date gaps align with expected weekend non-trading days (145 weekends).

### JNJ
- **Provider:** YahooQuery
- **Status:** PASS
- **Rows:** 692
- **Date Range:** 2020-01-02 to 2022-09-29
- **Missing Values:** 0
- **Unexpected Trading Gaps:** 0
- **Raw File:** `data\raw\market\JNJ.csv`
- **Processed File:** `data\processed\market\JNJ.parquet`
- **Validation Findings:**
  - [INFO] (Temporal) All date gaps align with expected weekend non-trading days (145 weekends).
