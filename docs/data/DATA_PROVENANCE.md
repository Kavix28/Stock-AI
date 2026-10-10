# Stock-AI Data Provenance and Asset Validation Audit

**Audit Date:** 2026-10-10  
**Overall Status:** **BLOCKED** on 12-Ticker Universe; **PASS** on 11-Ticker Active Universe

---

## 1. Overview and Status Classification

To maintain the highest standard of academic rigor and reproducibility, this document audits the provenance, retrieval mechanisms, corporate actions, and validation status of the 12 equities investigated in Stock-AI.

| Universe Scope        | Equities Count  | Date Range               | Total Sessions            | Status              | Reason                                                                                                                        |
| :-------------------- | :-------------- | :----------------------- | :------------------------ | :------------------ | :---------------------------------------------------------------------------------------------------------------------------- |
| **Active Equities**   | 11              | 2010-02-22 to 2026-10-09 | 4,185 per ticker          | **PASS (VERIFIED)** | Refreshed via YahooQuery / YFinance; zero missing values, zero boundary violations, verified forward test horizon.            |
| **Complete Universe** | 12 (incl. ATVI) | Varies                   | 4,185 (11) / 3,046 (ATVI) | **BLOCKED**         | ATVI delisted Oct 2023; legacy historical data lacks verifiable retrieval metadata; 0 observations in 2024–2026 test horizon. |

---

## 2. Active Equities Provenance (11 Stocks — VERIFIED)

The 11 active equities (`AAPL`, `AMT`, `BIO`, `JNJ`, `NDAQ`, `NFLX`, `NVDA`, `PLD`, `SBUX`, `SCHW`, `TSLA`) were refreshed directly from public market APIs:

- **Primary Provider:** YahooQuery (`yahooquery.Ticker.history`) with automatic fallback to YFinance (`yfinance.download`).
- **Retrieval Date:** 2026-10-10.
- **Licensing:** Standard public market data for non-commercial academic research.
- **Temporal Span:** 2010-02-22 through 2026-10-09 (4,185 daily trading sessions per ticker).
- **Split Partitions:**
  - **Train:** 2010-02-22 to 2022-12-31 (~3,238 sessions)
  - **Validation:** 2023-01-01 to 2024-06-30 (~375 sessions)
  - **Test (Forward Horizon):** 2024-07-01 to 2026-10-09 (572 sessions; **6,292 total observations**)
- **Validation Audit:**
  - Date Monotonicity: 100% strictly chronological, zero duplicate dates.
  - Boundary Invariants: $High \ge Low$, $High \ge Open$, $High \ge Close$, $Low \le Open$, $Low \le Close$ satisfied for 100% of rows.
  - Missing Values: Exactly 0 NaNs across all 16 technical indicators (first 33 warmup rows dropped).

---

## 3. Activision Blizzard (`ATVI`) Provenance Audit — BLOCKED

### A. Asset Background & Corporate Action

Activision Blizzard, Inc. (`ATVI`) was acquired by Microsoft Corporation for $68.7 billion. The acquisition officially closed on **October 13, 2023**, at which point ATVI was delisted from the Nasdaq stock exchange and ceased trading as an independent equity.

### B. Legacy Data File Audit

- **File Location:** `data/processed/market/ATVI.csv`
- **Total Records:** 3,046 daily trading sessions.
- **Date Range:** 2010-08-16 to 2022-09-20.
- **Forward Horizon Coverage (2024–2026):** **0 trading sessions**.

### C. Missing Provenance Artifacts

1. **Original URL & Provider:** The legacy file originates from the initial repository's historical archive. No recorded HTTP download logs, original provider API call parameters, or retrieval timestamps exist.
2. **Corporate Action Adjustment:** While prices satisfy basic OHLC bounds ($High \ge Low$), it cannot be independently proven whether historical stock splits (e.g., 2-for-1 splits) or special cash dividends were adjusted according to Center for Research in Security Prices (CRSP) standards.
3. **Evaluation Alignment:** The legacy file covers the historical paper's 2022 test period (180 dates from 2022-01-03 to 2022-09-20), but cannot participate in the out-of-sample forward test horizon (2024–2026).

### D. Audit Decision

In accordance with scientific integrity standards:

- **12-Ticker Universe Validation is officially marked as BLOCKED.**
- Results for the paper's original 12-ticker scope cannot be claimed as fully replicated until an authoritative, verifiable historical data provider (e.g., Wharton Research Data Services / CRSP) is integrated.
- The 11 active equities form the sole **VERIFIED** benchmark of this platform.
