# Research Paper Baseline — Market Data Methodology

**Paper Title:** A deep fusion model for stock market prediction with news headlines and time series data  
**Journal:** Neural Computing and Applications (2024) 36:21229–21271  
**Authors:** Pinyu Chen, Zois Boukouvalas, Roberto Corizzo  
**DOI:** 10.1007/s00521-024-10303-1

---

## Market Data Methodology

### 1. Market Data Provider

[FROM PAPER]

- **Provider:** Yahooquery Python package (`https://pypi.org/project/yahooquery/`)
- **Role:** Serves as an unofficial API client for retrieving historical stock prices.

### 2. Market Variables

[FROM PAPER]

- **Open:** Daily opening price
- **High:** Daily high value
- **Low:** Daily low value
- **Adjusted Close (`adj_close`):** Daily adjusted closing price (accounts for corporate actions, splits, dividends)
- **Volume:** Daily observed trading volume

### 3. Stock Universe

[FROM PAPER]
12 US stocks selected across 11 major sectors/industries:

- **ATVI:** Activision Blizzard (Communication Services)
- **NFLX:** Netflix, Inc. (Communication Services)
- **SBUX:** Starbucks Corporation (Consumer Discretionary)
- **TSLA:** Tesla, Inc. (Consumer Discretionary)
- **NVDA:** NVIDIA Corporation (Information Technology)
- **AAPL:** Apple Inc. (Information Technology)
- **AMT:** American Tower Corporation (Real Estate)
- **PLD:** Prologis, Inc. (Real Estate)
- **NDAQ:** Nasdaq, Inc. (Financials)
- **SCHW:** The Charles Schwab Corporation (Financials)
- **BIO:** Bio-Rad Laboratories, Inc. (Health Care)
- **JNJ:** Johnson & Johnson (Health Care)

### 4. Evaluation Periods & Market Conditions

[FROM PAPER]

- **Uptrend Period:** July 1, 2021 through December 31, 2021
- **Downtrend Period:** January 1, 2022 through September 20, 2022

### 5. Data Frequency

[FROM PAPER]

- **Frequency:** Daily ("daily stock prices")

### 6. Preprocessing & Downstream Feature Engineering

[FROM PAPER]

- **Normalization:** Min-max normalization (0, 1) applied to numerical features before model training.
- **Technical Indicators (to be added in Phase 3):** Computed via TA-Lib:
  - EMA 12 (Exponential Moving Average 12-day)
  - EMA 26 (Exponential Moving Average 26-day)
  - MACD (Moving Average Convergence/Divergence)
  - Parabolic SAR
  - Bollinger Bands (upper, middle, lower)
  - Stochastic Slow K
  - Stochastic Slow D

### 7. Historical Lookback & Split Strategy

[FROM PAPER]

- **Lookback / Sequence Length:** Not explicitly specified in the paper (optimized during hyperparameter tuning).
- **Evaluation Scheme:** Sliding window evaluation where models predict the next-day trend using historical multivariate sequence data.

---

### Unspecified Information Summary

- **Exact historical lookback start date before 2021-07-01:** Not explicitly specified in the paper.
- **Handling of zero volume days / trading halts:** Not explicitly specified in the paper.
