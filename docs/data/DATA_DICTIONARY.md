# Stock-AI Data Dictionary

This document defines the schema, data types, units, and mathematical definitions of all fields used across Stock-AI datasets.

---

## 1. Raw Market Data Schema

| Field Name | Type                | Description                                | Constraints                          |
| :--------- | :------------------ | :----------------------------------------- | :----------------------------------- |
| `date`     | String (YYYY-MM-DD) | Trading session date                       | Monotonically increasing             |
| `open`     | Float               | Session opening price in USD               | $> 0$                                |
| `high`     | Float               | Session highest recorded price in USD      | $\ge open, \ge low, \ge close$       |
| `low`      | Float               | Session lowest recorded price in USD       | $\le open, \le high, \le close, > 0$ |
| `close`    | Float               | Session unadjusted closing price in USD    | $> 0$                                |
| `adjclose` | Float               | Dividend- and split-adjusted closing price | $> 0$                                |
| `volume`   | Integer             | Total shares traded during session         | $\ge 0$                              |

---

## 2. Processed Features Schema (16 Core Indicators)

| Feature Name  | Category   | Mathematical Formulation                                          | Lookahead Risk Mitigation |
| :------------ | :--------- | :---------------------------------------------------------------- | :------------------------ |
| `open`        | Price      | Raw session open                                                  | $t-1$ cutoff enforced     |
| `high`        | Price      | Raw session high                                                  | $t-1$ cutoff enforced     |
| `low`         | Price      | Raw session low                                                   | $t-1$ cutoff enforced     |
| `close`       | Price      | Raw session close                                                 | $t-1$ cutoff enforced     |
| `adjclose`    | Price      | Adjusted close                                                    | $t-1$ cutoff enforced     |
| `volume`      | Volume     | Raw session volume                                                | $t-1$ cutoff enforced     |
| `12-Day EMA`  | Trend      | $EMA_t = \alpha P_t + (1-\alpha)EMA_{t-1}, \alpha = \frac{2}{13}$ | Backward recursive only   |
| `26-Day EMA`  | Trend      | $EMA_t = \alpha P_t + (1-\alpha)EMA_{t-1}, \alpha = \frac{2}{27}$ | Backward recursive only   |
| `MACD`        | Momentum   | $EMA_{12}(P) - EMA_{26}(P)$                                       | Backward recursive only   |
| `rsi`         | Momentum   | $100 - \frac{100}{1 + RS_{14}}$ (Wilder smoothing)                | 14-period lookback only   |
| `Upper Band`  | Volatility | $SMA_{20}(Close) + 2 \sigma_{20}(Close)$                          | 20-period lookback only   |
| `Middle Band` | Volatility | $SMA_{20}(Close)$                                                 | 20-period lookback only   |
| `Lower Band`  | Volatility | $SMA_{20}(Close) - 2 \sigma_{20}(Close)$                          | 20-period lookback only   |
| `SAR`         | Trend      | Parabolic Stop and Reverse (Wilder)                               | Stepwise acceleration     |
| `Slow k`      | Momentum   | 3-period rolling average of Fast %K                               | 14-period window          |
| `Slow d`      | Momentum   | 3-period rolling average of Slow %K                               | 3-period window           |

---

## 3. Target Variables (Excluded from Model Features)

| Column         | Type    | Definition                          | Usage                   |
| :------------- | :------ | :---------------------------------- | :---------------------- |
| `PrevDayClose` | Float   | $Close_{t-1}$                       | Ground truth reference  |
| `Diff`         | Integer | $\mathbb{I}(Close_t > Close_{t-1})$ | Supervised target $y_t$ |
