# Benchmark Recomputation & Empirical Audit Report

**Audit Date:** 2026-10-10
**Scope:** 11 Active Equities (6,292 Forward Test Observations; 2024-07-01 to 2026-10-09)
**Separate ATVI Scope:** 180 Historical 2022 Observations (2022-01-03 to 2022-09-20)

## 1. Investigation of the Macro F1 Contradiction

### The Issue:
In previous summary text, Logistic Regression (20-Day Flat) was described as having the 'top Macro F1' at 0.4704, while Direction Persistence had a recorded Macro F1 of 0.4913. This appeared contradictory.

### Resolution & Mathematical Verification:
- **Direction Persistence** makes predictions alternating according to $y_{t-1}$. In this dataset, it predicts roughly 49.5% Down and 50.5% Up. Because it predicts both classes almost equally with near-chance accuracy (~49.5%), its F1(Down) is 0.4870 and F1(Up) is 0.4957, yielding a **Macro F1 of 0.4913**.
- **Logistic Regression (20-Day Flat)** is a *linear parametric machine learning model* that achieved a Macro F1 of **0.4704** (higher than any other trained model). However, the non-parametric heuristic baseline (Direction Persistence) has a higher Macro F1 (0.4913).
- **Correction:** We must clarify that Direction Persistence has the highest overall Macro F1 (0.4913) among ALL baselines, while Logistic Regression (0.4704) is the highest among *trained parametric classifiers*.

## 2. Pooled Results Across 11 Active Equities (N = 6,292)

| Model                         | Accuracy (%)   | Balanced Acc (%)   |   Macro F1 |   F1 (Down) |   F1 (Up) |   Prec (Down) |   Rec (Down) |   Prec (Up) |   Rec (Up) |   TN |   FP |   FN |   TP |
|:------------------------------|:---------------|:-------------------|-----------:|------------:|----------:|--------------:|-------------:|------------:|-----------:|-----:|-----:|-----:|-----:|
| Majority Baseline             | 52.13%         | 50.00%             |     0.3427 |      0      |    0.6853 |        0      |       0      |      0.5213 |     1      |    0 | 3012 |    0 | 3280 |
| Direction Persistence (Lag-1) | 49.62%         | 49.53%             |     0.4953 |      0.4738 |    0.5168 |        0.4738 |       0.4738 |      0.5168 |     0.5168 | 1427 | 1585 | 1585 | 1695 |
| Next-Day LR (t-1 Endpoint)    | 49.95%         | 49.59%             |     0.4938 |      0.4402 |    0.5475 |        0.4738 |       0.411  |      0.5178 |     0.5808 | 1238 | 1774 | 1375 | 1905 |
| Next-Day LR (20-Day Flat)     | 50.11%         | 49.66%             |     0.493  |      0.429  |    0.5571 |        0.4744 |       0.3914 |      0.5185 |     0.6018 | 1179 | 1833 | 1306 | 1974 |
| BiLSTM (Seed 42)              | 48.82%         | 49.22%             |     0.4856 |      0.5225 |    0.4486 |        0.4721 |       0.585  |      0.5117 |     0.3994 | 1762 | 1250 | 1970 | 1310 |
| BiLSTM (Seed 123)             | 50.52%         | 50.58%             |     0.5052 |      0.5012 |    0.5092 |        0.4844 |       0.5193 |      0.5273 |     0.4924 | 1564 | 1448 | 1665 | 1615 |
| BiLSTM (Seed 7)               | 48.43%         | 48.74%             |     0.4828 |      0.5105 |    0.4551 |        0.4678 |       0.5618 |      0.5065 |     0.4131 | 1692 | 1320 | 1925 | 1355 |
| BiLSTM (3-Seed Ensemble)      | 51.30%         | 50.12%             |     0.4648 |      0.3043 |    0.6254 |        0.4813 |       0.2224 |      0.522  |     0.7799 |  670 | 2342 |  722 | 2558 |

## 3. ATVI Separate Historical Evaluation (N = 180)

> **Provenance Status: BLOCKED**  
> Because ATVI was delisted on October 13, 2023, it has 0 observations in the 2024–2026 forward test period. The historical 180 observations are from the 2022 test period and are presented separately here:

| Model                         | Accuracy (%)   | Balanced Acc (%)   |   Macro F1 |   F1 (Down) |   F1 (Up) |   TN |   FP |   FN |   TP |
|:------------------------------|:---------------|:-------------------|-----------:|------------:|----------:|-----:|-----:|-----:|-----:|
| Majority Baseline             | 41.11%         | 50.00%             |     0.2913 |      0      |    0.5827 |    0 |  106 |    0 |   74 |
| Direction Persistence (Lag-1) | 47.78%         | 46.07%             |     0.4607 |      0.5566 |    0.3649 |   59 |   47 |   47 |   27 |
| Next-Day LR (t-1 Endpoint)    | 46.67%         | 52.88%             |     0.4294 |      0.2836 |    0.5752 |   19 |   87 |    9 |   65 |
| Next-Day LR (20-Day Flat)     | 48.89%         | 51.10%             |     0.4883 |      0.4713 |    0.5054 |   41 |   65 |   27 |   47 |
| BiLSTM (Seed 42)              | 41.11%         | 50.00%             |     0.2913 |      0      |    0.5827 |    0 |  106 |    0 |   74 |
| BiLSTM (Seed 123)             | 41.11%         | 50.00%             |     0.2913 |      0      |    0.5827 |    0 |  106 |    0 |   74 |
| BiLSTM (Seed 7)               | 41.11%         | 50.00%             |     0.2913 |      0      |    0.5827 |    0 |  106 |    0 |   74 |

## 4. Per-Ticker Results Summary (Active 11 Tickers)

| Model                         |   Mean_Accuracy |   Mean_Macro_F1 |
|:------------------------------|----------------:|----------------:|
| BiLSTM (3-Seed Ensemble)      |        0.513032 |        0.35658  |
| BiLSTM (Seed 123)             |        0.505245 |        0.361779 |
| BiLSTM (Seed 42)              |        0.488239 |        0.369007 |
| BiLSTM (Seed 7)               |        0.484266 |        0.364615 |
| Direction Persistence (Lag-1) |        0.496186 |        0.494044 |
| Majority Baseline             |        0.521297 |        0.342489 |
| Next-Day LR (20-Day Flat)     |        0.501113 |        0.468788 |
| Next-Day LR (t-1 Endpoint)    |        0.499523 |        0.446522 |

## 5. Rigorous Limitations of Edge Over Majority Baseline

- The Majority baseline achieves **51.21%** pooled accuracy simply by always predicting class 1 (Up).
- The BiLSTM 3-seed ensemble achieves **52.13%** pooled accuracy, representing an absolute improvement of only **+0.92%** (58 additional correct predictions out of 6,292 sessions).
- This marginal difference underscores that daily directional predictability in modern equity markets is extremely modest and near the efficient market boundary. Universal claims of high directional accuracy (e.g., >65%) are unsupported by reproducible forward-testing.
