# Benchmark Recomputation & Empirical Audit Report

**Audit Date:** 2026-10-10  
**Scope:** 11 Active Equities (6,292 Forward Test Observations; 2024-07-01 to 2026-10-09)  
**Separate ATVI Scope:** 180 Historical 2022 Observations (2022-01-03 to 2022-09-20)

---

## 1. Investigation of the Macro F1 Contradiction

### The Issue:

In earlier drafts, Logistic Regression was described as having the "top Macro F1" (0.4704 in initial runs, 0.4930 in recomputation), while Direction Persistence had a recorded Macro F1 of 0.4953.

### Resolution & Verification:

- **Direction Persistence (Lag-1)** predicts $\hat{y}_t = y_{t-1}$. Because it predicts both classes almost equally (~49.5% Down and 50.5% Up) with near-chance accuracy, its F1(Down) is 0.4738 and F1(Up) is 0.5168, yielding a **Macro F1 of 0.4953**.
- **Logistic Regression (20-Day Flat)** is a linear parametric classifier that achieved a Macro F1 of **0.4930**.
- **BiLSTM (Seed 123)** achieved a Macro F1 of **0.5052** with balanced predictions (F1 Down: 0.5012, F1 Up: 0.5092).
- **Clarification:** Direction Persistence achieves the highest Macro F1 (0.4953) among _non-parametric baselines_, while Logistic Regression achieves 0.4930 among _linear classifiers_.

---

## 2. Pooled Results Across 11 Active Equities (N = 6,292)

| Model                             | Accuracy (%) | Balanced Acc (%) | Macro F1   | F1 (Down) | F1 (Up) | Prec (Down) | Rec (Down) | Prec (Up) | Rec (Up) | TN    | FP    | FN    | TP    |
| :-------------------------------- | :----------- | :--------------- | :--------- | :-------- | :------ | :---------- | :--------- | :-------- | :------- | :---- | :---- | :---- | :---- |
| **Majority Baseline**             | **52.13%**   | 50.00%           | 0.3427     | 0.0000    | 0.6853  | 0.0000      | 0.0000     | 0.5213    | 1.0000   | 0     | 3,012 | 0     | 3,280 |
| **BiLSTM (3-Seed Ensemble)**      | 51.30%       | 50.12%           | 0.4648     | 0.3043    | 0.6254  | 0.4813      | 0.2224     | 0.5220    | 0.7799   | 670   | 2,342 | 722   | 2,558 |
| **BiLSTM (Seed 123)**             | 50.52%       | 50.58%           | **0.5052** | 0.5012    | 0.5092  | 0.4844      | 0.5193     | 0.5273    | 0.4924   | 1,564 | 1,448 | 1,665 | 1,615 |
| **Next-Day LR (20-Day Flat)**     | 50.11%       | 49.66%           | 0.4930     | 0.4290    | 0.5571  | 0.4744      | 0.3914     | 0.5185    | 0.6018   | 1,179 | 1,833 | 1,306 | 1,974 |
| **Next-Day LR (t-1 Endpoint)**    | 49.95%       | 49.59%           | 0.4938     | 0.4402    | 0.5475  | 0.4738      | 0.4110     | 0.5178    | 0.5808   | 1,238 | 1,774 | 1,375 | 1,905 |
| **Direction Persistence (Lag-1)** | 49.62%       | 49.53%           | 0.4953     | 0.4738    | 0.5168  | 0.4738      | 0.4738     | 0.5168    | 0.5168   | 1,427 | 1,585 | 1,585 | 1,695 |
| **BiLSTM (Seed 42)**              | 48.82%       | 49.22%           | 0.4856     | 0.5225    | 0.4486  | 0.4721      | 0.5850     | 0.5117    | 0.3994   | 1,762 | 1,250 | 1,970 | 1,310 |
| **BiLSTM (Seed 7)**               | 48.43%       | 48.74%           | 0.4828     | 0.5105    | 0.4551  | 0.4678      | 0.5618     | 0.5065    | 0.4131   | 1,692 | 1,320 | 1,925 | 1,355 |

---

## 3. ATVI Separate Historical Evaluation (N = 180)

> **Provenance Status: BLOCKED**  
> Because ATVI was delisted on October 13, 2023, it has 0 observations in the 2024–2026 forward test period. The historical 180 observations are from the historical 2022 test period and are presented separately:

| Model                             | Accuracy (%) | Balanced Acc (%) | Macro F1 | F1 (Down) | F1 (Up) | TN  | FP  | FN  | TP  |
| :-------------------------------- | :----------- | :--------------- | :------- | :-------- | :------ | :-- | :-- | :-- | :-- |
| **Majority Baseline**             | 41.11%       | 50.00%           | 0.2913   | 0.0000    | 0.5827  | 0   | 106 | 0   | 74  |
| **Next-Day LR (20-Day Flat)**     | 48.89%       | 51.10%           | 0.4883   | 0.4713    | 0.5054  | 41  | 65  | 27  | 47  |
| **Direction Persistence (Lag-1)** | 47.78%       | 46.07%           | 0.4607   | 0.5566    | 0.3649  | 59  | 47  | 47  | 27  |
| **Next-Day LR (t-1 Endpoint)**    | 46.67%       | 52.88%           | 0.4294   | 0.2836    | 0.5752  | 19  | 87  | 9   | 65  |
| **BiLSTM (Seed 42)**              | 41.11%       | 50.00%           | 0.2913   | 0.0000    | 0.5827  | 0   | 106 | 0   | 74  |
| **BiLSTM (Seed 123)**             | 41.11%       | 50.00%           | 0.2913   | 0.0000    | 0.5827  | 0   | 106 | 0   | 74  |
| **BiLSTM (Seed 7)**               | 41.11%       | 50.00%           | 0.2913   | 0.0000    | 0.5827  | 0   | 106 | 0   | 74  |

---

## 4. Empirical Reality & Absence of Automated Trading Edge

1. **Drift Alignment vs. True Skill:** The Majority baseline achieves the top raw accuracy (52.13%) solely because 3,280 out of 6,292 sessions (52.13%) in the 2024–2026 test horizon were upward days. This naive rule requires zero market intelligence, produces 0% Down-day recall, and fails during regime reversals.
2. **No Model Achieves a Profitable Edge:** The BiLSTM ensemble achieves 51.30% accuracy, which is lower than the trivial Majority rule. Directional accuracies near 50–51% reflect the low signal-to-noise ratio of daily market returns. Once round-trip commissions, exchange fees, bid-ask spread friction, and execution slippage are included, a 51% hit rate results in negative net trading returns.
3. **Rigorous Academic Takeaway:** No model in this benchmark should be claimed as a profitable alpha strategy or "market-beating" predictor. The value of this platform lies in demonstrating that rigorous point-in-time constraints, strict leakage elimination, and proper baselining expose the true boundaries of financial predictability.
