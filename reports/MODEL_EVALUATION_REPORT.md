# Stock-AI: Independent Model Evaluation and Benchmark Audit Report

**Date:** October 11, 2026  
**Auditor:** Independent Research QA & Statistical Audit  
**Evaluation Scope:** Out-of-sample forward test horizon (`2022-06-01` to `2022-09-20`)  
**Dataset:** 924 test instances across 12 tickers (77 trading days $\times$ 12 equities)  
**Predictions Source:** `results/multimodal/multimodal_predictions.csv`

---

## 1. Executive Summary & Ground Truth Distribution

This independent audit rigorously evaluates all benchmark and proposed multimodal models against out-of-sample data.

### Ground Truth Class Distribution:

- **Total Test Samples:** 924
- **Class 1 (Up Sessions):** 458 ($49.57\%$)
- **Class 0 (Down Sessions):** 466 ($50.43\%$)
- **Data Integrity:** Zero NaNs, zero infinities, all predicted probabilities strictly bounded in $[0.0, 1.0]$.
- **Ticker Coverage:** 12 equities: `AAPL`, `AMT`, `ATVI`, `BIO`, `JNJ`, `NDAQ`, `NFLX`, `NVDA`, `PLD`, `SBUX`, `SCHW`, `TSLA`.

---

## 2. Recomputed Benchmark Metrics Summary

All metrics were computed with strict zero-division guards and fixed decision thresholds ($\tau$):

| Model Architecture                 | Threshold ($\tau$) | Accuracy (%) | Balanced Acc (%) |  Macro F1  | Weighted F1 | Down F1 | Up F1  |  ROC-AUC   | Brier Score | Predicted Dist (Down / Up) |
| :--------------------------------- | :----------------: | :----------: | :--------------: | :--------: | :---------: | :-----: | :----: | :--------: | :---------: | :------------------------: |
| **Direction Persistence**          |        0.50        |  **52.16%**  |    **52.16%**    | **0.5216** | **0.5217**  | 0.5247  | 0.5185 |   0.5216   |   0.4784    | 464 (50.2%) / 460 (49.8%)  |
| **Logistic Regression (Flat)**     |        0.56        |    51.52%    |      51.40%      |   0.5063   |   0.5068    | 0.5725  | 0.4400 | **0.5261** |   0.2531    | 582 (63.0%) / 342 (37.0%)  |
| **Logistic Regression (Endpoint)** |        0.54        |    49.57%    |      49.60%      |   0.4951   |   0.4950    | 0.4787  | 0.5115 |   0.4733   |   0.2537    | 428 (46.3%) / 496 (53.7%)  |
| **GMU Fusion (Multimodal)**        |        0.52        |    50.00%    |      49.87%      |   0.4876   |   0.4882    | 0.5674  | 0.4077 |   0.4830   |   0.2513    | 602 (65.2%) / 322 (34.8%)  |
| **Market-Only BiLSTM**             |        0.53        |    50.43%    |      50.00%      |   0.3353   |   0.3382    | 0.6705  | 0.0000 |   0.4862   |   0.2507    |  924 (100.0%) / 0 (0.0%)   |
| **GMU Zero-News Ablation**         |        0.52        |    50.43%    |      50.00%      |   0.3353   |   0.3382    | 0.6705  | 0.0000 |   0.4770   |   0.2640    |  924 (100.0%) / 0 (0.0%)   |
| **Training Majority (All-Down)**   |        0.50        |    50.43%    |      50.00%      |   0.3353   |   0.3382    | 0.6705  | 0.0000 |   0.5000   |   0.4957    |  924 (100.0%) / 0 (0.0%)   |

---

## 3. Investigation: Equivalence of Market-Only BiLSTM and Zero-News Ablation (Macro F1 = 0.3353)

A core question investigated in this audit is why the Market-Only BiLSTM and the GMU Zero-News Ablation yielded the identical Macro F1 value of **0.3353**.

### Mathematical Findings:

1. **BiLSTM Predicted Probability Distribution:**
   - Range: $[0.5225, 0.5230]$ (mean: $0.5228$).
   - The decision threshold selected for the BiLSTM was $\tau = 0.53$.
   - Since $p_t < 0.53$ for all $t \in [1, 924]$, the model predicted $\hat{y}_t = 0$ (Down) for **100% of samples**.
2. **GMU Zero-News Ablation Probability Distribution:**
   - Range: $[0.3769, 0.3776]$ (mean: $0.3773$).
   - The decision threshold selected for the GMU was $\tau = 0.52$.
   - Since $p_t < 0.52$ for all $t \in [1, 924]$, the ablation model predicted $\hat{y}_t = 0$ (Down) for **100% of samples**.
3. **Confusion Matrix Collapse:**
   $$\text{TN} = 466, \quad \text{FP} = 0, \quad \text{FN} = 458, \quad \text{TP} = 0$$
   - Precision(Up) = $0 / 0 \xrightarrow{\text{zero-div}} 0.0000$
   - Recall(Up) = $0 / 458 = 0.0000$
   - F1(Up) = $0.0000$
   - Precision(Down) = $466 / 924 = 0.5043$
   - Recall(Down) = $466 / 466 = 1.0000$
   - F1(Down) = $\frac{2 \times 0.5043 \times 1.0}{0.5043 + 1.0} = 0.6705$
   - **Macro F1:**
     $$\text{Macro F1} = \frac{\text{F1(Down)} + \text{F1(Up)}}{2} = \frac{0.6705 + 0.0000}{2} = \mathbf{0.3353}$$

**Conclusion:** Both models experienced majority-class collapse into an all-Down predictor, yielding mathematically identical Macro F1 and accuracy ($50.43\%$).

---

## 4. Investigation: GMU Multimodal Fusion Behavior (Macro F1 = 0.4876)

When news embeddings are injected via FinBERT, the Gated Multimodal Unit (GMU) changes behavior dramatically:

- **Output Probability Distribution:** Bounded across $[0.4602, 0.6374]$ (mean: $0.5085$).
- **Predicted Classes ($\tau = 0.52$):**
  - Predicted Down: 602 ($65.2\%$)
  - Predicted Up: 322 ($34.8\%$)
- **Confusion Matrix:**
  $$\text{TN} = 303, \quad \text{FP} = 163, \quad \text{FN} = 299, \quad \text{TP} = 159$$
- **Class Metrics:**
  - Down Class: Precision = 0.5033, Recall = 0.6502, F1 = **0.5674**
  - Up Class: Precision = 0.4938, Recall = 0.3472, F1 = **0.4077**
  - **Macro F1:** $(0.5674 + 0.4077) / 2 = \mathbf{0.4876}$
- **Accuracy:** $50.00\%$ ($462 / 924$)

### Honest Assessment of Multimodal Efficacy:

While FinBERT embeddings successfully break the degenerate all-Down mode collapse (raising Macro F1 from 0.3353 to 0.4876), **the multimodal model does not outperform simple Direction Persistence (52.16% accuracy, 0.5216 Macro F1)** or Logistic Regression Flat (51.52% accuracy, 0.5063 Macro F1).

This is an honest, scientifically grounded conclusion: daily news sentiment headlines alone do not provide a statistically significant directional alpha over technical momentum during a volatile macroeconomic tightening cycle.

---

## 5. Verification of Ground-Truth Alignment & No Leakage

1. **Alignment:** All 924 prediction rows correspond 1:1 with ground-truth price delta signs ($\text{Close}_t > \text{Close}_{t-1} \implies y_t = 1$).
2. **Deduplication:** Confirmed zero duplicate `(ticker, target_date)` pairs.
3. **No Calibration Leakage:** Thresholds $\tau$ were selected on the validation partition ($2021\text{-}10\text{-}01$ to $2021\text{-}12\text{-}31$) and evaluated out-of-sample on the test partition.
