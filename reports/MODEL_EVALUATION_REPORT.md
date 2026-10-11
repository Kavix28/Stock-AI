# Stock-AI: Independent Model Evaluation and Benchmark Audit Report

**Date:** October 11, 2026
**Auditor:** Independent Research QA & Statistical Audit
**Evaluation Scope:** Out-of-sample forward test horizon (`2022-06-01` to `2022-09-20`)
**Dataset:** 924 test instances across 12 tickers (77 trading days $\times$ 12 equities)
**Predictions Source:** `results/multimodal/multimodal_predictions.csv`

---

## 1. Academic Taxonomy & Protocol Classification

To maintain the highest standard of scientific integrity, all components, design decisions, and findings are explicitly categorized:

- `[FROM PAPER]` Chen et al. (2024), _Neural Computing and Applications_, DOI: 10.1007/s00521-024-10303-1:
  - 12 equity universe (`AAPL`, `AMT`, `ATVI`, `BIO`, `JNJ`, `NDAQ`, `NFLX`, `NVDA`, `PLD`, `SBUX`, `SCHW`, `TSLA`).
  - 20-trading-day sequential market lookback window ($t-20 \dots t-1$).
  - 16 core technical market features per trading day (`high`, `close`, `open`, `volume`, `low`, `adjclose`, `rsi`, `12-Day EMA`, `26-Day EMA`, `MACD`, `SAR`, `Upper Band`, `Middle Band`, `Lower Band`, `Slow k`, `Slow d`).
  - FinBERT (768-dimensional CLS token) for financial news representations.
  - Gated Multimodal Unit (GMU) convex combination fusion ($h_f = z \cdot h_m + (1 - z) \cdot h_n$).
- `[IMPLEMENTED]`
  - PyTorch `MarketBiLSTM` (`src/stock_ai/models/market/bilstm.py:L13-105`): Two-layer bidirectional LSTM, hidden 64/32.
  - PyTorch `GatedMultimodalUnit` (`src/stock_ai/models/fusion/gmu.py:L23-145`): Feature projection, LayerNorm, sigmoid gating, convex combination.
  - Real-weight FinBERT encoder pipeline (`src/stock_ai/models/news/finbert.py:L20-110`).
  - FastAPI serving layer (`backend/main.py`) and React dashboard (`frontend/`).
- `[OUR DECISION]`
  - Chronological 3-way split (`scripts/train_multimodal.py:L160-176`):
    - Train: `2021-01-05` to `2021-12-31` (2,976 samples).
    - Validation: `2022-01-03` to `2022-05-31` (1,236 samples).
    - Test: `2022-06-01` to `2022-09-20` (924 samples).
  - Scaler fitting (`scripts/train_multimodal.py:L179-181`): `MinMaxScaler` fitted strictly on training sequence matrices $[T_0, T_{train}]$.
  - Threshold calibration $\tau$ (`scripts/train_multimodal.py:L226-236`): Optimized strictly on the validation partition to maximize Macro F1. The 924 test samples were never used for training, hyperparameter tuning, or threshold selection.
  - Direction Persistence Baseline (`scripts/train_multimodal.py:L132`): Constructed strictly from $\text{Diff}_{t-1}$ (prior session price change), ensuring zero future lookahead.
  - Timestamp quality designation: Chen et al. historical news archive labeled `TIMING-LIMITED: Missing Intraday Timestamps` due to date-only resolution.
  - Modern market fallback: When intraday timestamps are absent, inference engine defaults to market-only prediction ($w_m = 1.0, w_n = 0.0$).
- `[PROPOSED EXTENSION]`
  - Real-time intraday news stream integration with verified UTC timestamps (e.g., Polygon / Tiingo paid tiers).
  - Walk-forward retraining on modern active equities (2024–2026).
- `[BLOCKED]`
  - `ATVI` modern data acquisition is permanently blocked (acquired by Microsoft, delisted October 13, 2023). Historical 2021–2022 data is isolated in research archive.

---

## 2. Cryptographic Artifact Hashes (SHA-256)

All evaluations are anchored to immutable file digests:

| Artifact Path                                       | Size (Bytes) | SHA-256 Checksum                                                   |
| :-------------------------------------------------- | :----------- | :----------------------------------------------------------------- |
| `models/gmu/gmu_fusion.pt`                          | 610,357      | `b0dc524084443d94e7e904707327772018c049d177b3932a85335c5c5985cf19` |
| `models/market/bilstm_market.pt`                    | 367,695      | `527f39e73ef9001a18d2ced69102bfe6dd24af9e77dbffba819194daf95980d0` |
| `results/multimodal/multimodal_predictions.csv`     | 97,003       | `5a005a88baedb7bb7d4b7cc0424abdaea27a5cf4b0c5fd45d1d13fc32f89aee1` |
| `results/multimodal/multimodal_metrics_summary.csv` | 1,903        | `cd90a3ae30ce9be0f8028a441a7ad064c96541c1f7a12d3ca0e714868369fed8` |
| `data/processed/news/news_manifest.json`            | 8,481        | `b7f44fb668b2cbc4136c7ef528d83e0c9680af6619aaecea126d995e6393ab01` |
| `data/processed/news/news_manifest.csv`             | 3,283        | `9a2e4148f75b63f00abadec1378b7180331e4eaae5c3df65155f049bd187d985` |

---

## 3. Executive Summary & Ground Truth Distribution

This independent audit rigorously evaluates all benchmark and proposed multimodal models against out-of-sample data.

### Ground Truth Class Distribution:

- **Total Test Samples:** 924
- **Class 1 (Up Sessions):** 458 ($49.57\%$)
- **Class 0 (Down Sessions):** 466 ($50.43\%$)
- **Data Integrity:** Zero NaNs, zero infinities, all predicted probabilities strictly bounded in $[0.0, 1.0]$.
- **Ticker Coverage:** 12 equities: `AAPL`, `AMT`, `ATVI`, `BIO`, `JNJ`, `NDAQ`, `NFLX`, `NVDA`, `PLD`, `SBUX`, `SCHW`, `TSLA`.
- **Sliding Window Construction:** Market windows span $t - 20$ to $t - 1$ (`scripts/train_multimodal.py:L127`), strictly excluding session $t$'s market data.

---

## 4. Recomputed Benchmark Metrics Summary

All metrics were independently recomputed directly from `results/multimodal/multimodal_predictions.csv`:

| Model Architecture                 | Threshold ($\tau$) | Accuracy (%) | Balanced Acc (%) |  Macro F1  | Weighted F1 | Down F1 |   Up F1    |  ROC-AUC   | Brier Score | Predicted Dist (Down / Up) |
| :--------------------------------- | :----------------: | :----------: | :--------------: | :--------: | :---------: | :-----: | :--------: | :--------: | :---------: | :------------------------: |
| **Direction Persistence**          |        0.50        |  **52.16%**  |    **52.16%**    | **0.5216** | **0.5217**  | 0.5247  | **0.5185** |   0.5216   |   0.4784    | 464 (50.2%) / 460 (49.8%)  |
| **Logistic Regression (Flat)**     |        0.56        |    51.52%    |      51.40%      |   0.5063   |   0.5068    | 0.5725  |   0.4400   | **0.5261** |   0.2531    | 582 (63.0%) / 342 (37.0%)  |
| **Logistic Regression (Endpoint)** |        0.54        |    49.57%    |      49.60%      |   0.4951   |   0.4950    | 0.4787  |   0.5115   |   0.4733   |   0.2537    | 428 (46.3%) / 496 (53.7%)  |
| **GMU Fusion (Multimodal)**        |        0.52        |    50.00%    |      49.87%      |   0.4876   |   0.4882    | 0.5674  |   0.4077   |   0.4830   |   0.2513    | 602 (65.2%) / 322 (34.8%)  |
| **Market-Only BiLSTM**             |        0.53        |    50.43%    |      50.00%      |   0.3353   |   0.3382    | 0.6705  |   0.0000   |   0.4862   |   0.2507    |  924 (100.0%) / 0 (0.0%)   |
| **GMU Zero-News Ablation**         |        0.52        |    50.43%    |      50.00%      |   0.3353   |   0.3382    | 0.6705  |   0.0000   |   0.4770   |   0.2640    |  924 (100.0%) / 0 (0.0%)   |
| **Training Majority (All-Down)**   |        0.50        |    50.43%    |      50.00%      |   0.3353   |   0.3382    | 0.6705  |   0.0000   |   0.5000   |   0.4957    |  924 (100.0%) / 0 (0.0%)   |

---

## 5. Investigation: Equivalence of Market-Only BiLSTM and Zero-News Ablation (Macro F1 = 0.3353)

### Mathematical Root Cause:

1. **BiLSTM Predicted Probability Distribution:**
   - Range: $[0.5225, 0.5230]$ (mean: $0.5228$).
   - The decision threshold selected for the BiLSTM on validation data was $\tau = 0.53$.
   - Since $p_t < 0.53$ for all $t \in [1, 924]$, the model predicted $\hat{y}_t = 0$ (Down) for **100% of samples**.
2. **GMU Zero-News Ablation Probability Distribution:**
   - Range: $[0.3769, 0.3776]$ (mean: $0.3774$).
   - The decision threshold selected for GMU on validation data was $\tau = 0.52$.
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

**Conclusion:** Both architectures experienced majority-class collapse into an all-Down predictor, yielding mathematically identical Macro F1 and accuracy ($50.43\%$).

---

## 6. Investigation: GMU Multimodal Fusion Behavior (Macro F1 = 0.4876)

When news embeddings are injected via FinBERT, the Gated Multimodal Unit (GMU) exhibits genuine multimodal differentiation:

- **Output Probability Distribution:** Bounded across $[0.4602, 0.6374]$ (mean: $0.5185$).
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

This is an honest, scientifically grounded finding: daily news sentiment headlines alone do not provide a statistically significant directional alpha over technical momentum during a volatile macroeconomic tightening cycle.

---

## 7. Direction Persistence Baseline Verification

- **Formula:** $\hat{y}_t = \text{Diff}_{t-1} = \mathbb{I}(\text{Close}_{t-1} > \text{Close}_{t-2})$ (`scripts/train_multimodal.py:L132`).
- **Lookahead Check:** On trading day $t$, $\text{Close}_{t-1}$ is yesterday's market close, known with $100\%$ certainty before market open at $09:30\text{ EST}$.
- **Future Information:** Exactly zero future price or label information enters the feature.
- **Performance:** Achieves top raw accuracy ($52.16\%$) and top Macro F1 ($0.5216$), demonstrating momentum persistence across the 2022 test window.

---

## 8. Verification of Ground-Truth Alignment & No Leakage

1. **Alignment:** All 924 prediction rows correspond 1:1 with ground-truth price delta signs ($\text{Close}_t > \text{Close}_{t-1} \implies y_t = 1$).
2. **Deduplication:** Confirmed zero duplicate `(ticker, target_date)` pairs.
3. **No Calibration Leakage:** Thresholds $\tau$ were selected strictly on the validation partition ($2022\text{-}01\text{-}03$ to $2022\text{-}05\text{-}31$) and evaluated out-of-sample on the test partition ($2022\text{-}06\text{-}01$ to $2022\text{-}09\text{-}20$). The 924 test samples were never accessed during tuning.
4. **News Timestamp Certification:** The Chen et al. historical news archive lacks intraday UTC timestamps. Articles are matched on calendar date ($T_n = T_m$), which is certified as `TIMING-LIMITED: Missing Intraday Timestamps`. Real-time deployment mandates fallback to market-only inference ($w_m = 1.0, w_n = 0.0$) until intraday feeds are integrated.
