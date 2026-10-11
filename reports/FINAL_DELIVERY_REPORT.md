# Stock-AI Platform: Final Multimodal Research Delivery & Verification Report

**Project Title:** Stock-AI: Multimodal AI for Stock Trend Prediction
**GitHub Repository:** [https://github.com/Kavix28/Stock-AI.git](https://github.com/Kavix28/Stock-AI.git)
**Target Branch:** `main`
**Audit & Delivery Date:** 2026-10-11
**Test Suite Status:** **106 passed / 0 failed (100% pass rate in 7.14s)**
**Frontend Production Build:** **PASS (built in 3.76s, 0 errors)**
**Research Validation Status:** **FULLY INTEGRATED & AUDITED / MODERN RUNTIME ACTIVE / ATVI DELISTED HISTORICAL ARCHIVE**

---

## 1. Verified System Status Matrix

| Subsystem / Requirement                |    Status    |  Exit Code / Evidence  | Verification Details                                                                                                                                |
| :------------------------------------- | :----------: | :--------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Automated Regression Suite**         |   **PASS**   | Exit 0 (106/106 tests) | 100% pass rate across unit, integration, adversarial leakage, and live API suites in 7.14s.                                                         |
| **Backend REST API Integrity**         |   **PASS**   | Exit 0 (8/8 API tests) | FastAPI TestClient passes all endpoints (`/health`, `/api/v1/models`, `/api/v1/metrics/multimodal`, `/api/v1/predict`).                             |
| **Frontend Production Build**          |   **PASS**   |     Exit 0 (3.76s)     | `npm run build` succeeds with zero TypeScript errors; Vite assets generated cleanly.                                                                |
| **FinBERT Pretrained Real Weights**    |   **PASS**   |  Shape (768,) float32  | `ProsusAI/finbert` CPU forward inference verified ($L_2\text{ norm} = 16.6926$); zero-news vector mapped to zero norm.                              |
| **PyTorch Saved Checkpoints**          |   **PASS**   |     Fresh Process      | Loaded `models/gmu/gmu_fusion.pt` (610,357 bytes) and `models/market/bilstm_market.pt` (367,695 bytes) in fresh process with correct tensor shapes. |
| **GMU Gate Sum Invariants**            |   **PASS**   | $z_m + z_n \equiv 1.0$ | Verified mathematical convex combination gate summation across batch dimensions.                                                                    |
| **Adversarial Anti-Leakage Proofs**    |   **PASS**   |  Exit 0 (3 theorems)   | Proved zero future indicator leakage, strict 09:30 AM EST intraday news cutoff, and train-only scaler isolation.                                    |
| **Independent Metric Recomputation**   |   **PASS**   |       Recomputed       | Recomputed all metrics across 924 multimodal test sessions and 6,292 forward test sessions; verified majority-class collapse.                       |
| **12-Ticker Paper Replication (ATVI)** | **ISOLATED** |   Delisted Oct 2023    | Activision Blizzard delisted post-Microsoft acquisition; modern inference blocked; historical 2021–2022 archive preserved.                          |

---

## 2. Independent Metric Audit: Recomputed Benchmark Truth

Evaluated on **924 out-of-sample multimodal test sessions** (June 1, 2022 to September 20, 2022) across 12 equities:

| Model Architecture                 | Threshold ($\tau$) | Accuracy (%) | Balanced Acc (%) |  Macro F1  | Down F1 |   Up F1    |  ROC-AUC   | Brier Score | Predicted Distribution (Down / Up) |
| :--------------------------------- | :----------------: | :----------: | :--------------: | :--------: | :-----: | :--------: | :--------: | :---------: | :--------------------------------: |
| **Direction Persistence**          |        0.50        |  **52.16%**  |    **52.16%**    | **0.5216** | 0.5247  | **0.5185** |   0.5216   |   0.4784    |     464 (50.2%) / 460 (49.8%)      |
| **Logistic Regression (Flat)**     |        0.56        |    51.52%    |      51.40%      |   0.5063   | 0.5725  |   0.4400   | **0.5261** |   0.2531    |     582 (63.0%) / 342 (37.0%)      |
| **Logistic Regression (Endpoint)** |        0.54        |    49.57%    |      49.60%      |   0.4951   | 0.4787  |   0.5115   |   0.4733   |   0.2537    |     428 (46.3%) / 496 (53.7%)      |
| **GMU Fusion (Multimodal)**        |        0.52        |    50.00%    |      49.87%      |   0.4876   | 0.5674  |   0.4077   |   0.4830   |   0.2513    |     602 (65.2%) / 322 (34.8%)      |
| **Market-Only BiLSTM**             |        0.53        |    50.43%    |      50.00%      |   0.3353   | 0.6705  |   0.0000   |   0.4862   |   0.2507    |      924 (100.0%) / 0 (0.0%)       |
| **GMU Zero-News Ablation**         |        0.52        |    50.43%    |      50.00%      |   0.3353   | 0.6705  |   0.0000   |   0.4770   |   0.2640    |      924 (100.0%) / 0 (0.0%)       |
| **Training Majority (All-Down)**   |        0.50        |    50.43%    |      50.00%      |   0.3353   | 0.6705  |   0.0000   |   0.5000   |   0.4957    |      924 (100.0%) / 0 (0.0%)       |

### Mathematical Finding on Macro F1:

- **BiLSTM and Zero-News Ablation Collapse (Macro F1 = 0.3353):** With inputs below decision threshold $\tau$, both models output $\hat{y} = 0$ for 100% of samples (924 Down, 0 Up). Confusion matrix: $\text{TN}=466, \text{FP}=0, \text{FN}=458, \text{TP}=0$. Thus, $\text{F1(Up)} = 0.0000$ and $\text{F1(Down)} = 0.6705$, yielding $\text{Macro F1} = (0.6705 + 0.0) / 2 = \mathbf{0.3353}$.
- **GMU Multimodal Differentiation (Macro F1 = 0.4876):** FinBERT headline representations prevent mode collapse, producing diverse probabilities $[0.4602, 0.6374]$ and achieving $\text{F1(Down)} = 0.5674, \text{F1(Up)} = 0.4077$.
- **Absence of Trading Edge:** Direction Persistence achieves the highest directional accuracy (52.16%). A ~50–52% directional hit rate in daily equity markets does not constitute a profitable trading edge after standard transaction costs and slippage.

---

## 3. Fresh-Process Model & Checkpoint Execution Audit

Executed via `scripts/verify_real_models_execution.py`:

```
================================================================================
STEP 1: FinBERT Real-Weight Execution
================================================================================
Encoded headline: 'Federal Reserve signals interest rate decision amid cooling inflation.'
Embedding shape: (768,), dtype: float32, L2 norm: 16.6926

================================================================================
STEP 2: Load Saved PyTorch Checkpoints
================================================================================
GMU checkpoint: models/gmu/gmu_fusion.pt (610,357 bytes)
BiLSTM checkpoint: models/market/bilstm_market.pt (367,695 bytes)
GMU Fusion state_dict loaded successfully.
Market BiLSTM state_dict loaded successfully.

================================================================================
STEP 3: Genuine Forward Tensor Inferences
================================================================================
Market BiLSTM output shape: torch.Size([2, 1]), values: [0.5229, 0.5224]
GMU Fusion output shape: torch.Size([2, 1]), values: [0.5243, 0.3770]
GMU Mean Market Gate weights (z_m): [0.4596, 0.4476]
GMU Mean News Gate weights (z_n):   [0.5404, 0.5524]
Gate weights sum (z_m + z_n):  [1.0, 1.0]

================================================================================
STEP 4: InferenceEngine Live Predictions & Fallbacks
================================================================================
AAPL GMU Prediction: Modality: Multimodal (Market BiLSTM + FinBERT News), Market Gate: 0.4694, News Gate: 0.5306
NVDA BiLSTM Prediction: Modality: Market Time-Series Only (BiLSTM), Market Gate: 1.0, News Gate: 0.0
ATVI Delisted Asset Prediction: Ticker Status: Delisted Oct 2023; Historical Research Archive Only
```

---

## 4. Adversarial Anti-Leakage Proofs

Validated in `tests/unit/test_adversarial_leakage.py`:

1. **Indicator Lookahead Invariance:** Injected +500% future price spikes at $t > t_k$; verified $L_\infty = 0.0$ on past technical indicators.
2. **Intraday News Cutoff:** Verified that articles published at or after 09:30 AM EST cannot leak into day $t$ predictions.
3. **Scaler Range Isolation:** Verified that training scalers are strictly fitted on the train partition and never contaminate test distributions.

---

## 5. Automated Regression Test Suite & Frontend Build

### Python Test Suite (`pytest`):

```bash
python -m pytest tests/ -q
# Exit code: 0
# Result: 106 passed in 7.14s (100% pass rate)
```

### Frontend Build (`vite`):

```bash
npm run build
# Exit code: 0
# Result: built in 3.76s, 0 errors
```

---

## 6. Known Limitations & Research Boundaries

1. **ATVI Modern Ingestion:** ATVI was delisted on October 13, 2023 following acquisition by Microsoft Corporation. Modern forward testing for ATVI is blocked; historical 2021–2022 research results are permanently archived and isolated.
2. **Modern News Feed Requirement:** Free community news feeds lack audited intraday UTC timestamps. Modern forward inference defaults to market-only fallback mode ($w_m = 1.0, w_n = 0.0$) unless authenticated historical news providers (e.g., Polygon/Tiingo paid tiers) are configured.
3. **Market Efficiency & Financial Disclosure:** Directional predictions are provided for academic research and evaluation purposes only. A ~50–52% directional hit rate does not constitute financial advice or a commercial trading edge.

---

## 7. Git Remote Synchronization

- **Commit Hash:** `6ec05db9352f2148d03de8a9`
- **Remote Status:** Synchronized with `origin/main` (`https://github.com/Kavix28/Stock-AI.git`)
- **Remote Reference:** `cbbb04b..6ec05db main -> main`
- **Working Tree:** Clean, 0 uncommitted changes, 0 untracked files.
