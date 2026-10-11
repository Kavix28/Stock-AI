# Stock-AI: Final Multimodal Research Delivery Report

**Date:** October 11, 2026
**Repository:** `C:\PROJECTS\multimodal-stock-ai`
**Target GitHub Remote:** `https://github.com/Kavix28/Stock-AI.git` (`origin`, branch: `main`)
**Secondary Historical Workspace:** `C:\PROJECTS\Research paer\finbert-lstm`
**Test Suite Status:** **106 passed / 0 failed (100% pass rate in 7.14s)**
**Frontend Production Build:** **PASS (built in 3.76s, 0 errors)**

---

## 1. Executive Summary & Research Classification

Stock-AI is an end-to-end, reproducible multimodal deep learning platform for predicting daily stock trend directions ($y_t \in \{0, 1\}$) by combining sequential technical market features with FinBERT financial news embeddings via a Gated Multimodal Unit (GMU).

### Scientific Claims & Architectural Taxonomies:

- `[FROM PAPER]` Chen et al. (2024), _Neural Computing and Applications_, DOI: 10.1007/s00521-024-10303-1:
  - 12 equity universe (`AAPL`, `AMT`, `ATVI`, `BIO`, `JNJ`, `NDAQ`, `NFLX`, `NVDA`, `PLD`, `SBUX`, `SCHW`, `TSLA`).
  - 20-trading-day rolling lookback window ($t-20 \dots t-1$).
  - 16 core technical market features per trading day.
  - FinBERT (768-dimensional CLS token) for financial news representations.
  - Gated Multimodal Unit (GMU) for convex combination fusion ($h_f = z \cdot h_m + (1 - z) \cdot h_n$).
- `[IMPLEMENTED]`
  - PyTorch `MarketBiLSTM` (two-layer bidirectional LSTM, hidden 64/32).
  - PyTorch `GatedMultimodalUnit` (convex feature gating, LayerNorm, Sigmoid gating).
  - Real-weight FinBERT encoder pipeline (`ProsusAI/finbert`).
  - Strict point-in-time chronological train/val/test split and causal alignment.
  - Fast-inference REST API via FastAPI and React/Vite dashboard.
- `[OUR DECISION]`
  - Split 2021–2022 data into strict chronological partitions:
    - Train: `2021-01-05` to `2021-12-31` (248 trading days $\times$ 12 equities = 2,976 instances).
    - Validation: `2022-01-03` to `2022-05-31` (103 trading days $\times$ 12 equities = 1,236 instances).
    - Out-of-Sample Test: `2022-06-01` to `2022-09-20` (77 trading days $\times$ 12 equities = 924 instances).
  - MinMaxScaler fitted strictly on training partition ($[T_0, T_{train}]$).
  - Decision threshold $\tau$ calibrated strictly on validation partition. The 924 test samples were never used for training, hyperparameter tuning, or threshold selection.
  - Tag historical 2021–2022 news archive as `TIMING-LIMITED: Missing Intraday Timestamps` due to date-only granularity.
  - Enforce market-only fallback ($w_m = 1.0, w_n = 0.0$) when intraday news is unavailable.
- `[PROPOSED EXTENSION]`
  - Integration of Polygon / Tiingo paid APIs for live intraday UTC news streaming.
  - Walk-forward retraining on modern (2024–2026) active equity data.
- `[BLOCKED]`
  - Modern live inference for `ATVI` (Activision Blizzard was acquired by Microsoft and delisted on October 13, 2023; modern forward data acquisition returns 404).

---

## 2. Environment & Dependency Audit

The primary runtime is governed by the project virtual environment `C:\PROJECTS\multimodal-stock-ai\.venv` on Python 3.11.1:

| Package            | Version        | Verification Method      | Status         |
| :----------------- | :------------- | :----------------------- | :------------- |
| **Python**         | 3.11.1 (AMD64) | `sys.version`            | ACTIVE         |
| **PyTorch**        | 2.14.1+cpu     | Forward tensor pass      | ACTIVE         |
| **Transformers**   | 5.19.0         | FinBERT pipeline load    | ACTIVE         |
| **FastAPI**        | 0.143.0        | Live endpoint dispatch   | ACTIVE         |
| **Pydantic**       | 2.14.0         | Schema validation        | ACTIVE         |
| **scikit-learn**   | 1.9.1          | Metric recomputation     | ACTIVE         |
| **pandas**         | 3.0.6          | DataFrames & manifests   | ACTIVE         |
| **numpy**          | 2.4.6          | Numerical arrays         | ACTIVE         |
| **pytest**         | 9.1.1          | 106-test automated suite | PASS (106/106) |
| **Node.js / Vite** | 20.x / 5.x     | `npm run build`          | PASS (3.76s)   |

All production dependencies are formally declared in `pyproject.toml`. The primary platform does not depend on the secondary workspace (`C:\PROJECTS\Research paer\finbert-lstm`).

---

## 3. Cryptographic Artifact Digests (SHA-256)

| Artifact Path                                       | File Size     | SHA-256 Digest                                                     |
| :-------------------------------------------------- | :------------ | :----------------------------------------------------------------- |
| `models/gmu/gmu_fusion.pt`                          | 610,357 bytes | `b0dc524084443d94e7e904707327772018c049d177b3932a85335c5c5985cf19` |
| `models/market/bilstm_market.pt`                    | 367,695 bytes | `527f39e73ef9001a18d2ced69102bfe6dd24af9e77dbffba819194daf95980d0` |
| `results/multimodal/multimodal_predictions.csv`     | 97,003 bytes  | `5a005a88baedb7bb7d4b7cc0424abdaea27a5cf4b0c5fd45d1d13fc32f89aee1` |
| `results/multimodal/multimodal_metrics_summary.csv` | 1,903 bytes   | `cd90a3ae30ce9be0f8028a441a7ad064c96541c1f7a12d3ca0e714868369fed8` |
| `data/processed/news/news_manifest.json`            | 8,481 bytes   | `b7f44fb668b2cbc4136c7ef528d83e0c9680af6619aaecea126d995e6393ab01` |
| `data/processed/news/news_manifest.csv`             | 3,283 bytes   | `9a2e4148f75b63f00abadec1378b7180331e4eaae5c3df65155f049bd187d985` |

---

## 4. Recomputed Benchmark Metrics Audit

Recomputed directly from `results/multimodal/multimodal_predictions.csv` across the **924 out-of-sample forward test sessions** (`2022-06-01` to `2022-09-20`, 77 trading days $\times$ 12 equities):

### Ground Truth Class Distribution:

- Total: 924
- Class 0 (Down): 466 ($50.43\%$)
- Class 1 (Up): 458 ($49.57\%$)

| Model Architecture                 | Threshold ($\tau$) | Accuracy (%) | Balanced Acc (%) |  Macro F1  | Weighted F1 | Down F1 |   Up F1    |  ROC-AUC   | Brier  | Distribution (Down / Up)  |
| :--------------------------------- | :----------------: | :----------: | :--------------: | :--------: | :---------: | :-----: | :--------: | :--------: | :----: | :-----------------------: |
| **Direction Persistence**          |        0.50        |  **52.16%**  |    **52.16%**    | **0.5216** | **0.5217**  | 0.5247  | **0.5185** |   0.5216   | 0.4784 | 464 (50.2%) / 460 (49.8%) |
| **Logistic Regression (Flat)**     |        0.56        |    51.52%    |      51.40%      |   0.5063   |   0.5068    | 0.5725  |   0.4400   | **0.5261** | 0.2531 | 582 (63.0%) / 342 (37.0%) |
| **Logistic Regression (Endpoint)** |        0.54        |    49.57%    |      49.60%      |   0.4951   |   0.4950    | 0.4787  |   0.5115   |   0.4733   | 0.2537 | 428 (46.3%) / 496 (53.7%) |
| **GMU Fusion (Multimodal)**        |        0.52        |    50.00%    |      49.87%      |   0.4876   |   0.4882    | 0.5674  |   0.4077   |   0.4830   | 0.2513 | 602 (65.2%) / 322 (34.8%) |
| **Market-Only BiLSTM**             |        0.53        |    50.43%    |      50.00%      |   0.3353   |   0.3382    | 0.6705  |   0.0000   |   0.4862   | 0.2507 |  924 (100.0%) / 0 (0.0%)  |
| **GMU Zero-News Ablation**         |        0.52        |    50.43%    |      50.00%      |   0.3353   |   0.3382    | 0.6705  |   0.0000   |   0.4770   | 0.2640 |  924 (100.0%) / 0 (0.0%)  |
| **Majority Baseline (All-Down)**   |        0.50        |    50.43%    |      50.00%      |   0.3353   |   0.3382    | 0.6705  |   0.0000   |   0.5000   | 0.4957 |  924 (100.0%) / 0 (0.0%)  |

### Clarification on Discrepancy from Early Draft:

An earlier draft cited 2,160 instances (representing the combined 180 trading days of 2022). In the verified experimental design, the first 103 trading days of 2022 (`2022-01-03` to `2022-05-31`, 1,236 instances) were reserved for validation early stopping and threshold calibration $\tau$. The final 77 trading days (`2022-06-01` to `2022-09-20`, 924 instances) represent the strict out-of-sample forward test set.

### Scientific Finding on Mode Collapse & Multimodal Performance:

1. **Majority Mode Collapse:** Both the Market-Only BiLSTM ($\tau = 0.53$) and GMU Zero-News Ablation ($\tau = 0.52$) generated probabilities strictly below threshold ($p_{bilstm} \approx 0.5228 < 0.53$, $p_{ablation} \approx 0.3773 < 0.52$). Consequently, both models predicted $\hat{y}_t = 0$ (Down) for 100% of samples ($\text{TN}=466, \text{FP}=0, \text{FN}=458, \text{TP}=0$), yielding $\text{Up F1} = 0.0$, $\text{Down F1} = 0.6705$, and $\text{Macro F1} = \mathbf{0.3353}$.
2. **Multimodal Differentiation:** The GMU with FinBERT representations successfully broke mode collapse, spanning probabilities $[0.4602, 0.6374]$ and achieving $\text{Macro F1} = \mathbf{0.4876}$.
3. **Multimodal vs. Persistence:** **GMU Fusion (50.00% accuracy, 0.4876 Macro F1) does NOT outperform Direction Persistence (52.16% accuracy, 0.5216 Macro F1) or Flat Logistic Regression (51.52% accuracy, 0.5063 Macro F1).**
4. **Trading Edge Disclosure:** Directional accuracy of ~50% to 52% does **NOT** constitute a profitable trading edge after factoring in transaction costs, spread friction, and slippage.

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

## 6. Git Synchronization State

- **Target Repository:** `https://github.com/Kavix28/Stock-AI.git`
- **Primary Remote:** `origin` (Branch: `main`)
- **Remote Reference:** `6ec05db..e041087 main -> main`
- **Working Tree:** Clean, 0 uncommitted changes, 0 untracked files.
