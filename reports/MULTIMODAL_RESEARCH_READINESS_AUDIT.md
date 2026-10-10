# Stock-AI: Multimodal Research Readiness Audit Report

**Audit Date:** 2026-10-10  
**Repository:** `C:\PROJECTS\multimodal-stock-ai`  
**Git Remote:** `https://github.com/Kavix28/Stock-AI.git` (`main`)  
**Audit Purpose:** Comprehensive inspection of multimodal architecture, news feature ingestion, model checkpoints, inference pipelines, and empirical reproducibility.

---

## 1. Executive Verdict

> ### **Concise Verdict:**
>
> **The current Stock-AI project is primarily a market-only quantitative prediction platform.**
>
> The multimodal research model described in the reference paper (_Chen et al., 2024_) is **incomplete, untrained on the forward evaluation horizon, missing a production news pipeline, and completely absent from live inference**:
>
> 1. In the production repository (`C:\PROJECTS\multimodal-stock-ai`), only a skeleton Keras architecture function for the Gated Multimodal Unit exists (`src/stock_ai/models/fusion/gmu.py`).
> 2. There is **no news encoder module**, no FinBERT feature extractor, and **zero saved weights or precomputed embeddings** (`.npy` / `.npz`) in the package.
> 3. The processed news files terminate on **September 20, 2022**. There are **0 news headlines** for the out-of-sample forward test horizon (July 2024 – October 2026).
> 4. The news dataset lacks intraday timestamps (`YYYY-MM-DD` only), creating potential post-close lookahead risk (`TIMING-LIMITED`).
> 5. The production `InferenceEngine` (`src/stock_ai/inference/engine.py`) and FastAPI REST backend serve predictions generated **strictly from market-only time-series BiLSTM models**.
> 6. On the historical 2022 single-stock evaluation (`AMT`, 160 sequences), GMU fusion underperformed simple baselines (**47.38% ± 1.63%** vs **58.12%** for persistence), with gate activations ($z \approx 0.494$) demonstrating that blending ungrounded headline noise degraded market signals.

---

## 2. Component-by-Component Architectural Trace

| Model Component               | Exact Repository File Path                        | Implementation Details                                                                                                                                                                                          | Training & Weight Status                                                                                | Used by Inference Engine?         | Evidence & Audit Findings                                                               |
| :---------------------------- | :------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------ | :-------------------------------- | :-------------------------------------------------------------------------------------- |
| **Time-Series Encoder**       | `src/stock_ai/models/market/bilstm.py`            | 2-Layer BiLSTM (64 & 32 units) + Dense(32) + Sigmoid head on $20 \times 16$ input windows.                                                                                                                      | **Trained & Saved** as predictions in `results/latest/predictions/` (3 seeds: 42, 123, 7).              | **YES**                           | Evaluated on 6,292 forward test targets. Primary driver of production API responses.    |
| **Market Baselines**          | `src/stock_ai/models/baselines.py`                | Majority Baseline, Direction Persistence (Lag-1), Logistic Regression (Endpoint & Flat-20).                                                                                                                     | **Trained & Benchmarked** across all 11 active equities + ATVI.                                         | **NO** (Available for validation) | Verified in `tests/test_baselines.py` and benchmark summaries.                          |
| **News Encoder**              | `src/stock_ai/models/news/` _(Missing)_           | **Does not exist** in the production package. A prototype function `extract_finbert_embeddings()` exists only in the secondary research workspace (`Research paer/finbert-lstm/scripts/run_gmu_experiment.py`). | **NOT TRAINED / NOT SAVED**                                                                             | **NO**                            | No PyTorch or HuggingFace FinBERT pipeline in `src/stock_ai/`.                          |
| **Financial-News Embeddings** | `data/processed/embeddings/` _(Missing)_          | Zero precomputed `.npy`, `.npz`, or `.pt` embedding matrices exist in `multimodal-stock-ai`.                                                                                                                    | **0 files found**                                                                                       | **NO**                            | `glob.glob('**/*.npy')` returns 0 files.                                                |
| **Multimodal Fusion (GMU)**   | `src/stock_ai/models/fusion/gmu.py`               | `build_gmu_fusion_model()` defines a dimensionally valid Gated Multimodal Unit: $z = \sigma(W_z [h_m, h_n])$, $h_{\text{fused}} = z \odot h_m + (1-z) \odot h_n$.                                               | **Untrained in production package**; 0 saved model checkpoints in `models/` (directory does not exist). | **NO**                            | Architecture exists in code, but is never invoked by `scripts/evaluate.py`.             |
| **Prediction Head**           | `src/stock_ai/models/fusion/gmu.py` (lines 50–52) | `Dense(32, activation="relu") -> Dropout(0.2) -> Dense(1, activation="sigmoid")`.                                                                                                                               | **Untrained**                                                                                           | **NO**                            | Integrated into GMU model function, but never fitted on multi-ticker data.              |
| **Inference Engine**          | `src/stock_ai/inference/engine.py`                | Serves predictions via `get_prediction(ticker, date)`.                                                                                                                                                          | Active (Exit code 0)                                                                                    | **MARKET-ONLY**                   | Loads `pred_bilstm_seed_42` from CSVs. Contains no logic or parameters for text inputs. |

---

## 3. Tripartite Component Classification

### Category A: Implemented and Tested

- **Time-Series Market Encoder (`bilstm.py`):** Fully implemented, verified against mathematical specs, trained across 11 active equities, tested with 51 passing automated tests.
- **Parametric & Non-Parametric Baselines (`baselines.py`):** Majority rule, Lag-1 persistence, and Logistic Regression models fully verified and benchmarked on 6,292 test sessions.
- **Temporal Window Builder (`window_builder.py`):** Strict point-in-time sequence generation ($20 \times 16$ matrices ending at $t-1$) preventing target leakage.
- **Market Data Quality & Preprocessing (`indicators.py`, `validator.py`):** 16 leakage-free indicators, schema validation, zero NaNs.

### Category B: Implemented but Not Yet Trained or Validated in Production

- **Gated Multimodal Unit Architecture (`src/stock_ai/models/fusion/gmu.py`):** The Python Keras code is fully implemented and dimensionally valid. However, it has **never been trained, checkpointed, or benchmarked** on the refreshed 11-equity dataset in `C:\PROJECTS\multimodal-stock-ai`.

### Category C: Planned or Missing

- **News Encoder & Tokenizer Module:** Missing from `src/stock_ai/models/news/`.
- **Precomputed News Embedding Store:** Missing from `data/processed/embeddings/`.
- **Forward-Test News Ingestion Pipeline (2024–2026):** Missing; all news data terminates in September 2022.
- **Multimodal Inference & Serving Integration:** Missing; `InferenceEngine` and REST endpoints accept only ticker and date, with no news input pathway.

---

## 4. News Dataset, Coverage & Timing Leakage Audit

A comprehensive audit was executed across all 12 news files in `data/processed/news/*.csv`:

```bash
python scripts/audit_multimodal_readiness.py
```

### Observed News Dataset Coverage:

| Ticker   | Total Rows | Historical Date Coverage | Intraday Timestamps?  | Forward Coverage (2024–2026) | Sample Headline                                                |
| :------- | :--------- | :----------------------- | :-------------------- | :--------------------------- | :------------------------------------------------------------- |
| **AAPL** | 404        | 2021-01-05 to 2022-09-20 | **False** (Date only) | **0 rows**                   | _Two former Apple insiders are building a laser technology..._ |
| **AMT**  | 423        | 2021-01-11 to 2022-09-20 | **False** (Date only) | **0 rows**                   | _Top 10 Dividend Stocks for Income..._                         |
| **ATVI** | 427        | 2021-01-05 to 2022-09-20 | **False** (Date only) | **0 rows**                   | _Activision Blizzard Faces Workplace Lawsuit..._               |
| **BIO**  | 420        | 2021-01-05 to 2022-09-20 | **False** (Date only) | **0 rows**                   | _Bio-Rad Reports Fourth Quarter Results..._                    |
| **JNJ**  | 427        | 2021-01-05 to 2022-09-20 | **False** (Date only) | **0 rows**                   | _Johnson & Johnson Vaccine Rollout Plans..._                   |
| **NDAQ** | 427        | 2021-01-05 to 2022-09-20 | **False** (Date only) | **0 rows**                   | _Nasdaq Reports Monthly Trading Volumes..._                    |
| **NFLX** | 427        | 2021-01-05 to 2022-09-20 | **False** (Date only) | **0 rows**                   | _Netflix Adds Record Subscribers in Q4..._                     |
| **NVDA** | 427        | 2021-01-05 to 2022-09-20 | **False** (Date only) | **0 rows**                   | _Nvidia Announces New RTX GPUs..._                             |
| **PLD**  | 427        | 2021-01-05 to 2022-09-20 | **False** (Date only) | **0 rows**                   | _Prologis Acquires Logistics Facilities..._                    |
| **SBUX** | 427        | 2021-01-05 to 2022-09-20 | **False** (Date only) | **0 rows**                   | _Starbucks Reopens Stores in China..._                         |
| **SCHW** | 427        | 2021-01-05 to 2022-09-20 | **False** (Date only) | **0 rows**                   | _Schwab Client Assets Reach New High..._                       |
| **TSLA** | 404        | 2021-01-06 to 2022-09-20 | **False** (Date only) | **0 rows**                   | _Tesla Delivery Numbers Beat Expectations..._                  |

### Critical Leakage & Provenance Findings:

1. **Zero Forward News Data:** The refreshed forward-testing benchmark spans July 2024 to October 2026. Because all news files terminate on September 20, 2022, **evaluating a multimodal model on the refreshed 2024–2026 benchmark is physically impossible without new data acquisition**.
2. **Post-Close Lookahead Risk (`TIMING-LIMITED`):** The news files record only the calendar date (`YYYY-MM-DD`) without hours, minutes, seconds, or timezones. In financial markets:
   - A news headline published at 7:00 PM EST occurs _after_ market close ($16:00$ EST).
   - If that headline is associated with day $t$'s market record to predict day $t$'s return, it leaks post-close market information into the predictor.
   - Without exact UTC publication timestamps, true point-in-time validity cannot be certified.

---

## 5. Controlled Multimodal Empirical Evaluation (Historical 2022 Benchmark)

Because forward news data is absent, the only empirically valid, reproducible multimodal comparison in the codebase is the historical experiment conducted on **`AMT` across 160 identical held-out test sequences** (`2022-02-01` to `2022-09-20`, 81 Down / 79 Up days), archived in `C:\PROJECTS\Research paer\finbert-lstm\results\gmu_fusion\gmu_metrics_summary.csv`:

### Head-to-Head Comparative Benchmark (AMT, N = 160):

| Model Architecture                           | Threshold ($\tau$) | Accuracy (%) | Balanced Acc (%) |  Macro F1  | F1 (Down) | F1 (Up) | TN  | FP  | FN  | TP  | Timing / Audit Status                |
| :------------------------------------------- | :----------------: | :----------: | :--------------: | :--------: | :-------: | :-----: | :-: | :-: | :-: | :-: | :----------------------------------- |
| **B. Direction Persistence (Lag-1)**         |        0.50        |  **58.13%**  |    **58.13%**    | **0.5812** |  0.5839   | 0.5786  | 47  | 34  | 33  | 46  | **Top Baseline (Heuristic)**         |
| **C1. Next-Day LR (t-1 Endpoint)**           |        0.50        |  **56.25%**  |      56.62%      |   0.5231   |  0.3860   | 0.6602  | 22  | 59  | 11  | 68  | Parametric Linear                    |
| **D. Market-Only BiLSTM**                    |        0.50        |  **52.50%**  |      52.93%      |   0.4639   |  0.2830   | 0.6449  | 15  | 66  | 10  | 69  | Market Deep Learning                 |
| **E. FinBERT Bottleneck (Reference)**        |        0.50        |  **50.62%**  |      50.81%      |   0.4958   |  0.4234   | 0.5683  | 29  | 52  | 27  | 52  | TIMING-LIMITED                       |
| **C2. Next-Day LR (20-Day Flat)**            |        0.50        |  **49.38%**  |      49.41%      |   0.4936   |  0.4841   | 0.5031  | 38  | 43  | 38  | 41  | Parametric Linear                    |
| **A. Training-Majority Baseline**            |        0.50        |  **49.38%**  |      50.00%      |   0.3305   |  0.0000   | 0.6611  |  0  | 81  |  0  | 79  | Heuristic Floor                      |
| **GMU Fusion (Seed 42)**                     |        0.50        |    44.38%    |      44.51%      |   0.4374   |  0.3776   | 0.4972  | 27  | 54  | 35  | 44  | TIMING-LIMITED                       |
| **GMU Fusion (Seed 123)**                    |        0.50        |    47.50%    |      48.05%      |   0.3507   |  0.0667   | 0.6348  |  3  | 78  |  6  | 73  | TIMING-LIMITED                       |
| **GMU Fusion (Seed 7)**                      |        0.50        |    48.12%    |      48.31%      |   0.4703   |  0.3942   | 0.5464  | 27  | 54  | 29  | 50  | TIMING-LIMITED                       |
| **GMU Fusion (Seed 2024)**                   |        0.50        |    49.38%    |      49.73%      |   0.4503   |  0.2957   | 0.6049  | 17  | 64  | 17  | 62  | TIMING-LIMITED                       |
| **GMU Fusion (Seed 999)**                    |        0.50        |    47.50%    |      47.60%      |   0.4720   |  0.4324   | 0.5116  | 32  | 49  | 35  | 44  | TIMING-LIMITED                       |
| **GMU Fusion (5-Seed Mean, $\tau=0.50$)**    |        0.50        |  **47.38%**  |    **47.64%**    | **0.4361** |     —     |    —    |  —  |  —  |  —  |  —  | **$\pm 1.63\%$ Acc, $\pm 0.045$ F1** |
| **GMU Fusion (5-Seed Mean, $\tau^*$ Calib)** |      $\tau^*$      |  **48.12%**  |    **48.05%**    | **0.4666** |     —     |    —    |  —  |  —  |  —  |  —  | **$\pm 3.14\%$ Acc, $\pm 0.039$ F1** |

### Empirical Insights on Fusion:

1. **Multimodal Fusion Degrades Performance:** On identical test sequences, the GMU fusion model (47.38% default, 48.12% calibrated) significantly underperformed market-only BiLSTM (52.50%) and Direction Persistence (58.13%).
2. **Gate Behavior:** Across all 5 seeds, the mean gate activation was $z = 0.4941 \pm 0.0230$, with 49.4% of gates favoring the market branch. The gate did not act as an intelligent noise filter; it settled into an uninformative 50/50 blend that injected text noise into the market representation.

---

## 6. Separation of the ATVI Provenance Blocker

The ATVI data issue is an independent data acquisition constraint that must be kept separate from the multimodal evaluation status:

- **ATVI Status: BLOCKED.** ATVI was acquired by Microsoft and delisted on October 13, 2023. It has 0 sessions in the 2024–2026 evaluation period, and its historical 2010–2022 CSV lacks documented source URLs and corporate action metadata.
- **Multimodal Status: INCOMPLETE.** The multimodal pipeline lacks forward news data across **all 12 stocks** (including active tickers like AAPL, MSFT, and NVDA), lacks a production news encoder, and is not connected to the inference engine.

---

## 7. Prioritized Roadmap to Complete Multimodal Research

To transition Stock-AI from a market-only platform into a verified multimodal research system, the following engineering phases must be executed:

```
[Phase 1: News API Ingestion] ---> [Phase 2: Point-in-Time Alignment] ---> [Phase 3: FinBERT Modular Encoder]
                                                                                        |
[Phase 6: Multimodal REST API] <--- [Phase 5: Multi-Ticker GMU Training] <--- [Phase 4: Embedding Cache (.npy)]
```

### Phase 1: Point-in-Time News Ingestion Pipeline

- Implement an automated news ingestion provider (e.g., Polygon.io or Tiingo News API) that fetches historical articles with **ISO-8601 UTC timestamps** for all active equities from 2020 through 2026.

### Phase 2: Strict Point-in-Time Headline Alignment

- Enforce strict causal alignment: for target day $t$, only headlines published between **Market Close $t-1$ (16:00 EST)** and **Market Open $t$ (09:30 EST)** are permissible. Any headline published during or after market hours on day $t$ must be strictly quarantined to eliminate lookahead bias.

### Phase 3: Modular News Feature Extractor

- Implement `src/stock_ai/models/news/finbert.py` using HuggingFace Transformers (`ProsusAI/finbert`) with GPU batching, max token length truncations (54 tokens), and pooled 768-d output extraction.

### Phase 4: Precomputed Embedding Store

- Extract and cache news embeddings to `data/processed/embeddings/{ticker}_finbert.npy` to eliminate redundant transformer inference during hyperparameter tuning.

### Phase 5: Multi-Ticker GMU Training & Checkpointing

- Build `scripts/train_multimodal.py` to train `build_gmu_fusion_model()` across pooled sequences from all 11 active equities.
- Save calibrated model weights to `models/gmu/{ticker}_gmu_seed_42.keras`.

### Phase 6: Multimodal Inference Integration

- Refactor `InferenceEngine.get_prediction(ticker, headline_text)` to dynamically encode live headlines and evaluate the fused GMU model, with graceful fallback to market-only BiLSTM when news is unavailable.
