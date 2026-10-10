# Stock-AI: Final Project Gap Analysis & Remediation Matrix

**Document Version:** 1.0.0  
**Audit Date:** 2026-10-11  
**Repository:** `C:\PROJECTS\multimodal-stock-ai`  
**Target Branch:** `main`  
**Primary Execution Environment:** `C:\PROJECTS\multimodal-stock-ai\.venv` (Python 3.11.1)  
**Secondary Research Environment:** `C:\PROJECTS\Research paer\finbert-lstm\.venv` (Python 3.10.11)

---

## 1. Executive Overview

This audit establishes the comprehensive defect, risk, and gap analysis across all subsystems of the **Stock-AI** multimodal stock trend prediction platform. Every identified issue is cataloged with its root cause, severity, affected files, proposed repair, verification method, and implementation status.

In accordance with strict research integrity standards:

- **No data or metrics are fabricated.**
- **Claims are tagged:** `[FROM PAPER]`, `[IMPLEMENTED]`, `[OUR DECISION]`, `[PROPOSED EXTENSION]`, or `[BLOCKED]`.
- **Date-only historical news is strictly tagged `TIMING-LIMITED`.**
- **Forward modern news (2024–2026) is identified as a critical data blocker (`[BLOCKED]`).**
- **ATVI is isolated as delisted (`[BLOCKED]`).**

---

## 2. Defect & Gap Remediation Matrix

| Defect ID  | Category              |   Severity   | Description & Discovered Defect                                                                                                                       | Affected Files                                                              | Proposed Repair                                                                                                                                  | Verification Method                                                                                                     |    Status    |
| :--------- | :-------------------- | :----------: | :---------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------- | :----------: |
| **GAP-01** | **Environment**       |   **HIGH**   | `multimodal-stock-ai\.venv` lacked deep learning and NLP packages (`torch`, `transformers`) required for FinBERT and GMU inference.                   | `pyproject.toml`, `requirements.txt`                                        | Install PyTorch (CPU) and Hugging Face Transformers in `multimodal-stock-ai\.venv` and update dependencies.                                      | Import audit script (`audit_envs_and_models.py`) verifying `torch` and `transformers`.                                  | **RESOLVED** |
| **GAP-02** | **FinBERT Encoder**   | **CRITICAL** | `src/stock_ai/models/news/finbert.py` did not exist. No real FinBERT news encoder was implemented; only static news text existed.                     | `src/stock_ai/models/news/finbert.py`                                       | Implement `FinBertNewsEncoder` using `ProsusAI/finbert` via Hugging Face Transformers. Output deterministic 768-dim pooled representations.      | Unit tests (`test_finbert_encoder.py`) checking output shapes, batch consistency, truncation, and zero-news embeddings. | **RESOLVED** |
| **GAP-03** | **Embedding Cache**   |   **HIGH**   | `data/processed/embeddings/` did not exist. Repeated news encoding would cause massive redundant CPU/GPU latency.                                     | `src/stock_ai/data/news/cache.py`, `data/processed/embeddings/`             | Implement deterministic `NewsEmbeddingCache` storing `.pt` / `.npy` representations with SHA-256 metadata manifests.                             | Unit tests validating cache hit/miss, hash invalidation, and no-news vector creation.                                   | **RESOLVED** |
| **GAP-04** | **GMU Architecture**  | **CRITICAL** | `src/stock_ai/models/fusion/gmu.py` was a TensorFlow/Keras prototype, disconnected from PyTorch FinBERT and the actual training loop.                 | `src/stock_ai/models/fusion/gmu.py`, `src/stock_ai/models/market/bilstm.py` | Implement native PyTorch `GatedMultimodalUnit` and `MarketBiLSTM` with dimensionally verified gated combination $z \odot h_m + (1-z) \odot h_n$. | Gradient and tensor dimension tests; verify modality perturbation changes prediction.                                   | **RESOLVED** |
| **GAP-05** | **Training Pipeline** | **CRITICAL** | No runnable multimodal training script existed (`scripts/train_multimodal.py`). Models could not be trained or checkpointed.                          | `scripts/train_multimodal.py`, `src/stock_ai/training/`                     | Implement end-to-end `train_multimodal.py` with chronological splitting, EarlyStopping, checkpointing to `models/gmu/`, and ablations.           | Run real training experiment on historical 2021–2022 dataset; produce genuine checkpoint.                               | **RESOLVED** |
| **GAP-06** | **Inference Engine**  |   **HIGH**   | `InferenceEngine` only returned static CSV lookups with hardcoded strings (`Shared-BiLSTM-v2.1`). Did not run genuine checkpoints or news fusion.     | `src/stock_ai/inference/engine.py`                                          | Refactor `InferenceEngine` to load trained PyTorch GMU/BiLSTM checkpoints, support news input, report modality status and timing safety.         | Integration tests verifying checkpoint inference, fallback behavior, and missing-news handling.                         | **RESOLVED** |
| **GAP-07** | **Backend API**       |  **MEDIUM**  | `backend/main.py` news endpoint had wrong filename pattern (`{ticker}_news.csv` instead of `{ticker}.csv`), and predict route lacked model selection. | `backend/main.py`                                                           | Fix news file resolution; update `PredictionRequest` and response schemas to report model type, modality used, and timing status.                | FastAPI endpoint regression tests (`test_backend_api.py`, `test_api_verification.py`).                                  | **RESOLVED** |
| **GAP-08** | **Frontend Display**  |  **MEDIUM**  | Frontend did not show timing-validation status, news availability, or distinguish market-only vs. multimodal models.                                  | `frontend/src/views/PredictionDemoView.tsx`, `frontend/src/api.ts`          | Update UI cards and API client to display modality badges (`Market-Only` vs `Multimodal GMU`), timing badges, and headline counts.               | `npm run build` production bundle verification (passes cleanly).                                                        | **RESOLVED** |
| **GAP-09** | **Modern News Gap**   |   **HIGH**   | 0 news headlines exist for July 2024 to October 2026. Without API keys, forward multimodal benchmarking cannot run.                                   | `data/processed/news/`, `reports/PHASE_1_NEWS_PROVENANCE_AUDIT.md`          | Explicitly classify forward multimodal evaluation as `[BLOCKED]`. Run multimodal evaluation on verified historical 2021–2022 archive.            | Provenance audit documentation and historical benchmark execution.                                                      | **RESOLVED** |
| **GAP-10** | **ATVI Delisting**    |  **MEDIUM**  | ATVI delisted Oct 2023. Historical market data exists but modern data is physically impossible.                                                       | `backend/main.py`, `reports/`                                               | Keep ATVI isolated with explicit `BLOCKED (Delisted)` status across all metrics and routes.                                                      | Automated test asserting ATVI is flagged delisted.                                                                      | **RESOLVED** |

---

## 3. Methodological Design Decisions

1. **Modeling Framework: PyTorch (`[OUR DECISION]`)**  
   PyTorch 2.14+ (CPU-compatible) is selected as the primary ML framework across FinBERT, BiLSTM, and GMU. This avoids framework bridging between TensorFlow and Hugging Face Transformers, ensures deterministic weight serialization (`.pt`), and runs efficiently across developer and server environments without GPU constraints.

2. **News Representation: CLS Token vs. Mean Pooling (`[FROM PAPER]`, `[OUR DECISION]`)**  
   We utilize the 768-dimensional `[CLS]` token representation from `ProsusAI/finbert` passed through a dedicated linear projection layer with layer normalization into a 64-dimensional latent space, matching the BiLSTM market projection dimension.

3. **Missing News Protocol (`[OUR DECISION]`)**  
   On trading sessions where no news headlines exist in the observation window, the news vector is initialized as a learned zero-news prior (or explicit zero-tensor with mask) rather than fabricating synthetic headlines or imputing arbitrary sentiment.

4. **Prediction Horizon & Anti-Leakage (`[FROM PAPER]`, `[IMPLEMENTED]`)**  
   Market features $\mathbf{X}_t \in \mathbb{R}^{20 \times 16}$ are computed strictly from sessions $t-20$ through $t-1$. The label $y_t \in \{0, 1\}$ represents next-session direction:
   $$y_t = \mathbb{I}(\text{Close}_t - \text{Close}_{t-1} > 0)$$
   Permitted news must be timestamped strictly before market open of session $t$.
