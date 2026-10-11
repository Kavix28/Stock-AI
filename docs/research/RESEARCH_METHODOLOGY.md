# Stock-AI Research Methodology

This document outlines the theoretical foundation, experimental controls, baseline hierarchy, and empirical findings governing the research evaluation of Stock-AI.

---

## 1. Problem Formulation

We treat next-day stock price directional forecasting as a supervised binary classification task:

$$y_t = \mathbb{I}(Close_t > Close_{t-1}) \in \{0, 1\}$$

Where:

- $y_t = 1$ denotes an **Up** day ($Close_t > Close_{t-1}$).
- $y_t = 0$ denotes a **Down** day ($Close_t \le Close_{t-1}$).

### Information Set Cutoff

The model's information set at forecasting time is strictly bounded:
$$\mathcal{I}_{t-1} = \{ X_{t-20 : t-1}, S_{t-20 : t-1} \}$$

Where $X$ represents the 16 core technical market features and $S$ represents linguistic news embeddings. No information from session $t$ (such as $Open_t$ or intraday prices) is permitted.

---

## 2. Experimental Controls & Partitioning Regimes

### A. Multimodal FinBERT Research Benchmark (2021–2022)

- **Train Horizon:** `2021-01-05` to `2021-12-31` (2,976 samples across 12 equities).
- **Validation Horizon:** `2022-01-03` to `2022-05-31` (1,236 samples; used strictly for early stopping and threshold calibration).
- **Test Horizon:** `2022-06-01` to `2022-09-20` (**924 test instances** across 12 equities). Evaluated strictly once.
- **Modality Composition:** 20-session market technical indicators (16-d) + FinBERT headline representations (768-d CLS) fused via Gated Multimodal Unit (GMU).

### B. Longitudinal Market-Only Benchmark (2010–2026)

- **Train Horizon:** `2010-02-22` to `2022-12-31` (~3,200 sessions; 35,000+ total windows).
- **Validation Horizon:** `2023-01-01` to `2024-06-30` (~375 sessions per ticker).
- **Test Horizon:** `2024-07-01` to `2026-10-09` (572 sessions per active ticker; **6,292 test targets** across 11 active equities).

### C. Threshold Calibration & Anti-Leakage Protocol

1. Decision threshold $\tau$ is chosen via grid search on the validation set to optimize Macro F1.
2. The test set is evaluated strictly once using the pre-calibrated $\tau$.
3. Scalers are fit exclusively on training data and never exposed to validation or test distributions.
4. Neural models are evaluated across multiple random seeds (`[42, 123, 7]`) to measure stochastic initialization variance.

---

## 3. Empirical Findings

### Finding 1: Mode Collapse Mitigation via Multimodal Fusion

In the 2022 out-of-sample bear market test set, Market-Only BiLSTM and Zero-News Ablation collapsed to all-Down predictions under decision threshold $\tau$, yielding $\text{F1(Up)} = 0.0000$ and a degraded $\text{Macro F1} = 0.3353$. Integrating FinBERT 768-d representations via learned GMU convex combination broke mode collapse, achieving diverse probabilities and a balanced $\text{Macro F1} = 0.4876$.

### Finding 2: Lack of Predictive Alpha Over Persistence

Despite breaking mode collapse, the GMU multimodal model (50.00% accuracy, 0.4876 Macro F1) did not outperform simple lag-1 Direction Persistence (52.16% accuracy, 0.5216 Macro F1). In liquid equity markets, daily directional accuracy of ~50–52% yields no commercial trading alpha after transaction friction.

### Finding 3: Shared Multi-Ticker Advantage in Long-Horizon Pretraining

Pooling training windows across 11 active equities (40,000+ training windows) yields **52.13% out-of-sample accuracy**, outperforming individual single-ticker BiLSTM models (48.58%) by **+3.55%**. Cross-asset training acts as an effective regularizer, mitigating idiosyncratic noise.

### Finding 4: Financial Reality Bounds

In modern efficient equity markets, directional point-in-time predictability rarely exceeds **52% to 54%**. Claims of 65% to 75% accuracy reported in uncalibrated literature reflect lookahead leakage, target contamination, or test-set snooping.
