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

## 2. Experimental Controls

1. **Chronological Splitting:**
   - **Train:** 2010-02-22 to 2022-12-31 (~3,200 sessions; 35,000+ total windows).
   - **Validation:** 2023-01-01 to 2024-06-30 (~375 sessions; used for early stopping and threshold calibration).
   - **Test:** 2024-07-01 to 2026-10-09 (572 sessions per active ticker; **6,292 test targets**).
2. **Threshold Calibration:**
   - Decision threshold $\tau$ is chosen via grid search on the validation set to optimize Macro F1.
   - The test set is evaluated strictly once using the chosen $\tau$.
3. **Multi-Seed Stability:**
   - Neural models are evaluated across multiple random seeds (`[42, 123, 7]`) to measure stochastic initialization variance.

---

## 3. Empirical Findings

### Finding 1: Shared Multi-Ticker Advantage

Pooling training windows across 11 active equities (40,000+ training windows) yields **52.13% out-of-sample accuracy**, outperforming individual single-ticker BiLSTM models (48.58%) by **+3.55%**. Cross-asset training acts as a powerful regularizer, mitigating idiosyncratic noise.

### Finding 2: Accuracy vs. Macro F1 Tradeoff

- Recurrent neural networks align with the positive market drift (~52% Up days), maximizing overall accuracy.
- Linear classifiers (Logistic Regression) achieve higher Macro F1 (0.4704) by predicting both upward and downward regimes.

### Finding 3: Financial Reality Bounds

In modern efficient equity markets, directional point-in-time predictability rarely exceeds **52% to 54%**. Claims of 65% to 75% accuracy reported in uncalibrated literature reflect lookahead leakage or test-set snooping.
