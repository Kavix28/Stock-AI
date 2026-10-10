# Stock-AI Model Architecture & Baseline Guide

This document describes the models implemented in Stock-AI: baselines, linear classifiers, recurrent neural networks, and multimodal fusion units.

---

## 1. Baselines

### A. Majority Baseline (`MajorityBaseline`)

- **Strategy:** Counts class frequencies in training set: $\hat{y} = \arg\max_c \sum_{i=1}^{N_{train}} \mathbb{I}(y_i = c)$.
- **Empirical Value:** 1 (Up) across all equities due to historical US equity upward drift (~51.2% to 54.0%).
- **Role:** Strict benchmark floor. Any model with accuracy below Majority is adding negative value.

### B. Direction Persistence Baseline (`DirectionPersistenceBaseline`)

- **Strategy:** Assumes persistence of directional momentum: $\hat{y}_t = y_{t-1}$.
- **Performance:** 49.47% accuracy across 6,292 out-of-sample observations, demonstrating that daily stock returns are largely uncorrelated ($Autocorr \approx 0$).

### C. Logistic Regression (`LogisticRegressionBaseline`)

- **Variants:**
  1. **t-1 Endpoint:** Fits on the 16 features from session $t-1$.
  2. **20-Day Flat:** Fits on the flattened $20 \times 16 = 320$ feature vector.
- **Performance:** 50.01% accuracy, **0.4704 Macro F1**. Serves as the strongest balanced baseline.

---

## 2. Recurrent Neural Networks

### A. Market-Only BiLSTM

- **Input Shape:** $(20, 16)$ (20 trading days, 16 standardized technical indicators).
- **Architecture:**
  - Layer 1: Bidirectional LSTM (64 units, `return_sequences=True`)
  - Dropout: 0.2
  - Layer 2: Bidirectional LSTM (32 units, `return_sequences=False`)
  - Dropout: 0.2
  - Dense: 32 units, ReLU activation
  - Output: 1 unit, Sigmoid activation
- **Variants:**
  - **Individual BiLSTM:** Trained per ticker (mean accuracy: 48.58%).
  - **Shared Multi-Ticker BiLSTM:** Trained jointly on all 11 active equities (accuracy: **52.13%**).

---

## 3. Gated Multimodal Unit (GMU)

- **Market Branch:** BiLSTM projecting $(20, 16) \rightarrow \mathbb{R}^{64}$.
- **News Branch:** Frozen FinBERT embeddings $(768) \rightarrow \mathbb{R}^{64}$ with Layer Normalization.
- **Learned Gate:**
  $$z = \sigma(W_z [h_{market} \parallel h_{news}] + b_z) \in (0, 1)^{64}$$
- **Convex Combination:**
  $$h_{fused} = z \odot h_{market} + (1 - z) \odot h_{news}$$
- **Advantage:** Enables the network to dynamically attenuate linguistic noise during routine market days while elevating corporate sentiment during major breaking announcements.
