# Multimodal Research Training and Evaluation Report

**Generated:** 2026-10-11 01:38:35 UTC
**Random Seed:** 42
**Universe:** 12 equities (AAPL, AMT, ATVI, BIO, JNJ, NDAQ, NFLX, NVDA, PLD, SBUX, SCHW, TSLA)
**Evaluation Range:** 2021-01-05 to 2022-09-20 (Verified historical news archive)
**Chronological Split:** Train (< 2022-01-01, N=3012) | Val (2022-01-01 to 2022-05-31, N=1236) | Test (>= 2022-06-01, N=924)

## 1. Executive Summary

This report presents the complete empirical benchmark for the multimodal stock trend prediction platform, evaluating the newly trained **Gated Multimodal Unit (GMU)** against competitive heuristic, linear, and deep market-only baselines under strict anti-leakage invariants.

### Benchmark Results Table

| model                                       |   threshold |   accuracy |   balanced_accuracy |   macro_f1 |   weighted_f1 |   class0_precision |   class0_recall |   class0_f1 |   class1_precision |   class1_recall |   class1_f1 |   tn |   fp |   fn |   tp |   pred_down |   pred_up |   total_n |   auc_roc |   brier_score |
|:--------------------------------------------|------------:|-----------:|--------------------:|-----------:|--------------:|-------------------:|----------------:|------------:|-------------------:|----------------:|------------:|-----:|-----:|-----:|-----:|------------:|----------:|----------:|----------:|--------------:|
| 1. Training-Majority Baseline               |        0.5  |      49.57 |               50    |     0.3314 |        0.3285 |             0      |          0      |      0      |             0.4957 |          1      |      0.6628 |    0 |  466 |    0 |  458 |           0 |       924 |       924 |    0.5    |        0.5043 |
| 2. Direction Persistence (Lag-1)            |        0.5  |      52.16 |               52.16 |     0.5216 |        0.5217 |             0.5259 |          0.5236 |      0.5247 |             0.5174 |          0.5197 |      0.5185 |  244 |  222 |  220 |  238 |         464 |       460 |       924 |    0.5216 |        0.4784 |
| 3a. Next-Day LR (Lag-1 Endpoint, Tau=0.50)  |        0.5  |      50.22 |               50.63 |     0.3629 |        0.3603 |             0.6154 |          0.0343 |      0.065  |             0.4989 |          0.9782 |      0.6608 |   16 |  450 |   10 |  448 |          26 |       898 |       924 |    0.4733 |        0.2537 |
| 3a. Next-Day LR (Lag-1 Endpoint, Tau*=0.55) |        0.55 |      48.59 |               48.49 |     0.4781 |        0.4787 |             0.4921 |          0.603  |      0.5419 |             0.4759 |          0.3668 |      0.4143 |  281 |  185 |  290 |  168 |         571 |       353 |       924 |    0.4733 |        0.2537 |
| 3b. Next-Day LR (20-Day Flat, Tau=0.50)     |        0.5  |      51.84 |               52.07 |     0.4845 |        0.4833 |             0.5475 |          0.2597 |      0.3523 |             0.5092 |          0.7817 |      0.6167 |  121 |  345 |  100 |  358 |         221 |       703 |       924 |    0.5261 |        0.2531 |
| 3b. Next-Day LR (20-Day Flat, Tau*=0.56)    |        0.56 |      51.52 |               51.4  |     0.5063 |        0.5068 |             0.5155 |          0.6438 |      0.5725 |             0.5146 |          0.3843 |      0.44   |  300 |  166 |  282 |  176 |         582 |       342 |       924 |    0.5261 |        0.2531 |
| 4. Market-Only BiLSTM (Tau=0.50)            |        0.5  |      49.57 |               50    |     0.3314 |        0.3285 |             0      |          0      |      0      |             0.4957 |          1      |      0.6628 |    0 |  466 |    0 |  458 |           0 |       924 |       924 |    0.4862 |        0.2507 |
| 4. Market-Only BiLSTM (Tau*=0.53)           |        0.53 |      50.43 |               50    |     0.3353 |        0.3382 |             0.5043 |          1      |      0.6705 |             0      |          0      |      0      |  466 |    0 |  458 |    0 |         924 |         0 |       924 |    0.4862 |        0.2507 |
| 5. GMU Fusion (Tau=0.50)                    |        0.5  |      48.81 |               49.23 |     0.3353 |        0.3326 |             0.2667 |          0.0086 |      0.0166 |             0.4917 |          0.976  |      0.654  |    4 |  462 |   11 |  447 |          15 |       909 |       924 |    0.483  |        0.2513 |
| 5. GMU Fusion (Tau*=0.52)                   |        0.52 |      50    |               49.87 |     0.4876 |        0.4882 |             0.5033 |          0.6502 |      0.5674 |             0.4938 |          0.3472 |      0.4077 |  303 |  163 |  299 |  159 |         602 |       322 |       924 |    0.483  |        0.2513 |
| 6. GMU Zero-News Ablation (Tau=0.50)        |        0.5  |      50.43 |               50    |     0.3353 |        0.3382 |             0.5043 |          1      |      0.6705 |             0      |          0      |      0      |  466 |    0 |  458 |    0 |         924 |         0 |       924 |    0.477  |        0.264  |
| 6. GMU Zero-News Ablation (Tau*=0.52)       |        0.52 |      50.43 |               50    |     0.3353 |        0.3382 |             0.5043 |          1      |      0.6705 |             0      |          0      |      0      |  466 |    0 |  458 |    0 |         924 |         0 |       924 |    0.477  |        0.264  |

## 2. Modality Gate Analysis

- **Mean Market Gate Weight ($z$):** `0.4649`
- **Mean News Gate Weight ($1 - z$):** `0.5351`
- **Mean Market Weight on Days With News:** `0.4649` (News weight: `0.5351`)
## 3. Methodological Invariants Enforced

1. **Zero Future Lookahead:** Technical indicators are computed using strictly backward-looking windows. PrevDayClose and Diff are lagged.
2. **Causal News Alignment:** Only headlines published prior to prediction cutoff enter daily representations.
3. **Preprocessing Isolation:** MinMaxScaler fitted strictly on training data; transform applied downstream.
4. **Decision Threshold Calibration:** $\tau^*$ selected on validation data without touching test sets.
5. **Missing News Integrity:** Sessions with zero news receive deterministic zero-vectors rather than synthetic data.

## 4. Checkpoints & Deliverables

- **GMU Model Checkpoint:** `models/gmu/gmu_fusion.pt`
- **Market BiLSTM Checkpoint:** `models/market/bilstm_market.pt`
- **Model Configuration:** `models/gmu/model_config.json`
- **Predictions CSV:** `results/multimodal/multimodal_predictions.csv`
- **Metrics Summary CSV:** `results/multimodal/multimodal_metrics_summary.csv`
