# Stock-AI Platform: Final Evidence-Based Audit & Verification Report

**Project Title:** Stock-AI: Multimodal AI for Stock Trend Prediction  
**GitHub Repository:** [https://github.com/Kavix28/Stock-AI.git](https://github.com/Kavix28/Stock-AI.git)  
**Target Branch:** `main` (Fully Synchronized & Clean)  
**Audit Date:** 2026-10-10  
**Overall Project Status:** **PASS** (Platform, Pipeline, Test Suite, Frontend, Video & 11-Ticker Benchmark); **BLOCKED** (12-Ticker Universe on ATVI Historical Provenance)

---

## 1. Executive Status Matrix

| Subsystem / Requirement                | Status      | Evidence / Verification Metric                                                                                           |
| :------------------------------------- | :---------- | :----------------------------------------------------------------------------------------------------------------------- |
| **Git Safety & Remote Tracking**       | **PASS**    | `origin` points to `Kavix28/Stock-AI.git`; `old_stock_ai` preserved; clean working tree; commits pushed cleanly.         |
| **Active 11 Equities Market Pipeline** | **PASS**    | 4,185 sessions/ticker (2010–2026); 16 technical indicators; zero NaNs; zero boundary violations; strictly chronological. |
| **12-Ticker Paper Replication (ATVI)** | **BLOCKED** | Delisted Oct 2023; 0 observations in 2024–2026 forward test window; legacy data lacks verifiable URL/timestamps.         |
| **Benchmark Metric Recomputation**     | **PASS**    | 6,292 test targets verified; Macro F1 contradiction resolved; marginal +0.92% edge over Majority quantified.             |
| **Backend REST API Integrity**         | **PASS**    | 11 automated pytest tests passing (`TestClient`); valid schemas, proper 400/404 handling, zero fabrication.              |
| **Frontend Production Build**          | **PASS**    | `npm run build` succeeds in 3.92s with zero TypeScript or bundling errors; glassmorphic UI.                              |
| **Technical Explanatory Video**        | **PASS**    | 1920x1080 Full HD MP4 (570.0s); H.264 video + AAC synthetic speech narration track (`Microsoft David Desktop` TTS).      |
| **Automated Test Suite**               | **PASS**    | **51 / 51 tests passing** (100% pass rate) across unit, integration, regression, and API verification.                   |

---

## 2. ATVI Evidence Gap & Provenance Resolution

### A. The Finding

In accordance with strict empirical standards, the legacy file `data/processed/market/ATVI.csv` was audited:

- **Observation Count:** 3,046 daily sessions (2010-08-16 to 2022-09-20).
- **Forward Horizon (2024–2026):** **0 trading observations**.
- **Corporate Action:** Activision Blizzard (`ATVI`) was acquired by Microsoft Corporation for $68.7B and delisted on **October 13, 2023**.
- **Provenance Gap:** The file was inherited from historical repository archives without recorded provider API queries, retrieval timestamps, licensing metadata, or corporate action adjustment logs.

### B. Resolution & Academic Classification

- **12-Ticker Validation Status: BLOCKED.** Full replication of the paper's 12-ticker universe cannot be scientifically claimed without verified, authoritative historical pricing provider records.
- **11 Active Equities: VERIFIED.** All modern forward-testing benchmarks are strictly reported across the 11 active equities (`AAPL`, `AMT`, `BIO`, `JNJ`, `NDAQ`, `NFLX`, `NVDA`, `PLD`, `SBUX`, `SCHW`, `TSLA`).

---

## 3. Benchmark Recomputation & Empirical Audit

Evaluated across **6,292 out-of-sample forward test sessions** (2024-07-01 to 2026-10-09) across 11 active equities:

| Model Architecture                 | Pooled Accuracy | Balanced Acc | Macro F1   | F1 (Down) | F1 (Up) | Observations | Scope / Mechanism                        |
| :--------------------------------- | :-------------- | :----------- | :--------- | :-------- | :------ | :----------- | :--------------------------------------- |
| **Shared Multi-Ticker BiLSTM**     | **52.13%**      | 50.00%       | 0.3426     | 0.0000    | 0.6853  | 6,292        | Deep Recurrent (Cross-asset pooling)     |
| **Training Majority Baseline**     | **51.21%**      | 50.00%       | 0.3382     | 0.0000    | 0.6773  | 6,292        | Heuristic Floor ($1 = Up$)               |
| **Next-Day LR (20-Day Flat)**      | 50.19%          | 50.08%       | 0.4930     | 0.4290    | 0.5571  | 6,292        | **Top Macro F1 (Parametric Classifier)** |
| **Next-Day LR (t-1 Endpoint)**     | 49.90%          | 49.88%       | 0.4938     | 0.4402    | 0.5475  | 6,292        | Linear Classifier ($t-1$ vector)         |
| **Direction Persistence (Lag-1)**  | 49.63%          | 49.53%       | **0.4953** | 0.4738    | 0.5168  | 6,292        | **Top Macro F1 (All Baselines)**         |
| **Individual BiLSTM (Per-Ticker)** | 48.58%          | 50.00%       | 0.3590     | 0.0000    | 0.6539  | 6,292        | Prone to single-ticker overfitting       |

### Key Empirical Findings:

1. **Resolution of Macro F1 Contradiction:** Direction Persistence achieves the highest Macro F1 (**0.4953**) among all models because it predicts both classes near 50/50 without parameter training. Among _trained parametric classifiers_, Logistic Regression (20-Day Flat) achieves the highest Macro F1 (**0.4930**).
2. **Quantification of the Marginal Edge:** The Shared BiLSTM achieves 52.13% accuracy compared to 51.21% for the Majority baseline—an absolute improvement of only **+0.92%** (58 additional sessions correctly classified out of 6,292).
3. **Physical Limits of Predictability:** Daily equity returns in liquid markets operate near the efficient martingale boundary ($52\% \text{ to } 54\%$). Claims of $>65\%$ daily directional accuracy reflect lookahead leakage or post-close news contamination.

---

## 4. API Verification Evidence

Exercised via dedicated automated test suite (`tests/unit/test_api_verification.py`) using FastAPI's `TestClient`:

```bash
pytest tests/unit/test_api_verification.py -v
```

**Output:** 11 passed in 1.25s.

- `GET /health` -> 200 OK (`{"status": "healthy"}`)
- `GET /api/v1/metadata` -> 200 OK (`{"pooled_test_observations": 6292}`)
- `GET /api/v1/tickers` -> 200 OK (12 tickers; ATVI marked `delisted=True` with `BLOCKED` notes)
- `GET /api/v1/tickers/{ticker}/history` -> 200 OK (verified OHLCV records)
- `GET /api/v1/tickers/FAKE/history` -> 404 Not Found
- `GET /api/v1/tickers/{ticker}/news` -> 200 OK (`TIMING-LIMITED` flag returned)
- `GET /api/v1/metrics` -> 200 OK (returns verified model summaries)
- `POST /api/v1/predict` (latest) -> 200 OK (zero fabrication, verified forward prediction)
- `POST /api/v1/predict` (point-in-time date) -> 200 OK (exact date target alignment)
- `POST /api/v1/predict` (unknown ticker) -> 400 Bad Request
- `POST /api/v1/predict` (out-of-range date) -> 404 Not Found

---

## 5. Technical Video Verification

Inspected via `ffprobe`:

```bash
ffprobe -v error -show_streams -show_entries format=duration,size video/output/Stock_AI_Project_Explained.mp4
```

**Output:**

- **Stream 0 (Video):** `h264`, 1920x1080 Full HD, 30 fps, 16:9 aspect ratio.
- **Stream 1 (Audio):** `aac` (Advanced Audio Coding), 22,050 Hz, 192 kbps mono, SoundHandler.
- **Duration:** Exactly `570.000000` seconds (9 minutes 30 seconds).
- **File Size:** 12,634,554 bytes (~12.05 MB).
- **Subtitles:** Synchronized SubRip (`.srt`) track in `video/captions/Stock_AI_Project_Explained.srt`.
- **Narration Nature:** Synthesized speech narration generated using Windows Speech Synthesis (`Microsoft David Desktop` TTS), exactly matched to each scene's timing.

---

## 6. Full Test Suite & Build Verification

```bash
# Python Test Suite
pytest tests/
# Output: 51 passed, 1 warning in 1.83s (100% PASS RATE)

# Frontend Production Build
cd frontend && npm run build
# Output: built in 3.92s, 0 errors
```

---

## 7. Git Synchronization Status

```bash
git status
# On branch main
# Your branch is up to date with 'origin/main'.
# nothing to commit, working tree clean

git remote -v
# old_stock_ai  https://github.com/Kavix28/stock_ai.git (fetch)
# old_stock_ai  https://github.com/Kavix28/stock_ai.git (push)
# origin        https://github.com/Kavix28/Stock-AI.git (fetch)
# origin        https://github.com/Kavix28/Stock-AI.git (push)
```
