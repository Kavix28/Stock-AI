# Stock-AI Platform: Final Closure & Evidence-Based Verification Report

**Project Title:** Stock-AI: Multimodal AI for Stock Trend Prediction  
**GitHub Repository:** [https://github.com/Kavix28/Stock-AI.git](https://github.com/Kavix28/Stock-AI.git)  
**Target Branch:** `main` (Synchronized & Clean)  
**Final Commit Hash:** `b542f76`  
**Audit & Delivery Date:** 2026-10-10  
**Research Validation Status:** **PARTIALLY VALIDATED / BLOCKED ON ATVI PROVENANCE**  
_(The platform, test suite, and 11 active equity forward pipelines are fully verified; however, full replication of the paper's 12-ticker universe is classified as BLOCKED due to unresolved legacy ATVI data provenance.)_

---

## 1. Verified Status Matrix

| Subsystem / Requirement                | Status      | Exit Code / Evidence             | Verification Details                                                                                                                  |
| :------------------------------------- | :---------- | :------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------ |
| **Git Safety & Remote Tracking**       | **PASS**    | Exit 0                           | `origin` tracked to `Kavix28/Stock-AI.git`; `old_stock_ai` preserved; clean working tree; commits pushed cleanly.                     |
| **Active 11 Equities Market Pipeline** | **PASS**    | Exit 0                           | 4,185 sessions/ticker (2010–2026); 16 technical indicators; zero NaNs; zero lookahead leakage.                                        |
| **12-Ticker Paper Replication (ATVI)** | **BLOCKED** | Unresolved                       | Delisted Oct 13, 2023; 0 sessions in 2024–2026 forward test window; legacy CSV lacks verified provider URL and retrieval metadata.    |
| **Benchmark Metric Reconciliation**    | **PASS**    | Recomputed                       | 6,292 out-of-sample sessions; exact rankings verified; market drift vs. model skill clarified; absence of trading edge documented.    |
| **Backend REST API Integrity**         | **PASS**    | Exit 0 (11/11 tests)             | FastAPI TestClient passes all endpoints; exact prediction alignment against saved artifacts; error handling verified.                 |
| **Frontend Production Build**          | **PASS**    | Exit 0 (3.74s)                   | `npm run build` succeeds with zero TypeScript errors; Vite assets generated cleanly.                                                  |
| **Technical Explanatory Video**        | **PASS**    | Exit 0 (ffprobe)                 | 1920x1080 Full HD MP4 (569.93s, 12.02 MB); H.264 video + AAC audio track; previous render safely backed up to `.mp4.bak`.             |
| **Video Subtitles & Captions**         | **PASS**    | Exit 0 (`validate_subtitles.py`) | 37 subtitle cues covering 00:00:01 to 00:09:25 (565s); strictly monotonic timestamps; non-overlapping; synchronized to audio.         |
| **Audio Narration Disclosure**         | **PASS**    | Disclosed                        | Narration is synthetic text-to-speech generated via Windows Speech Synthesis (SAPI `Microsoft David Desktop`), NOT human, NOT neural. |
| **Automated Test Suite**               | **PASS**    | Exit 0 (51/51 tests)             | 100% pass rate across unit, integration, regression, and API verification tests.                                                      |

---

## 2. ATVI Provenance Investigation & Resolution

### A. Provenance Audit Findings

- **Legacy File:** `data/processed/market/ATVI.csv` (3,046 daily sessions from 2010-08-16 to 2022-09-20).
- **Forward Test Coverage (2024–2026):** **0 trading sessions**.
- **Delisting Event:** Activision Blizzard (`ATVI`) was acquired by Microsoft Corporation for $68.7B and delisted on **October 13, 2023**.
- **API Availability:** Public Yahoo Finance / YahooQuery endpoints purge delisted symbols and return HTTP 404 / empty DataFrames.
- **Provenance Gap:** The historical file was inherited from earlier research student capstone archives (`stocks_ts_ATVI_2021-1-4_2022-9-20.csv`). It lacks recorded original download URLs, API query parameters, retrieval timestamps, licensing documentation, and corporate action adjustment logs.

### B. Resolution & Academic Classification

- **12-Ticker Universe Status: BLOCKED.** The project cannot be certified as fully research-validated against the paper's original 12-ticker scope until an authoritative, licensed commercial historical provider (e.g., CRSP, Bloomberg, or Refinitiv) is integrated and audited.
- **11 Active Equities: VERIFIED.** All out-of-sample forward testing (2024–2026) is strictly reported and benchmarked across the 11 active equities (`AAPL`, `AMT`, `BIO`, `JNJ`, `NDAQ`, `NFLX`, `NVDA`, `PLD`, `SBUX`, `SCHW`, `TSLA`). ATVI historical results (180 sessions in 2022) are reported separately in `reports/BENCHMARK_RECOMPUTATION_AUDIT.md`.

---

## 3. Recomputed Benchmark Metrics & Ranking Audit

Evaluated across **6,292 out-of-sample forward test sessions** (July 1, 2024 to October 9, 2026) across 11 active equities:

| Model Architecture                | Accuracy (%) | Balanced Acc (%) | Macro F1   | F1 (Down) | F1 (Up) | TN    | FP    | FN    | TP    | Total Sessions |
| :-------------------------------- | :----------- | :--------------- | :--------- | :-------- | :------ | :---- | :---- | :---- | :---- | :------------- |
| **Majority Baseline**             | **52.13%**   | 50.00%           | 0.3427     | 0.0000    | 0.6853  | 0     | 3,012 | 0     | 3,280 | 6,292          |
| **BiLSTM (3-Seed Ensemble)**      | 51.30%       | 50.12%           | 0.4648     | 0.3043    | 0.6254  | 670   | 2,342 | 722   | 2,558 | 6,292          |
| **BiLSTM (Seed 123)**             | 50.52%       | **50.58%**       | **0.5052** | 0.5012    | 0.5092  | 1,564 | 1,448 | 1,665 | 1,615 | 6,292          |
| **Next-Day LR (20-Day Flat)**     | 50.11%       | 49.66%           | 0.4930     | 0.4290    | 0.5571  | 1,179 | 1,833 | 1,306 | 1,974 | 6,292          |
| **Next-Day LR (t-1 Endpoint)**    | 49.95%       | 49.59%           | 0.4938     | 0.4402    | 0.5475  | 1,238 | 1,774 | 1,375 | 1,905 | 6,292          |
| **Direction Persistence (Lag-1)** | 49.62%       | 49.53%           | 0.4953     | 0.4738    | 0.5168  | 1,427 | 1,585 | 1,585 | 1,695 | 6,292          |
| **BiLSTM (Seed 42)**              | 48.82%       | 49.22%           | 0.4856     | 0.5225    | 0.4486  | 1,762 | 1,250 | 1,970 | 1,310 | 6,292          |
| **BiLSTM (Seed 7)**               | 48.43%       | 48.74%           | 0.4828     | 0.5105    | 0.4551  | 1,692 | 1,320 | 1,925 | 1,355 | 6,292          |

### Critical Empirical Clarifications:

1. **Market Drift vs. Model Skill:** The Majority baseline achieves the top raw accuracy (52.13%) because the test horizon exhibited 3,280 Up sessions out of 6,292 (52.13% positive drift). Naively predicting Up every session captures this drift without requiring any predictive skill, but yields 0.0000 F1 on Down days.
2. **True Model Rankings:** The BiLSTM ensemble achieves 51.30% accuracy, while individual BiLSTM configurations achieve 48.43% to 50.52%. Direction Persistence achieves the top Macro F1 (0.4953) among heuristic baselines, while Logistic Regression achieves 0.4930 among linear models.
3. **Absence of Commercial Trading Edge:** A directional accuracy of ~51% in daily equity markets is consistent with the efficient market hypothesis and does **not** generate a profitable trading strategy after factoring in transaction costs, exchange fees, bid-ask spread friction, and slippage. No model in this repository is claimed to produce a commercial trading edge.

---

## 4. Video, Subtitles & Narration Validation

- **File Path:** `video/output/Stock_AI_Project_Explained.mp4`
- **Backup File:** `video/output/Stock_AI_Project_Explained.mp4.bak`
- **Duration:** Exactly `569.933333` seconds (9 minutes 30 seconds).
- **Streams:**
  - Video: H.264 / AVC, 1920x1080 (16:9), 30 fps, profile High.
  - Audio: AAC, 22,050 Hz, 1 channel mono, 192 kbps.
- **Narration Synthesis Method:** Synthesized speech generated using Windows SAPI (`System.Speech.Synthesis.SpeechSynthesizer`, voice `Microsoft David Desktop`). The narration is explicitly synthetic (rule-based desktop TTS), **not human** and **not neural**.
- **Subtitle Verification:** `video/captions/Stock_AI_Project_Explained.srt` contains 37 subtitle cues extending from 00:00:01 to 00:09:25 (565.0s). Validated via `scripts/validate_subtitles.py`: strictly monotonic, non-overlapping, zero gaps in chapter structure.

---

## 5. Automated Regression Test Suite & Frontend Build

### Python Test Suite (`pytest`):

```bash
pytest tests/ -v
# Exit code: 0
# Result: 51 passed, 1 warning in 1.70s (100% pass rate)
```

### Frontend Build (`vite`):

```bash
cd frontend && npm run build
# Exit code: 0
# Result: built in 3.74s, 0 errors
```

### API Inference Match:

Verified that `InferenceEngine.get_prediction("AAPL")` returns the exact target date (`2026-10-09`), ground truth label (`1`), and predicted label (`1`) stored in `results/latest/predictions/AAPL_predictions.csv`.

---

## 6. Final Git Delivery State

```bash
git log -n 3 --oneline
# b542f76 (HEAD -> main, origin/main) feat(video): complete 37 synchronized subtitles for 570s video and add validator script
# 9c733ee fix(benchmarks): reconcile exact recomputed metrics, absence of trading edge, and ATVI separation
# 6eb54a4 docs(audit): add final evidence-based audit delivery report

git status
# On branch main
# Your branch is up to date with 'origin/main'.
# nothing to commit, working tree clean
```
