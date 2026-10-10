# Stock-AI: Technical Explanatory Video Storyboard

**Output**: `video/output/Stock_AI_Project_Explained.mp4`  
**Resolution**: 1920x1080 Full HD  
**Framerate**: 30 fps  
**Chapters**: 7 Scenes

---

### Scene 1: Title & Research Objective

- **Time**: `00:00 - 00:45` (Duration: 45s)
- **Background**: Deep Navy `#070b14` with subtle radial indigo glow.
- **Top Badge**: `QUANTITATIVE AI RESEARCH` (Cyan `#38bdf8`)
- **Main Heading**: `Stock-AI: Multimodal AI for Stock Trend Prediction`
- **Subheading**: `Deep Multimodal Fusion of Market Sequences & FinBERT Sentiment`
- **Key Cards**:
  - `12 Benchmark Equities` (16-Year History)
  - `Zero Lookahead Leakage` (t-1 Sequenced)
  - `6,292 Forward Test Observations`

---

### Scene 2: Paper Context & Multimodal Formulation

- **Time**: `00:45 - 02:00` (Duration: 75s)
- **Reference**: _Neural Computing and Applications_ (DOI: `10.1007/s00521-024-10303-1`)
- **Mathematical Target**: $y_t = \mathbb{I}(Close_t > Close_{t-1}) \in \{0, 1\}$
- **Dual-Branch Architecture Diagram**:
  - Left Branch: 20-Day Market Time Series $\times$ 16 Features $\rightarrow$ Stacked BiLSTM $\rightarrow$ 64d Projection
  - Right Branch: Daily News Headlines $\rightarrow$ Frozen FinBERT $\rightarrow$ 768d $\rightarrow$ 64d Projection
  - Fusion Node: Joint Representation Space

---

### Scene 3: Data Engineering & Leakage Prevention

- **Time**: `02:00 - 03:30` (Duration: 90s)
- **Timeline Diagram**:
  - `Train (2010 - 2022)`: 35,000+ Samples (MinMaxScaler fitted strictly here)
  - `Validation (2023 - Jun 2024)`: 375 Samples (Threshold Calibration)
  - `Out-of-Sample Test (Jul 2024 - Oct 2026)`: 6,292 Samples (Forward Out-of-Sample)
- **Safety Rules Highlighted**:
  - Window lookback strictly ends on session $t-1$.
  - 33 warmup rows dropped to prevent indicator NaNs.
  - Corporate action audit: ATVI historical fallback provider documented.

---

### Scene 4: Model Architectures & Gated Fusion (GMU)

- **Time**: `03:30 - 05:15` (Duration: 105s)
- **Architectures Covered**:
  - Baselines: Training Majority, Lag-1 Direction Persistence, Logistic Regression.
  - Recurrent: 2-Layer Bidirectional LSTM (64/32 units with 0.2 dropout).
  - Fusion: Gated Multimodal Unit (GMU).
- **GMU Formulation Display**:
  - $z = \sigma(W_z [h_{market} \parallel h_{news}] + b_z)$
  - $h_{fused} = z \odot h_{market} + (1 - z) \odot h_{news}$
  - Explanation: Dynamic noise suppression for noisy financial news.

---

### Scene 5: Empirical Results & Financial Reality

- **Time**: `05:15 - 07:00` (Duration: 105s)
- **Benchmark Results Table (6,292 Forward Test Set)**:
  - Shared Multi-Ticker BiLSTM: **52.13%** (Top Accuracy)
  - Training Majority Baseline: **51.21%**
  - Next-Day Logistic Regression (Flat): **50.01%** (Top Macro F1: 0.4704)
  - Direction Persistence: **49.47%**
  - Individual BiLSTM: **48.58%**
- **Key Takeaways**:
  - Multi-ticker cross-asset pooling improves accuracy by **+3.55%**.
  - In efficient markets, 52–54% represents realistic alpha, debunking inflated claims.

---

### Scene 6: Platform Architecture, Frontend & API

- **Time**: `07:00 - 08:30` (Duration: 90s)
- **Three-Tier Architecture Diagram**:
  - Presentation Tier: React + Vite + TypeScript + Recharts Dashboard
  - Service Tier: FastAPI RESTful API (`/api/v1/tickers`, `/api/v1/predict`, `/api/v1/metrics`)
  - Engine Tier: Modular `stock_ai` package with 34 passing tests.
- **UI Views Highlighted**: Stock Explorer, Model Laboratory, Prediction Demonstration.

---

### Scene 7: Limitations, Future Work & Conclusion

- **Time**: `08:30 - 09:30` (Duration: 60s)
- **Honest Limitations**: Daily news headlines lack intraday timestamps (`TIMING-LIMITED`).
- **Future Directions**: SEC Edgar 8-K filings integration with point-in-time microsecond timestamps; cross-asset graph attention.
- **Repository Reference**: `https://github.com/Kavix28/Stock-AI.git`
- **Closing Academic Disclaimer**: Not financial advice.
