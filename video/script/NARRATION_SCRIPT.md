# Stock-AI: Multimodal AI for Stock Trend Prediction

## Technical Video Narration Script

- **Target Output**: `video/output/Stock_AI_Project_Explained.mp4`
- **Total Duration**: ~9 minutes 30 seconds (570 seconds)
- **Target Audience**: Quantitative Researchers, Machine Learning Engineers, Academic Reviewers
- **Tone**: Authoritative, rigorously quantitative, intellectually honest, scientific

---

### CHAPTER 1: Title and Research Objective (0:00 – 0:45)

**Visual**: Deep navy title slide with animated glowing badge "Stock-AI: Multimodal AI for Stock Trend Prediction", DOI reference, author information, and core research question.

**Voiceover Narration**:
"Welcome to the technical overview of Stock-AI: a rigorous quantitative machine learning platform investigating multimodal deep fusion for stock trend prediction.

Financial markets represent one of the lowest signal-to-noise ratio domains in applied artificial intelligence. While academic literature frequently reports dramatic directional forecasting accuracies exceeding 65 or 70 percent, such results frequently collapse under strict out-of-sample forward testing.

In this project, our primary research objective is twofold: first, to establish an academically rigorous, leakage-free benchmark across twelve diverse equities over sixteen years of historical market data; and second, to systematically evaluate whether fusing high-dimensional financial news sentiment with temporal price sequences yields genuine predictive alpha beyond classical econometric baselines."

---

### CHAPTER 2: Paper Context and Multimodal Formulation (0:45 – 2:00)

**Visual**: DOI reference card (10.1007/s00521-024-10303-1), dual-stream multimodal architecture diagram showing Market Sequence (left) and News Stream (right) projecting into a joint fusion space.

**Voiceover Narration**:
"Our research builds upon recent work published in Neural Computing and Applications, investigating deep multimodal architectures that combine numerical price sequences with financial news headlines.

Formally, the directional trend forecasting problem is defined as binary classification. Given a twenty-day rolling sequence of market prices and technical indicators terminating on trading session t minus one, alongside news headlines aggregated over the preceding interval, the model must forecast whether the close price on day t will exceed the close price on day t minus one.

To model this, we construct a dual-branch neural architecture. The market branch processes a twenty-day by sixteen-feature temporal tensor through stacked Bidirectional LSTMs, capturing sequential momentum and mean-reversion dynamics. The linguistic branch processes financial news headlines through FinBERT, a domain-specialized transformer pre-trained on corporate filings and market commentary, extracting dense 768-dimensional semantic embeddings. These two streams are projected into a common sixty-four-dimensional latent space for downstream fusion."

---

### CHAPTER 3: Data Engineering and Leakage Prevention (2:00 – 3:30)

**Visual**: Chronological partition timeline (Train: 2010–2022, Val: 2023–mid 2024, Test: mid 2024–late 2026). Mathematical definition of 20-day window ending on t-1. Provenance alert for ATVI delisting.

**Voiceover Narration**:
"The cornerstone of any quantitative research system is absolute data integrity. In financial forecasting, data leakage can effortlessly produce fictitious accuracy gains. We eliminate leakage through three non-negotiable architectural guarantees.

First: Strict Chronological Partitioning. We divide our sixteen-year dataset into three contiguous, non-overlapping horizons: training through December 2022 with over thirty-five thousand observations; validation through June 2024 for hyperparameter tuning; and an untouched out-of-sample forward test horizon from July 2024 through October 2026 comprising six thousand two hundred and ninety-two target sessions.

Second: Sequence Isolation. Every twenty-day market window strictly terminates on session t minus one. Under no circumstances do current-day open, high, low, or volume enter the model prior to forecasting day t. MinMax scaling parameters are fitted solely on training data, preventing distributional leakage across temporal boundaries.

Third: Handling Corporate Actions. We document and audit our twelve-stock universe with complete transparency. For Activision Blizzard, which was delisted in October 2023 following its acquisition by Microsoft, we enforce a historical fallback provider, evaluating its performance strictly on historical 2022 targets rather than fabricating modern observations."

---

### CHAPTER 4: Model Architectures and Gated Fusion (3:30 – 5:15)

**Visual**: Mathematical equation of Gated Multimodal Unit (GMU), tensor dimension annotations, Bidirectional LSTM layer diagrams.

**Voiceover Narration**:
"We explore four tiers of model complexity.

Tier One comprises our heuristics and linear baselines: the empirical Training-Majority baseline, the Lag-One Direction Persistence baseline, and Scikit-learn Logistic Regression operating on either the t-minus-one endpoint or the flattened twenty-day window.

Tier Two introduces deep recurrent networks: a two-layer Bidirectional LSTM with dropout regularization, evaluated both on single-ticker regimes and as a shared pooled model trained simultaneously across eleven active equities.

Tier Three explores multimodal fusion. Rather than naive concatenation, which allows high-variance news representations to corrupt stable market signals, we implement a Gated Multimodal Unit, or GMU. The GMU computes a learned gating vector z via a sigmoid activation over the concatenated market and linguistic projections. It then performs a convex combination: z element-wise multiplied by the market representation, plus one minus z multiplied by the news representation. This allows the neural network to dynamically suppress news noise during routine market regimes while elevating linguistic sentiment during major corporate news catalysts."

---

### CHAPTER 5: Empirical Results and Financial Reality (5:15 – 7:00)

**Visual**: Full benchmark comparison table with 6,292 out-of-sample observations. Bar charts comparing Shared BiLSTM (52.13%), Majority (51.21%), and Baselines. Macro F1 trade-off breakdown.

**Voiceover Narration**:
"Let us examine the empirical results across our six thousand two hundred and ninety-two forward test observations.

Our highest overall directional accuracy is achieved by the Shared Multi-Ticker BiLSTM at 52.13 percent, closely followed by the Training-Majority baseline at 51.21 percent. While an accuracy of 52 percent may appear modest compared to uncalibrated literature claims, this is precisely what empirical quantitative theory predicts: in modern efficient markets, daily directional predictability rarely exceeds fifty-two to fifty-four percent.

Crucially, our shared multi-ticker architecture outperforms individual per-ticker models by three point five five percentage points, confirming that pooling market sequences across eleven diverse equities acts as an effective regularizer, mitigating idiosyncratic overfitting.

We also uncover a critical trade-off between Accuracy and Macro F1. While neural networks converge toward predicting the empirical market drift—where approximately fifty-two percent of trading sessions are upward—linear models achieve higher Macro F1 scores around 0.47 by making balanced predictions across both bull and bear sessions. Validation threshold calibration enables researchers to shift between maximum accuracy and balanced sensitivity depending on their risk objectives."

---

### CHAPTER 6: Platform Architecture, Frontend, and API (7:00 – 8:30)

**Visual**: High-level platform architecture diagram showing React/Vite dashboard, FastAPI REST endpoints, and modular `stock_ai` engine. Screenshots of Stock Explorer and Model Laboratory.

**Voiceover Narration**:
"To make these research findings accessible, transparent, and actionable, we have transformed Stock-AI into a full-stack platform.

The core computational engine is packaged as a modular, testable Python library under src/stock_ai, complete with automated data pipelines, technical indicator calculators, and thirty-four passing automated unit and integration tests.

Above the core library sits a high-performance FastAPI service. It exposes RESTful endpoints for ticker metadata, historical OHLCV series, technical indicators, benchmark metrics, and point-in-time inference.

Finally, we built a modern research dashboard using React, TypeScript, and Vite. The dashboard features an interactive Stock Explorer with Bollinger Bands and EMAs, a Model Laboratory for inspecting empirical ablation metrics, and a Prediction Demonstration view that executes model inference with explicit risk disclaimers and ground-truth validation."

---

### CHAPTER 7: Limitations, Future Work, and Conclusion (8:30 – 9:30)

**Visual**: Summary takeaway bullet points, GitHub repository URL (`https://github.com/Kavix28/Stock-AI.git`), academic disclaimer, and closing slide.

**Voiceover Narration**:
"In conclusion, Stock-AI demonstrates that rigorous quantitative methodology, strict leakage prevention, and objective baselining are essential for advancing financial machine learning.

We transparently acknowledge the limitations of current multimodal news fusion: daily-aggregated headlines lack intraday timestamps, warranting a conservative TIMING-LIMITED designation to avoid post-close lookahead contamination. Future research will integrate high-frequency SEC Edgar 8-K filings with microsecond timestamps and explore graph-based cross-asset attention mechanisms.

The complete codebase, reproduction scripts, verified benchmark artifacts, and interactive dashboard are open source on GitHub at Kavix28 slash Stock-AI. Thank you for exploring Stock-AI."
