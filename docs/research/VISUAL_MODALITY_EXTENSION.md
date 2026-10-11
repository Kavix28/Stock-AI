# Visual Modality Extension: Candlestick & Technical Chart Representations [PROPOSED EXTENSION]

> **Status:** `[PROPOSED EXTENSION]`  
> **Modality Tier:** Tertiary Modality (Visual / Image)  
> **Core Pipeline Impact:** Zero (Non-Destructive, Isolated Experimental Branch)  
> **Pre-requisite:** Unaltered preservation of 2-modality core (Market Time-Series + FinBERT News)

---

## 1. Research Motivation & Hypotheses

While sequential numerical time series (16 technical indicators) and natural language sentiment (FinBERT) capture quantitative trajectory and corporate semantics respectively, technical traders frequently rely on **visual geometric patterns** (e.g., support/resistance levels, head-and-shoulders, consolidation wedges, candlestick wick proportions).

### Core Hypotheses:

1. **Geometric Pattern Extraction:** Convolutional neural networks (CNNs) or Vision Transformers (ViTs) can detect spatial geometric formations in rendered candlestick charts that are obscured in flat 1D sequence tensors.
2. **Orthogonal Information Hypothesis:** If visual chart representations carry information orthogonal to 1D tabular indicators, adding a visual branch to the Gated Multimodal Unit (GMU) should improve out-of-sample directional macro F1.
3. **Redundancy Hypothesis (Null Hypothesis):** Because the 20-day candlestick image is rendered directly from OHLCV data already encoded by the BiLSTM branch, a vision encoder may simply learn a lossy projection of existing market features, adding parameter complexity without statistically significant alpha.

---

## 2. Anti-Leakage Image Generation Protocol

Under strict point-in-time constraints, any chart generation pipeline must satisfy the following zero-leakage invariants:

1. **Strict $t-1$ Cutoff:** The chart image rendered for predicting session $t$ direction must end at session $t-1$. No session $t$ price (Open, High, Low, Close) or volume may be rendered.
2. **Lookback Window:** Fixed 20-day rolling lookback window matching the market BiLSTM sequence length: $[t-20, t-1]$.
3. **No Axis Lookahead / Dynamic Scaling:** Chart axes must be scaled strictly to the local $[t-20, t-1]$ price extremes ($\min, \max$) without reference to future session $t$ prices or full-dataset extrema.
4. **No Future Overlays:** Indicator overlays (e.g., 20-day SMA, Bollinger Bands, Volume bars) rendered on the image must terminate at $t-1$.
5. **Deterministic Artifact Rendering:** Rendered at fixed resolution (e.g., $224 \times 224$ pixels, 3 channels RGB) using Matplotlib or PIL without interactive rendering or non-deterministic anti-aliasing artifacts.

---

## 3. Mathematical Formulation: 3-Way Tri-Modal GMU

The existing two-modality GMU combines market representation $h_m \in \mathbb{R}^{d}$ and news representation $h_n \in \mathbb{R}^{d}$.

The proposed tri-modal GMU extension incorporates visual feature representation $h_v \in \mathbb{R}^{d}$:

### Feature Encoders:

- **Market Branch:** $\text{BiLSTM}(X_{t-20:t-1}) \rightarrow h_m \in \mathbb{R}^{64}$
- **News Branch:** $\text{MLP}(\text{FinBERT}(S_{t-1})) \rightarrow h_n \in \mathbb{R}^{64}$
- **Visual Branch:** $\text{CNN}_{224 \times 224}(I_{t-20:t-1}) \rightarrow h_v \in \mathbb{R}^{64}$ (e.g., ResNet-18 or MobileNetV3 backbone pre-trained on ImageNet, fine-tuned projection).

### Softmax Tri-Modal Gating:

Instead of a binary sigmoid gate, the 3-way GMU employs a softmax gate over the concatenated representation $[h_m \parallel h_n \parallel h_v] \in \mathbb{R}^{192}$:

$$[z_m, z_n, z_v] = \text{Softmax}(W_z [h_m \parallel h_n \parallel h_v] + b_z) \in \Delta^2$$

Where $\Delta^2$ is the 2-simplex, guaranteeing the convex combination invariant:
$$z_m + z_n + z_v \equiv 1.0, \quad z_m, z_n, z_v \ge 0$$

### Fused Representation:

$$h_{\text{fused}} = z_m \odot h_m + z_n \odot h_n + z_v \odot h_v$$

$$\hat{y}_t = \sigma(W_o h_{\text{fused}} + b_o)$$

---

## 4. Experimental Ablation Protocol

To scientifically validate whether the visual modality contributes genuine value, the proposed extension mandates a 4-way ablation evaluated on the identical 924 out-of-sample test instances:

| Configuration                    | Modalities Active      | Encoders                  |
| :------------------------------- | :--------------------- | :------------------------ |
| **Ablation 1 (Baseline)**        | Market Only            | BiLSTM                    |
| **Ablation 2 (Core Research)**   | Market + News          | BiLSTM + FinBERT          |
| **Ablation 3 (Visual Ablation)** | Market + Visual        | BiLSTM + ResNet           |
| **Ablation 4 (Tri-Modal)**       | Market + News + Visual | BiLSTM + FinBERT + ResNet |

### Acceptance Criteria for Visual Adoption:

1. **Alpha Threshold:** Tri-modal fusion must demonstrate a statistically significant increase in Macro F1 ($p < 0.05$ via paired bootstrap test, $B = 10,000$) over the 2-modality Core GMU.
2. **Computational Overhead:** Vision forward pass must not increase inference latency by $> 50\text{ ms}$ on CPU.
3. **Redundancy Guard:** If $z_v \approx 0$ across test instances or Macro F1 improvement is $< 0.005$, the visual modality is rejected under Occam's Razor.
