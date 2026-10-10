"""
Synthesize voiceover narration for the 7 video chapters using Windows Speech Synthesizer
and mux into Stock_AI_Project_Explained.mp4 with AAC audio track.
"""
import os
import subprocess
import json

ROOT = r"C:\PROJECTS\multimodal-stock-ai"
AUDIO_DIR = os.path.join(ROOT, "video", "assets", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

CHAPTERS = [
    {
        "id": 1,
        "duration": 45,
        "text": (
            "Welcome to the technical overview of Stock-AI: a rigorous quantitative machine learning platform "
            "investigating multimodal deep fusion for stock trend prediction. "
            "Financial markets represent one of the lowest signal-to-noise ratio domains in applied artificial intelligence. "
            "While academic literature frequently reports dramatic directional forecasting accuracies exceeding 65 or 70 percent, "
            "such results frequently collapse under strict out-of-sample forward testing. "
            "In this project, our primary research objective is twofold: first, to establish an academically rigorous, "
            "leakage-free benchmark across twelve diverse equities over sixteen years of historical market data; and second, "
            "to systematically evaluate whether fusing high-dimensional financial news sentiment with temporal price sequences "
            "yields genuine predictive alpha beyond classical econometric baselines."
        )
    },
    {
        "id": 2,
        "duration": 75,
        "text": (
            "Our research builds upon recent work published in Neural Computing and Applications, "
            "investigating deep multimodal architectures that combine numerical price sequences with financial news headlines. "
            "Formally, the directional trend forecasting problem is defined as binary classification. "
            "Given a twenty-day rolling sequence of market prices and technical indicators terminating on trading session t minus one, "
            "alongside news headlines aggregated over the preceding interval, the model must forecast whether the close price on day t "
            "will exceed the close price on day t minus one. "
            "To model this, we construct a dual-branch neural architecture. "
            "The market branch processes a twenty-day by sixteen-feature temporal tensor through stacked Bidirectional LSTMs, "
            "capturing sequential momentum and mean-reversion dynamics. "
            "The linguistic branch processes financial news headlines through FinBERT, a domain-specialized transformer pre-trained on corporate filings and market commentary, "
            "extracting dense 768-dimensional semantic embeddings. "
            "These two streams are projected into a common sixty-four-dimensional latent space for downstream fusion."
        )
    },
    {
        "id": 3,
        "duration": 90,
        "text": (
            "The cornerstone of any quantitative research system is absolute data integrity. "
            "In financial forecasting, data leakage can effortlessly produce fictitious accuracy gains. "
            "We eliminate leakage through three non-negotiable architectural guarantees. "
            "First: Strict Chronological Partitioning. We divide our sixteen-year dataset into three contiguous, non-overlapping horizons: "
            "training through December 2022 with over thirty-five thousand observations; validation through June 2024 for hyperparameter tuning; "
            "and an untouched out-of-sample forward test horizon from July 2024 through October 2026 comprising six thousand two hundred and ninety-two target sessions. "
            "Second: Sequence Isolation. Every twenty-day market window strictly terminates on session t minus one. "
            "Under no circumstances do current-day open, high, low, or volume enter the model prior to forecasting day t. "
            "MinMax scaling parameters are fitted solely on training data, preventing distributional leakage across temporal boundaries. "
            "Third: Handling Corporate Actions. We document and audit our twelve-stock universe with complete transparency. "
            "For Activision Blizzard, which was delisted in October 2023 following its acquisition by Microsoft, "
            "we enforce a historical fallback provider, evaluating its performance strictly on historical 2022 targets rather than fabricating modern observations."
        )
    },
    {
        "id": 4,
        "duration": 105,
        "text": (
            "We explore four tiers of model complexity. "
            "Tier One comprises our heuristics and linear baselines: the empirical Training-Majority baseline, "
            "the Lag-One Direction Persistence baseline, and Scikit-learn Logistic Regression operating on either the t-minus-one endpoint or the flattened twenty-day window. "
            "Tier Two introduces deep recurrent networks: a two-layer Bidirectional LSTM with dropout regularization, "
            "evaluated both on single-ticker regimes and as a shared pooled model trained simultaneously across eleven active equities. "
            "Tier Three explores multimodal fusion. Rather than naive concatenation, which allows high-variance news representations to corrupt stable market signals, "
            "we implement a Gated Multimodal Unit, or GMU. "
            "The GMU computes a learned gating vector z via a sigmoid activation over the concatenated market and linguistic projections. "
            "It then performs a convex combination: z element-wise multiplied by the market representation, plus one minus z multiplied by the news representation. "
            "This allows the neural network to dynamically suppress news noise during routine market regimes while elevating linguistic sentiment during major corporate news catalysts."
        )
    },
    {
        "id": 5,
        "duration": 105,
        "text": (
            "Let us examine the empirical results across our six thousand two hundred and ninety-two forward test observations. "
            "Our highest overall directional accuracy is achieved by the Shared Multi-Ticker BiLSTM at 52.13 percent, "
            "closely followed by the Training-Majority baseline at 51.21 percent. "
            "While an accuracy of 52 percent may appear modest compared to uncalibrated literature claims, "
            "this is precisely what empirical quantitative theory predicts: in modern efficient markets, daily directional predictability rarely exceeds fifty-two to fifty-four percent. "
            "Crucially, our shared multi-ticker architecture outperforms individual per-ticker models by three point five five percentage points, "
            "confirming that pooling market sequences across eleven diverse equities acts as an effective regularizer, mitigating idiosyncratic overfitting. "
            "We also uncover a critical trade-off between Accuracy and Macro F1. "
            "While neural networks converge toward predicting the empirical market drift—where approximately fifty-two percent of trading sessions are upward—"
            "linear models achieve higher Macro F1 scores around 0.47 by making balanced predictions across both bull and bear sessions. "
            "Validation threshold calibration enables researchers to shift between maximum accuracy and balanced sensitivity depending on their risk objectives."
        )
    },
    {
        "id": 6,
        "duration": 90,
        "text": (
            "To make these research findings accessible, transparent, and actionable, we have transformed Stock-AI into a full-stack platform. "
            "The core computational engine is packaged as a modular, testable Python library under src/stock_ai, "
            "complete with automated data pipelines, technical indicator calculators, and forty passing automated unit, integration, and regression tests. "
            "Above the core library sits a high-performance FastAPI service. "
            "It exposes RESTful endpoints for ticker metadata, historical OHLCV series, technical indicators, benchmark metrics, and point-in-time inference. "
            "Finally, we built a modern research dashboard using React, TypeScript, and Vite. "
            "The dashboard features an interactive Stock Explorer with Bollinger Bands and EMAs, a Model Laboratory for inspecting empirical ablation metrics, "
            "and a Prediction Demonstration view that executes model inference with explicit risk disclaimers and ground-truth validation."
        )
    },
    {
        "id": 7,
        "duration": 60,
        "text": (
            "In conclusion, Stock-AI demonstrates that rigorous quantitative methodology, strict leakage prevention, "
            "and objective baselining are essential for advancing financial machine learning. "
            "We transparently acknowledge the limitations of current multimodal news fusion: "
            "daily-aggregated headlines lack intraday timestamps, warranting a conservative TIMING-LIMITED designation to avoid post-close lookahead contamination. "
            "Future research will integrate high-frequency SEC Edgar 8-K filings with microsecond timestamps and explore graph-based cross-asset attention mechanisms. "
            "The complete codebase, reproduction scripts, verified benchmark artifacts, and interactive dashboard are open source on GitHub at Kavix28 slash Stock-AI. "
            "Thank you for exploring Stock-AI."
        )
    }
]

def synthesize_chapter(ch):
    wav_path = os.path.join(AUDIO_DIR, f"chapter_{ch['id']:02d}.wav")
    text_clean = ch["text"].replace('"', "'")
    ps_cmd = f"""
    Add-Type -AssemblyName System.Speech
    $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $speak.SelectVoice('Microsoft David Desktop')
    $speak.Rate = 0
    $speak.SetOutputToWaveFile('{wav_path}')
    $speak.Speak("{text_clean}")
    $speak.Dispose()
    """
    res = subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error in TTS for chapter {ch['id']}: {res.stderr}")
    else:
        print(f"Synthesized: chapter_{ch['id']:02d}.wav")
    return wav_path

def pad_and_concat():
    padded_wavs = []
    for ch in CHAPTERS:
        raw_wav = synthesize_chapter(ch)
        target_dur = ch["duration"]
        padded_wav = os.path.join(AUDIO_DIR, f"chapter_{ch['id']:02d}_padded.wav")
        # Pad with silence or trim to exactly target_dur
        # apad filter adds silence at the end up to target_dur
        cmd = [
            "ffmpeg", "-y",
            "-i", raw_wav,
            "-af", f"apad=whole_dur={target_dur}",
            "-t", str(target_dur),
            padded_wav
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        padded_wavs.append(padded_wav)

    # Concat all padded wavs
    concat_list = os.path.join(AUDIO_DIR, "audio_concat.txt")
    with open(concat_list, "w") as f:
        for pw in padded_wavs:
            clean_p = pw.replace("\\", "/")
            f.write(f"file '{clean_p}'\n")

    full_audio = os.path.join(AUDIO_DIR, "full_narration.wav")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list,
        "-c", "copy",
        full_audio
    ], check=True, capture_output=True)
    print(f"Full audio concatenated: {full_audio}")
    return full_audio

def mux_video_with_audio(audio_path):
    video_in = os.path.join(ROOT, "video", "output", "Stock_AI_Project_Explained.mp4")
    temp_muxed = os.path.join(ROOT, "video", "output", "Stock_AI_Project_Explained_muxed.mp4")

    # Mux video and audio
    cmd = [
        "ffmpeg", "-y",
        "-i", video_in,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        temp_muxed
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Mux error:", res.stderr)
    else:
        # Replace original with muxed
        os.replace(temp_muxed, video_in)
        print("Muxed video with audio successfully replaced original!")

if __name__ == "__main__":
    audio = pad_and_concat()
    mux_video_with_audio(audio)
