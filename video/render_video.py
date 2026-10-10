"""
Video rendering pipeline for Stock-AI project explanatory video.
Generates 1920x1080 high-resolution visual slides and compiles into MP4 with ffmpeg.
"""

import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ASSETS_DIR = r"C:\PROJECTS\multimodal-stock-ai\video\assets"
OUTPUT_DIR = r"C:\PROJECTS\multimodal-stock-ai\video\output"
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 7 Scenes with their duration in seconds
SCENES = [
    {
        "id": 1,
        "title": "Stock-AI: Multimodal AI for Stock Trend Prediction",
        "subtitle": "A Rigorous, Leakage-Free Benchmark Across 12 Equities Over 16 Years",
        "duration": 45,
        "bullet_points": [
            "Research Domain: Directional stock movement forecasting in low-SNR regime",
            "Core Objective: Contrast empirical alpha against persistent baselines",
            "Multi-Asset Scope: 12 S&P 500 / Nasdaq equities spanning 2010 to 2026",
            "Integrity Standard: Zero future-data leakage; strict point-in-time isolation"
        ],
        "badge": "QUANTITATIVE AI RESEARCH PLATFORM"
    },
    {
        "id": 2,
        "title": "Paper Context & Multimodal Problem Formulation",
        "subtitle": "Dual-Stream Deep Learning for Financial Sequences & News Semantics",
        "duration": 75,
        "bullet_points": [
            "Reference Paper: Neural Computing and Applications (DOI: 10.1007/s00521-024-10303-1)",
            "Target Formulation: Binary classification y_t in {0, 1} predicting (Close_t > Close_t-1)",
            "Market Stream: 20-day x 16-feature input tensor processed via Bidirectional LSTM",
            "Linguistic Stream: Daily financial news headlines embedded via FinBERT (768 dimensions)",
            "Joint Projection: Both modalities projected into 64-dimensional latent subspace"
        ],
        "badge": "MATHEMATICAL & ARCHITECTURAL FOUNDATION"
    },
    {
        "id": 3,
        "title": "Data Engineering & Strict Leakage Prevention",
        "subtitle": "Chronological Partitions, Indicator Warmup & Point-in-Time Guarantees",
        "duration": 90,
        "bullet_points": [
            "Strict Chronological Splits: Train (<2023), Validation (2023-mid 2024), Test (mid 2024-2026)",
            "Market Sequence Isolation: 20-day window strictly terminates at trading session t-1",
            "MinMax Scaling Preprocessing: Fitted exclusively on historical training partition",
            "Technical Indicators Warmup: First 33 rows dropped to ensure zero NaNs in EMA/MACD/RSI",
            "Corporate Action Audit: ATVI delisting (Oct 2023) handled via verified historical archive"
        ],
        "badge": "DATA INTEGRITY & REPRODUCIBILITY"
    },
    {
        "id": 4,
        "title": "Model Architecture & Gated Multimodal Unit (GMU)",
        "subtitle": "Combating Linguistic Noise via Dynamic Learned Convex Gating",
        "duration": 105,
        "bullet_points": [
            "Baseline Hierarchy: Training-Majority, Direction Persistence (Lag-1), Logistic Regression",
            "Deep Recurrent Network: 2-Layer Bidirectional LSTM (64/32 units with 0.2 dropout)",
            "Gating Formulation: z = sigmoid(Dense(64)([h_market, h_news]))",
            "Convex Fusion: h_fused = z * h_market_proj + (1 - z) * h_news_proj",
            "Design Rationale: Suppresses news noise during routine periods; elevates high-conviction catalysts"
        ],
        "badge": "DEEP MULTIMODAL FUSION ARCHITECTURE"
    },
    {
        "id": 5,
        "title": "Empirical Benchmark Results & Financial Reality",
        "subtitle": "6,292 Forward Test Observations Across 11 Active Equities (2024–2026)",
        "duration": 105,
        "bullet_points": [
            "Shared Multi-Ticker BiLSTM: 52.13% Accuracy (Highest overall out-of-sample)",
            "Empirical Majority Baseline: 51.21% Accuracy (Reflects market upward drift)",
            "Cross-Asset Regularization: Shared multi-ticker outperforms single-ticker by +3.55%",
            "Accuracy vs. Macro F1 Tradeoff: Linear models achieve higher Macro F1 (0.4704)",
            "Financial Reality: Efficient markets yield 52-54% directional bounds, debunking inflated claims"
        ],
        "badge": "EMPIRICAL OUT-OF-SAMPLE EVALUATION"
    },
    {
        "id": 6,
        "title": "Platform Architecture, FastAPI & React Dashboard",
        "subtitle": "Full-Stack Quantitative AI Research & Inference Demonstration Platform",
        "duration": 90,
        "bullet_points": [
            "Core Library: Modular stock_ai package with 40 passing unit and integration tests",
            "FastAPI REST Service: Endpoints for tickers, historical OHLCV, indicators, news, and inference",
            "Interactive Frontend: Modern dark-theme React/Vite/TypeScript research dashboard",
            "Interactive Views: Stock Explorer (Bollinger/EMA), Model Lab (Ablations), Prediction Demo",
            "Transparent Disclaimers: Point-in-time ground truth display with academic risk warnings"
        ],
        "badge": "PRODUCTION-GRADE FULL-STACK PLATFORM"
    },
    {
        "id": 7,
        "title": "Limitations, Future Research & Conclusion",
        "subtitle": "Toward Point-in-Time Intraday News Streams & Graph Asset Networks",
        "duration": 60,
        "bullet_points": [
            "News Timing Limitation: Daily aggregated news lack intraday timestamps (TIMING-LIMITED)",
            "Next Phase: Microsecond point-in-time SEC Edgar 8-K filings integration",
            "Graph Modeling: Cross-asset attention networks capturing supply chain dependencies",
            "Open Source Repository: https://github.com/Kavix28/Stock-AI.git",
            "Academic Notice: Developed strictly for scientific inquiry; not financial advice"
        ],
        "badge": "RESEARCH OUTLOOK & OPEN SCIENCE"
    }
]


def render_slide(scene, filepath):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor('#070b14')
    ax.set_facecolor('#070b14')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Background card panel
    card = patches.FancyBboxPatch(
        (0.8, 0.8), 14.4, 7.4,
        boxstyle="round,pad=0.2,rounding_size=0.4",
        edgecolor='#6366f1',
        facecolor='#0d1527',
        linewidth=2,
        alpha=0.9
    )
    ax.add_patch(card)

    # Accent top border
    top_bar = patches.Rectangle((0.8, 8.1), 14.4, 0.1, facecolor='#38bdf8', edgecolor='none')
    ax.add_patch(top_bar)

    # Badge
    ax.text(
        1.3, 7.6, f"  {scene['badge']}  ",
        fontsize=11, fontweight='bold', color='#38bdf8',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#1e293b', edgecolor='#38bdf8', alpha=0.9)
    )

    # Chapter index
    ax.text(
        14.5, 7.6, f"SCENE {scene['id']} OF 7",
        fontsize=12, fontweight='bold', color='#94a3b8',
        ha='right'
    )

    # Main Title
    ax.text(
        1.3, 6.7, scene['title'],
        fontsize=24, fontweight='bold', color='#ffffff',
        va='top'
    )

    # Subtitle
    ax.text(
        1.3, 5.9, scene['subtitle'],
        fontsize=15, fontstyle='italic', color='#a5b4fc',
        va='top'
    )

    # Divider line
    ax.plot([1.3, 14.5], [5.5, 5.5], color='#334155', linewidth=1.5)

    # Bullet points
    y_pos = 4.9
    for i, bp in enumerate(scene['bullet_points']):
        # Bullet marker
        ax.plot(1.6, y_pos, marker='o', markersize=7, color='#38bdf8')
        # Bullet text
        ax.text(
            2.0, y_pos, bp,
            fontsize=14, color='#f1f5f9',
            va='center'
        )
        y_pos -= 0.8

    # Footer banner
    ax.text(
        1.3, 1.2, "Stock-AI Research Platform  •  https://github.com/Kavix28/Stock-AI.git",
        fontsize=11, color='#64748b', va='bottom'
    )
    ax.text(
        14.5, 1.2, f"Duration: ~{scene['duration']}s",
        fontsize=11, color='#64748b', ha='right', va='bottom'
    )

    plt.tight_layout()
    plt.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor='none', dpi=120)
    plt.close()
    print(f"Generated slide: {filepath}")


def build_video():
    slide_files = []
    concat_txt_path = os.path.join(ASSETS_DIR, "slides_concat.txt")

    with open(concat_txt_path, "w") as f:
        for s in SCENES:
            slide_path = os.path.join(ASSETS_DIR, f"slide_{s['id']:02d}.png")
            render_slide(s, slide_path)
            slide_files.append(slide_path)
            # Write to concat demuxer format
            # Using forward slashes for ffmpeg path compatibility
            clean_path = slide_path.replace("\\", "/")
            f.write(f"file '{clean_path}'\n")
            f.write(f"duration {s['duration']}\n")
        # ffmpeg concat demuxer requires last file repeated without duration
        clean_last = slide_files[-1].replace("\\", "/")
        f.write(f"file '{clean_last}'\n")

    output_mp4 = os.path.join(OUTPUT_DIR, "Stock_AI_Project_Explained.mp4")

    # Run ffmpeg to compile video
    # Generates a silent MP4 with 1920x1080 resolution, 30fps, H.264 video
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_txt_path,
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
        "-r", "30",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_mp4
    ]

    print("Executing ffmpeg compilation...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        size_mb = os.path.getsize(output_mp4) / (1024 * 1024)
        print(f"Successfully generated: {output_mp4} ({size_mb:.2f} MB)")
    else:
        print(f"FFmpeg error: {res.stderr}")


if __name__ == "__main__":
    build_video()
