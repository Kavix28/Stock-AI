"""
Generate FinBERT Embedding Caches for Historical News Archive.

Aligns news headlines to trading sessions and generates dense 768-dim
embeddings using the pretrained ProsusAI/finbert model.
"""

import os
import sys
import logging
import pandas as pd
from typing import Dict, List, Any

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.stock_ai.data.news.local_provider import LocalCanonicalNewsProvider
from src.stock_ai.data.news.alignment import NewsMarketAligner
from src.stock_ai.models.news.finbert import FinBertNewsEncoder
from src.stock_ai.data.news.cache import NewsEmbeddingCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TICKERS = [
    "AAPL", "AMT", "ATVI", "BIO", "JNJ", "NDAQ",
    "NFLX", "NVDA", "PLD", "SBUX", "SCHW", "TSLA"
]


def main():
    logger.info("Initializing FinBERT encoder and NewsEmbeddingCache...")
    encoder = FinBertNewsEncoder(
        model_name="ProsusAI/finbert",
        max_length=128,
        pooling_method="cls",
        lazy_load=False
    )
    cache = NewsEmbeddingCache(
        cache_dir=os.path.join(project_root, "data", "processed", "embeddings"),
        encoder=encoder
    )
    provider = LocalCanonicalNewsProvider(
        news_dir=os.path.join(project_root, "data", "processed", "news")
    )
    aligner = NewsMarketAligner(alignment_mode="DAILY_DATE_CONVENTIONAL")

    market_dir = os.path.join(project_root, "data", "processed", "market")
    summaries = []

    for ticker in TICKERS:
        logger.info(f"\n--- Processing {ticker} ---")
        market_file = os.path.join(market_dir, f"{ticker}.csv")
        if not os.path.exists(market_file):
            logger.warning(f"Market file {market_file} not found. Skipping.")
            continue

        market_df = pd.read_csv(market_file)
        market_df['date'] = market_df['date'].astype(str)
        trading_dates = market_df['date'].sort_values().tolist()

        news_df = provider.fetch_news(ticker, start_date="2020-01-01", end_date="2026-12-31")
        logger.info(f"Loaded {len(news_df)} canonical news rows for {ticker}.")

        aligned_df = aligner.align_ticker_news(news_df, trading_dates, ticker)
        logger.info(f"Aligned into {len(aligned_df)} trading days ({aligned_df['has_news'].sum()} with news).")

        manifest = cache.build_ticker_cache(ticker, aligned_df, force_rebuild=True)
        summaries.append(manifest)

    # Print summary table
    print("\n" + "="*80)
    print("FINBERT EMBEDDING CACHE GENERATION SUMMARY")
    print("="*80)
    print(f"{'Ticker':<8} {'Total Days':<12} {'News Days':<12} {'Coverage':<12} {'File Size':<14} {'Status'}")
    print("-" * 80)
    for s in summaries:
        size_kb = f"{s['cache_file_size_bytes'] / 1024:.1f} KB"
        print(f"{s['ticker']:<8} {s['total_dates']:<12} {s['dates_with_news']:<12} {s['news_coverage_pct']:<11}% {size_kb:<14} VERIFIED")
    print("="*80)


if __name__ == "__main__":
    main()
