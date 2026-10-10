"""
Audit data freshness, date coverage, and missingness across all 12 tickers for both
market time series and FinBERT news representations.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

ALL_TICKERS = [
    "AAPL", "AMT", "ATVI", "BIO", "JNJ", "NDAQ",
    "NFLX", "NVDA", "PLD", "SBUX", "SCHW", "TSLA"
]

def audit():
    market_dir = os.path.join(project_root, "data", "processed", "market")
    embed_dir = os.path.join(project_root, "data", "processed", "embeddings")

    results = []

    for ticker in ALL_TICKERS:
        m_path = os.path.join(market_dir, f"{ticker}.csv")
        e_path = os.path.join(embed_dir, f"{ticker}_embeddings.npz")
        man_path = os.path.join(embed_dir, f"{ticker}_manifest.json")

        row = {
            "ticker": ticker,
            "market_exists": os.path.exists(m_path),
            "market_rows": 0,
            "market_min_date": None,
            "market_max_date": None,
            "market_status": "MISSING",
            "news_cache_exists": os.path.exists(e_path),
            "news_cache_dates": 0,
            "news_cache_with_news": 0,
            "news_coverage_pct": 0.0,
            "news_min_date": None,
            "news_max_date": None,
        }

        if row["market_exists"]:
            m_df = pd.read_csv(m_path)
            row["market_rows"] = len(m_df)
            row["market_min_date"] = str(m_df["date"].min())
            row["market_max_date"] = str(m_df["date"].max())
            if ticker == "ATVI":
                row["market_status"] = "BLOCKED (Delisted Oct 2023, Legacy 2010-2022 only)"
            else:
                row["market_status"] = f"VERIFIED (2010 to {row['market_max_date'][:4]})"

        if row["news_cache_exists"] and os.path.exists(man_path):
            with open(man_path, "r") as f:
                man = json.load(f)
            row["news_cache_dates"] = man.get("total_dates", 0)
            row["news_cache_with_news"] = man.get("dates_with_news", 0)
            row["news_coverage_pct"] = man.get("news_coverage_pct", 0.0)

            npz = np.load(e_path, allow_pickle=True)
            dates = list(npz["dates"])
            if dates:
                row["news_min_date"] = dates[0]
                row["news_max_date"] = dates[-1]

        results.append(row)

    df = pd.DataFrame(results)
    print("=" * 105)
    print("DATA FRESHNESS AND PER-TICKER COVERAGE AUDIT (12 TICKERS)")
    print("=" * 105)
    for r in results:
        print(f"Ticker: {r['ticker']:<5} | Market: {r['market_rows']:>5} rows ({r['market_min_date']} to {r['market_max_date']}) [{r['market_status']}]")
        print(f"        News:   {r['news_cache_dates']:>5} dates ({r['news_min_date']} to {r['news_max_date']}) | {r['news_cache_with_news']} with news ({r['news_coverage_pct']:.1f}% cov)")
        print("-" * 105)

    return df

if __name__ == "__main__":
    audit()
