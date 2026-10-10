"""
Module to refresh historical market data and audit financial news for all 12 tickers.
Uses yfinance with fallback for delisted tickers (ATVI).
Computes 16 core market features with strict mathematical definitions.
Stores all refreshed data in data/stock_data_latest/ and data/news_latest/.
"""

import os
import sys
import glob
import time
import json
import argparse
import numpy as np
import pandas as pd
import yfinance as yf

# All 12 discovered tickers and metadata
ALL_TICKERS = [
    ("AAPL", "Apple Inc.", False),
    ("AMT", "American Tower Corporation", False),
    ("ATVI", "Activision Blizzard, Inc.", True),  # Delisted Oct 2023
    ("BIO", "Bio-Rad Laboratories, Inc.", False),
    ("JNJ", "Johnson & Johnson", False),
    ("NDAQ", "Nasdaq, Inc.", False),
    ("NFLX", "Netflix, Inc.", False),
    ("NVDA", "NVIDIA Corporation", False),
    ("PLD", "Prologis, Inc.", False),
    ("SBUX", "Starbucks Corporation", False),
    ("SCHW", "The Charles Schwab Corporation", False),
    ("TSLA", "Tesla, Inc.", False)
]

def calculate_parabolic_sar(high, low, close, af_step=0.02, af_max=0.20):
    """Calculates Parabolic SAR iteratively."""
    sar = np.zeros(len(close))
    trend = 1  # 1 for uptrend, -1 for downtrend
    ep = high[0]
    af = af_step
    sar[0] = low[0]

    for i in range(1, len(close)):
        prev_sar = sar[i - 1]
        if trend == 1:
            curr_sar = prev_sar + af * (ep - prev_sar)
            curr_sar = min(curr_sar, low[max(0, i - 1)], low[max(0, i - 2)])
            if low[i] < curr_sar:
                trend = -1
                curr_sar = ep
                ep = low[i]
                af = af_step
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            curr_sar = prev_sar + af * (ep - prev_sar)
            curr_sar = max(curr_sar, high[max(0, i - 1)], high[max(0, i - 2)])
            if high[i] > curr_sar:
                trend = 1
                curr_sar = ep
                ep = high[i]
                af = af_step
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)
        sar[i] = curr_sar

    return sar

def compute_technical_indicators(df):
    """
    Computes all 16 core market features matching the research paper:
    high, close, open, volume, low, adjclose,
    rsi, 12-Day EMA, 26-Day EMA, MACD, SAR,
    Upper Band, Middle Band, Lower Band, Slow k, Slow d
    """
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']

    # 1. EMAs and MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['12-Day EMA'] = ema12
    df['26-Day EMA'] = ema26
    df['MACD'] = ema12 - ema26

    # 2. Bollinger Bands (5-day rolling, 2 population std)
    mid5 = close.rolling(5).mean()
    std5_pop = close.rolling(5).std(ddof=0)
    df['Middle Band'] = mid5
    df['Upper Band'] = mid5 + 2 * std5_pop
    df['Lower Band'] = mid5 - 2 * std5_pop

    # 3. RSI (14-day Wilder RSI)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100.0 - (100.0 / (1.0 + rs))

    # 4. Stochastic Oscillator (14-day Fast %K, 3-day Slow %K, 3-day Slow %D)
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    fast_k = 100.0 * ((close - low14) / (high14 - low14 + 1e-10))
    slow_k = fast_k.rolling(3).mean()
    slow_d = slow_k.rolling(3).mean()
    df['Slow k'] = slow_k
    df['Slow d'] = slow_d

    # 5. Parabolic SAR
    sar = calculate_parabolic_sar(high.values, low.values, close.values)
    df['SAR'] = sar

    # 6. Targets and Alignment Columns
    df['PrevDayClose'] = close.shift(1)
    df['Diff'] = ((close - df['PrevDayClose']) > 0).astype(int)

    return df

def refresh_single_ticker(ticker, company, is_delisted, stock_out_dir, news_out_dir, max_retries=3):
    print(f"[*] Processing {ticker} ({company})...")
    retrieval_time = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S UTC')
    
    # 1. Download or Fallback for Stock Data
    stock_df = None
    source = "Yahoo Finance (yfinance)"
    status = "SUCCESS"
    notes = ""

    if is_delisted:
        print(f"    [-] {ticker} is delisted. Using verified historical archive fallback...")
        historical_file = f"data/stock_data/{ticker.lower()}_Simon/stocks_ts_{ticker}_2021-1-4_2022-9-20.csv"
        if os.path.exists(historical_file):
            raw_hist = pd.read_csv(historical_file)
            raw_hist['date'] = pd.to_datetime(raw_hist['date'], errors='coerce').dt.strftime('%Y-%m-%d')
            # Drop warmup rows (first 33 rows contain NaNs in technical indicators)
            core_cols = [
                'high', 'close', 'open', 'volume', 'low', 'adjclose',
                'rsi', '12-Day EMA', '26-Day EMA', 'MACD', 'SAR',
                'Upper Band', 'Middle Band', 'Lower Band', 'Slow k', 'Slow d'
            ]
            stock_df = raw_hist.dropna(subset=core_cols).sort_values('date').reset_index(drop=True)
            if 'PrevDayClose' not in stock_df.columns:
                stock_df['PrevDayClose'] = stock_df['close'].shift(1)
            if 'Diff' not in stock_df.columns:
                stock_df['Diff'] = ((stock_df['close'] - stock_df['PrevDayClose']) > 0).astype(int)
            source = "Historical Archive Fallback (Delisted Oct 2023)"
            notes = "Delisted Oct 2023; preserved historical archive coverage without warmup NaNs."
        else:
            status = "FAILED: Delisted and no archive found"
    else:
        for attempt in range(1, max_retries + 1):
            try:
                yf_ticker = yf.Ticker(ticker)
                hist = yf_ticker.history(start="2010-01-01", auto_adjust=False)
                if len(hist) > 0:
                    hist = hist.reset_index()
                    hist['date'] = pd.to_datetime(hist['Date']).dt.strftime('%Y-%m-%d')
                    hist = hist.rename(columns={
                        'Open': 'open', 'High': 'high', 'Low': 'low',
                        'Close': 'close', 'Adj Close': 'adjclose', 'Volume': 'volume'
                    })
                    hist['symbol'] = ticker
                    hist = hist.sort_values('date').drop_duplicates(subset=['date']).reset_index(drop=True)
                    
                    # Sanity Validations
                    assert (hist['high'] >= hist['low']).all(), "High < Low violation!"
                    assert (hist['high'] >= hist['close'] - 1e-3).all(), "High < Close violation!"
                    assert (hist['low'] <= hist['close'] + 1e-3).all(), "Low > Close violation!"
                    assert (hist['close'] > 0).all(), "Zero or negative prices!"
                    assert (hist['volume'] >= 0).all(), "Negative volume!"
                    
                    # Compute Indicators
                    stock_df = compute_technical_indicators(hist)
                    # Drop warmup period (first 33 rows)
                    stock_df = stock_df.iloc[33:].reset_index(drop=True)
                    notes = "Live refreshed via yfinance through latest trading session."
                    break
                else:
                    time.sleep(2)
            except Exception as e:
                print(f"    [!] Attempt {attempt} failed: {e}")
                time.sleep(2)
        if stock_df is None:
            status = "FAILED: Download error"
            notes = "Could not retrieve data via yfinance."

    # Save Refreshed Stock Data
    stock_path = os.path.join(stock_out_dir, f"{ticker}.csv")
    stock_rows = len(stock_df) if stock_df is not None else 0
    start_date = stock_df['date'].iloc[0] if stock_df is not None and stock_rows > 0 else "N/A"
    end_date = stock_df['date'].iloc[-1] if stock_df is not None and stock_rows > 0 else "N/A"

    if stock_df is not None:
        stock_df.to_csv(stock_path, index=False)
        print(f"    [+] Saved {ticker} stock data ({stock_rows} rows, {start_date} to {end_date}) -> {stock_path}")

    # 2. Process News Data
    news_rows = 0
    news_start = "N/A"
    news_end = "N/A"
    news_hist_file = f"data/stock_data/{ticker.lower()}_Simon/Single_newsheadline_day_{ticker.lower()}.csv"
    news_status = "TIMING-LIMITED"
    news_notes = "Daily calendar date only; lacks intraday publication timestamps (HH:MM:SS)."

    if os.path.exists(news_hist_file):
        ndf = pd.read_csv(news_hist_file)
        ndf['date'] = pd.to_datetime(ndf['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        ndf = ndf.dropna(subset=['date', 'Titles']).sort_values('date').drop_duplicates(subset=['date']).reset_index(drop=True)
        news_rows = len(ndf)
        if news_rows > 0:
            news_start = ndf['date'].iloc[0]
            news_end = ndf['date'].iloc[-1]
            news_path = os.path.join(news_out_dir, f"{ticker}.csv")
            ndf.to_csv(news_path, index=False)
            print(f"    [+] Saved {ticker} news headlines ({news_rows} rows, {news_start} to {news_end}) -> {news_path}")

    market_record = {
        "ticker": ticker,
        "company_name": company,
        "source": source,
        "retrieval_timestamp": retrieval_time,
        "start_date": start_date,
        "end_date": end_date,
        "row_count": stock_rows,
        "status": status,
        "notes": notes
    }

    news_record = {
        "ticker": ticker,
        "company_name": company,
        "source": "Historical News Headline Archive",
        "retrieval_timestamp": retrieval_time,
        "start_date": news_start,
        "end_date": news_end,
        "row_count": news_rows,
        "timestamp_granularity": "Daily (Date only, missing HH:MM:SS)",
        "timing_limitation": news_status,
        "notes": news_notes
    }

    return market_record, news_record

def main():
    parser = argparse.ArgumentParser(description="Refresh all 12 stock datasets and news feeds")
    parser.add_argument("--stock-dir", default="data/stock_data_latest", help="Output dir for refreshed stock CSVs")
    parser.add_argument("--news-dir", default="data/news_latest", help="Output dir for refreshed news CSVs")
    parser.add_argument("--results-dir", default="results/latest", help="Output dir for manifests")
    args = parser.parse_args()

    os.makedirs(args.stock_dir, exist_ok=True)
    os.makedirs(args.news_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print("================================================================================")
    print("      REFRESHING AND AUDITING ALL 12 TICKER DATASETS & NEWS FEEDS               ")
    print("================================================================================")

    market_manifest_rows = []
    news_manifest_rows = []

    for ticker, company, is_delisted in ALL_TICKERS:
        m_rec, n_rec = refresh_single_ticker(ticker, company, is_delisted, args.stock_dir, args.news_dir)
        market_manifest_rows.append(m_rec)
        news_manifest_rows.append(n_rec)

    # Save Manifests
    m_df = pd.DataFrame(market_manifest_rows)
    m_path = os.path.join(args.results_dir, "market_data_manifest.csv")
    m_df.to_csv(m_path, index=False)
    print(f"\n[+] Saved Market Data Manifest -> {m_path}")

    n_df = pd.DataFrame(news_manifest_rows)
    n_path = os.path.join(args.results_dir, "news_coverage_manifest.csv")
    n_df.to_csv(n_path, index=False)
    print(f"[+] Saved News Coverage Manifest -> {n_path}")

    print("\nMarket Data Manifest Summary:")
    print(m_df[["ticker", "start_date", "end_date", "row_count", "status"]].to_string(index=False))

    print("\nNews Coverage Manifest Summary:")
    print(n_df[["ticker", "start_date", "end_date", "row_count", "timing_limitation"]].to_string(index=False))
    print("================================================================================\n")

if __name__ == "__main__":
    main()
