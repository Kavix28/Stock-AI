"""
Technical Indicator Engineering for Stock-AI.

Computes 16 core indicators from raw OHLCV prices:
1. Exponential Moving Average: 12-Day EMA, 26-Day EMA
2. Moving Average Convergence Divergence: MACD, Signal
3. Relative Strength Index: RSI (14-period)
4. Bollinger Bands: Upper Band, Middle Band, Lower Band (20-period, 2-std)
5. Parabolic SAR
6. Stochastic Oscillator: Slow %K, Slow %D
7. Target variables: PrevDayClose, Diff
"""

import numpy as np
import pandas as pd


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Computes standard technical indicators without future lookahead."""
    data = df.copy()
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date').reset_index(drop=True)

    # 1. EMAs
    data['12-Day EMA'] = data['close'].ewm(span=12, adjust=False).mean()
    data['26-Day EMA'] = data['close'].ewm(span=26, adjust=False).mean()

    # 2. MACD
    data['MACD'] = data['12-Day EMA'] - data['26-Day EMA']

    # 3. RSI (14-period Wilder)
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    data['rsi'] = 100 - (100 / (1 + rs))

    # 4. Bollinger Bands (20-period)
    data['Middle Band'] = data['close'].rolling(window=20).mean()
    std = data['close'].rolling(window=20).std()
    data['Upper Band'] = data['Middle Band'] + (2 * std)
    data['Lower Band'] = data['Middle Band'] - (2 * std)

    # 5. Parabolic SAR (Wilder approximation)
    high = data['high'].values
    low = data['low'].values
    close = data['close'].values
    n = len(data)
    sar = np.zeros(n)
    trend = 1
    ep = high[0]
    af = 0.02
    sar[0] = low[0]

    for i in range(1, n):
        sar[i] = sar[i-1] + af * (ep - sar[i-1])
        if trend == 1:
            if low[i] < sar[i]:
                trend = -1
                sar[i] = ep
                ep = low[i]
                af = 0.02
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + 0.02, 0.20)
        else:
            if high[i] > sar[i]:
                trend = 1
                sar[i] = ep
                ep = high[i]
                af = 0.02
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + 0.02, 0.20)

    data['SAR'] = sar

    # 6. Stochastic Oscillator (14-period, 3-period smooth)
    low14 = data['low'].rolling(window=14).min()
    high14 = data['high'].rolling(window=14).max()
    fast_k = 100 * ((data['close'] - low14) / (high14 - low14 + 1e-9))
    data['Slow k'] = fast_k.rolling(window=3).mean()
    data['Slow d'] = data['Slow k'].rolling(window=3).mean()

    # 7. PrevDayClose and Diff (Strict lag-1 definition)
    data['PrevDayClose'] = data['close'].shift(1)
    data['Diff'] = ((data['close'] - data['PrevDayClose']) > 0).astype(int)

    # Warmup cutoff: drop first 33 rows where rolling indicators have NaNs
    return data.iloc[33:].reset_index(drop=True)
