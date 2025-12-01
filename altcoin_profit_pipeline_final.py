import requests
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timezone
import time
import os

# ================= CONFIG ==================
YEARS_BACK = 3
OUTPUT_DIR = "top50_data_3y"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES_1H = [
    'rsi_14', 'macd', 'macd_signal', 'bb_width', 'bb_position',
    'volume_sma_ratio', 'close_to_high', 'body_size', 'wick_ratio',
    'engulfing', 'hammer', 'shooting_star', 'doji'
]
# ==========================================

def get_top50_coins():
    url = "https://api.binance.com/api/v3/ticker/24hr"  # FIXED: no trailing space
    data = requests.get(url, timeout=10).json()
    coins = []
    for d in data:
        symbol = d['symbol']
        if not symbol.endswith('USDT'): continue
        if any(stable in symbol for stable in ['USDC','BUSD','DAI','TUSD']): continue
        if float(d['quoteVolume']) < 5_000_000: continue
        coins.append((symbol, float(d['quoteVolume'])))
    coins.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, vol in coins[:50]]

def fetch_klines(symbol, interval, days=365*3):
    url = "https://api.binance.com/api/v3/klines"  # FIXED: no trailing space
    end = int(datetime.now(timezone.utc).timestamp() * 1000)
    start = end - days * 86400000
    all_k = []
    current = start
    while current < end:
        params = {"symbol": symbol, "interval": interval, "startTime": current, "limit": 1000}
        try:
            res = requests.get(url, params=params, timeout=10)
            if res.status_code == 429: time.sleep(30); continue
            klines = res.json()
            if not klines: break
            all_k.extend(klines)
            current = klines[-1][0] + 1
            time.sleep(0.1)
        except Exception as e:
            print(f"[ERROR] {symbol} {interval}: {e}")
            break
    return all_k

def add_candlestick_patterns(df):
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    df['engulfing'] = ((c.shift(1) < o.shift(1)) & (c > o) & (c > o.shift(1)) & (o < c.shift(1))).astype(int)
    df['hammer'] = ((h - l) > 3 * (o - c)) & ((c - l) > 2 * (o - c)) & (c > o).astype(int)
    df['shooting_star'] = ((h - l) > 3 * (c - o)) & ((h - c) > 2 * (c - o)) & (c < o).astype(int)
    df['doji'] = (abs(c - o) < 0.1 * (h - l)).astype(int)
    return df

def engineer_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    
    df['rsi_14'] = ta.momentum.RSIIndicator(close=c, window=14).rsi()
    macd = ta.trend.MACD(close=c, window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(close=c, window=20, window_dev=2)
    df['bb_width'] = bb.bollinger_wband()
    df['bb_position'] = (c - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-8)
    
    # FIXED: Use pandas rolling instead of non-existent ta.volume.VolumeSMAIndicator
    df['volume_sma'] = v.rolling(window=20, min_periods=1).mean()
    df['volume_sma_ratio'] = v / df['volume_sma'].replace(0, 1)
    
    df['close_to_high'] = (h - c) / (h - l + 1e-8)
    df['body_size'] = abs(c - df['open']) / c
    df['wick_ratio'] = (h - l) / c
    
    df = add_candlestick_patterns(df)
    return df

def label_profit_signals(df_1h):
    df = df_1h.sort_values('timestamp').reset_index(drop=True)
    n = len(df)
    lookahead = 12
    if n <= lookahead: return pd.DataFrame()
    labels = []
    tp_reached = []
    hold_time = []
    
    for i in range(n - lookahead):
        entry = df.iloc[i]['close']
        effective_entry = entry * 1.002
        future_highs = df.iloc[i+1:i+1+lookahead]['high'].values
        max_rally = (future_highs.max() / effective_entry - 1) * 100
        
        if max_rally >= 2.0:
            labels.append(1)
            tp_reached.append(max_rally)
            hit_idx = np.where((future_highs / effective_entry - 1) * 100 >= 2.0)[0]
            hold_time.append(hit_idx[0] + 1 if len(hit_idx) > 0 else lookahead)
        else:
            labels.append(0)
            tp_reached.append(0.0)
            hold_time.append(0)
    
    if not labels: return pd.DataFrame()
    df = df.iloc[:len(labels)].copy()
    df['label'] = labels
    df['max_tp'] = tp_reached
    df['hold_hours'] = hold_time
    return df[df['label'] == 1]

def resample_to_interval(df, interval):
    df = df.set_index('timestamp')
    rule = '2H' if interval == '2h' else '4H'
    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    return engineer_features(resampled)

def fetch_coin_data(symbol):
    print(f"  {symbol}")
    klines_1h = fetch_klines(symbol, "1h", 365*3)
    if len(klines_1h) < 10000: return
    
    df_1h = pd.DataFrame(klines_1h, columns=[
        'timestamp','open','high','low','close','volume',
        'ct','qav','ntr','tbb','tbq','ignore'
    ])
    df_1h = engineer_features(df_1h)
    if len(df_1h) < 8000: return
    
    df_labeled = label_profit_signals(df_1h)
    if len(df_labeled) < 100: return
    
    df_2h = resample_to_interval(df_1h.copy(), "2h")
    df_4h = resample_to_interval(df_1h.copy(), "4h")
    
    df_final = df_labeled.copy()
    for df_other, suffix in [(df_2h, "_2h"), (df_4h, "_4h")]:
        if len(df_other) == 0: continue
        df_other = df_other[['timestamp'] + FEATURES_1H].copy()
        df_other.columns = ['timestamp'] + [f"{col}{suffix}" for col in FEATURES_1H]
        df_final = pd.merge_asof(
            df_final.sort_values('timestamp'),
            df_other.sort_values('timestamp'),
            on='timestamp',
            direction='backward'
        )
    
    df_final.to_parquet(f"{OUTPUT_DIR}/{symbol}.parquet", index=False)
    print(f"    ✅ {len(df_final)} setups")

def main():
    print("🚀 Fetching 3-year multi-timeframe data for top 50 coins...")
    coins = get_top50_coins()
    print(f"Found {len(coins)} coins. Processing...")
    for i, symbol in enumerate(coins, 1):
        print(f"[{i}/{len(coins)}] {symbol}")
        fetch_coin_data(symbol)
    print(f"\n✅ Data saved to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()