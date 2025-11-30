import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
import talib
import os

# ================== CONFIG ==================
DAYS_BACK = 500
INTERVAL = "1h"
OUTPUT_DIR = "altcoin_data_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_10 = ["FETUSDT", "FILUSDT", "STRKUSDT", "ICPUSDT", "NEARUSDT", 
          "IMXUSDT", "WLDUSDT", "JTOUSDT", "CELOUSDT", "OSMOUSDT"]
STABLES = ['USDC', 'FDUSD', 'DAI', 'TUSD', 'BUSD', 'USDP', 'UST', 'EURT']
EXCLUDED = TOP_10 + ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
# ==========================================

def get_altcoin_watchlist():
    print("📡 Fetching altcoin watchlist...")
    url = "https://api.binance.com/api/v3/ticker/24hr"
    data = requests.get(url, timeout=10).json()
    
    altcoins = []
    for d in data:
        if not d['symbol'].endswith('USDT'): continue
        if d['symbol'] in EXCLUDED: continue
        if any(stable in d['symbol'] for stable in STABLES): continue
            
        volume = float(d['quoteVolume'])
        est_mc = volume * 50
        if volume < 1_000_000 or est_mc > 500_000_000: continue
            
        altcoins.append((d['symbol'], volume))
    
    altcoins.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, vol in altcoins[:30]]

def fetch_klines(symbol, days=500):
    url = "https://api.binance.com/api/v3/klines"
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = end_time - days * 24 * 60 * 60 * 1000
    all_klines = []
    current_start = start_time

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": current_start,
            "endTime": min(current_start + 1000 * 3600000, end_time),
            "limit": 1000
        }
        try:
            res = requests.get(url, params=params, timeout=10)
            if res.status_code == 429: time.sleep(30); continue
            klines = res.json()
            if not klines: break
            all_klines.extend(klines)
            current_start = klines[-1][0] + 1
            time.sleep(0.2)
        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")
            break
    return all_klines

def engineer_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Trend
    df['ema_20'] = talib.EMA(close, 20)
    df['ema_50'] = talib.EMA(close, 50)
    df['ema_ratio'] = df['ema_20'] / df['ema_50']
    
    # Momentum
    df['rsi'] = talib.RSI(close, 14)
    df['rsi_ma'] = talib.SMA(df['rsi'], 6)
    df['rsi_diff'] = df['rsi'] - df['rsi_ma']
    
    # Volatility
    upper, middle, lower = talib.BBANDS(close, 20, 2, 2)
    df['bb_width'] = (upper - lower) / middle
    df['bb_position'] = (close - lower) / (upper - lower)
    df['atr'] = talib.ATR(high, low, close, 14)
    df['volatility'] = df['atr'] / close
    
    # Volume
    df['volume_ma'] = talib.SMA(volume, 20)
    df['volume_ratio'] = volume / df['volume_ma'].replace(0, np.nan)
    df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
    
    # Price action
    df['high_low_ratio'] = (high - low) / close
    df['body_size'] = abs(close - df['open']) / close
    
    return df

def create_labels(df):
    df = df.sort_values('timestamp').reset_index(drop=True)
    labels = []
    
    for i in range(len(df)):
        if i < 12:  # Need history
            labels.append(0)
            continue
            
        # ONLY use past data to decide if it's a buy signal
        current_price = df.iloc[i]['close']
        recent_high = df.iloc[i-12:i+1]['high'].max()
        drop_pct = (recent_high - current_price) / recent_high * 100
        
        # Volume confirmation
        volume_ratio = df.iloc[i]['volume'] / df.iloc[i]['volume_ma']
        
        # Label = 1 if 2% dip + volume spike (ACTIONABLE setup)
        if drop_pct >= 2.0 and volume_ratio >= 1.5:
            labels.append(1)
        else:
            labels.append(0)
            
    df['label'] = labels
    return df

def main():
    print("🚀 FETCHING ALTCOIN DATA V3 (REALISTIC LABELS)")
    coins = get_altcoin_watchlist()
    print(f"Found {len(coins)} altcoins. Processing...")
    
    for i, symbol in enumerate(coins, 1):
        print(f"[{i}/{len(coins)}] Processing {symbol}...")
        klines = fetch_klines(symbol, DAYS_BACK)
        if len(klines) < 1000:
            continue
            
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'ct', 'qav', 'ntr', 'tbb', 'tbq', 'ignore'
        ])
        df = engineer_features(df)
        if df.empty:
            continue
            
        df = create_labels(df)
        df.to_parquet(f"{OUTPUT_DIR}/{symbol}.parquet", index=False)
    
    print(f"\n✅ Data saved to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()