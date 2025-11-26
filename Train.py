import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
import time
import talib
import joblib
import os

# ================== CONFIG ==================
COINS = ["FETUSDT", "FILUSDT", "STRKUSDT", "ICPUSDT", "NEARUSDT", 
         "IMXUSDT", "WLDUSDT", "JTOUSDT", "CELOUSDT", "OSMOUSDT"]
DAYS_BACK = 500
INTERVAL = "1h"
MODELS_DIR = "models_per_coin"
os.makedirs(MODELS_DIR, exist_ok=True)
# ==========================================

def fetch_klines(symbol, days=500):
    url = "https://api.binance.com/api/v3/klines"
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = end_time - days * 24 * 60 * 60 * 1000
    all_klines = []
    current_start = start_time
    while current_start < end_time:
        params = {"symbol": symbol, "interval": INTERVAL, "limit": 1000,
                  "startTime": current_start, "endTime": min(current_start + 1000*3600000, end_time)}
        try:
            res = requests.get(url, params=params, timeout=10)
            if res.status_code == 429: time.sleep(30); continue
            klines = res.json()
            if not klines: break
            all_klines.extend(klines)
            current_start = klines[-1][0] + 1
            time.sleep(0.2)
        except: break
    return all_klines

def engineer_features(df):
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['close'])
    close = df['close']
    df['ema_9'] = talib.EMA(close, 14)
    df['rsi'] = talib.RSI(close, 14)
    df['volume_ma'] = talib.SMA(df['volume'], 20)
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    return df

def create_labels(df):
    df = df.sort_values('timestamp').reset_index(drop=True)
    labels = []
    tps = []
    holds = []
    
    for i in range(len(df)):
        if i + 4 >= len(df):
            labels.append(0); tps.append(0); holds.append(0)
            continue
            
        entry = df.iloc[i]['close']
        future_high = df.iloc[i+1:i+5]['high'].max()
        rally_pct = (future_high - entry) / entry * 100
        
        if rally_pct >= 2.5:
            # Find when rally hit
            exit_idx = None
            for j in range(1, 5):
                if df.iloc[i+j]['high'] >= future_high * 0.99:
                    exit_idx = j
                    break
            hold_min = exit_idx * 60 if exit_idx else 240
            
            labels.append(1)
            tps.append(rally_pct)
            holds.append(hold_min)
        else:
            labels.append(0); tps.append(0); holds.append(0)
            
    df['label'] = labels
    df['optimal_tp'] = tps
    df['hold_minutes'] = holds
    return df

def train_coin_model(symbol):
    print(f"Training model for {symbol}...")
    klines = fetch_klines(symbol, DAYS_BACK)
    df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume',
                                       'ct','qav','ntr','tbb','tbq','ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = engineer_features(df)
    if df.empty: return
    
    df = create_labels(df)
    df = df[df['label'] == 1]  # ONLY train on profitable signals
    
    if len(df) < 100:
        print(f"  ⚠️  Not enough signals for {symbol}")
        return
        
    # Use rally features to predict TP/hold time
    X = df[['ema_9','rsi','volume_ratio']].fillna(0)
    y_tp = df['optimal_tp']
    y_hold = df['hold_minutes']
    
    from sklearn.ensemble import RandomForestRegressor
    tp_model = RandomForestRegressor(n_estimators=100, random_state=42)
    hold_model = RandomForestRegressor(n_estimators=100, random_state=42)
    tp_model.fit(X, y_tp)
    hold_model.fit(X, y_hold)
    
    joblib.dump({
        'tp_model': tp_model,
        'hold_model': hold_model,
        'avg_tp': df['optimal_tp'].mean(),
        'avg_hold': df['hold_minutes'].mean()
    }, f"{MODELS_DIR}/{symbol}.pkl")
    print(f"  ✅ Saved model with {len(df)} signals | Avg TP: {df['optimal_tp'].mean():.1f}%")

def main():
    for symbol in COINS:
        train_coin_model(symbol)
    print("✅ All models trained and saved in 'models_per_coin/'")

if __name__ == "__main__":
    main()