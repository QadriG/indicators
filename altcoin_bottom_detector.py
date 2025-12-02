import os
import time
import requests
import pandas as pd
import numpy as np
import talib
import joblib
from datetime import datetime, timezone
from lightgbm import LGBMClassifier

# =============== CONFIG ===============
YEARS = 3
INTERVAL = "1h"
RAW_DATA_DIR = "top50_raw_data"
CLEAN_DATA_DIR = "top50_bottom_data"
MODELS_DIR = "top50_bottom_models"

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(CLEAN_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# =============== STEP 1: FETCH RAW OHLCV DATA ===============
def get_top50_coins():
    url = "https://api.binance.com/api/v3/ticker/24hr"
    data = requests.get(url, timeout=10).json()
    coins = []
    for d in data:
        symbol = d['symbol']
        if not symbol.endswith('USDT'): continue
        if any(stable in symbol for stable in ['USDC','BUSD','DAI','TUSD']): continue
        if float(d['quoteVolume']) < 2_000_000: continue
        coins.append(symbol)
    return sorted(coins, key=lambda s: next(d['quoteVolume'] for d in data if d['symbol']==s), reverse=True)[:50]

def fetch_klines(symbol, days=365*3):
    url = "https://api.binance.com/api/v3/klines"
    end = int(datetime.now(timezone.utc).timestamp() * 1000)
    start = end - days * 86400000
    all_k = []
    current = start
    while current < end:
        params = {"symbol": symbol, "interval": INTERVAL, "startTime": current, "limit": 1000}
        try:
            res = requests.get(url, params=params, timeout=10)
            if res.status_code == 429: time.sleep(30); continue
            klines = res.json()
            if not klines: break
            all_k.extend(klines)
            current = klines[-1][0] + 1
            time.sleep(0.1)
        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")
            break
    return all_k

def save_raw_data():
    print("📥 Fetching 3 years of 1h OHLCV data for top 50 coins...")
    coins = get_top50_coins()
    for symbol in coins:
        print(f"  {symbol}")
        klines = fetch_klines(symbol)
        if len(klines) < 25000:
            continue
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume',
            'ct','qav','ntr','tbb','tbq','ignore'
        ])
        df.to_parquet(f"{RAW_DATA_DIR}/{symbol}.parquet", index=False)
    print(f"✅ Raw data saved to {RAW_DATA_DIR}/")

# =============== STEP 2: CLEAN + LABEL FOR TRUE BOTTOMS ===============
def engineer_features(df):
    df = df.sort_values('timestamp').reset_index(drop=True)
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    
    df['rsi'] = talib.RSI(c, 14)
    df['ema_20'] = talib.EMA(c, 20)
    df['ema_50'] = talib.EMA(c, 50)
    df['ema_dev'] = (c - df['ema_20']) / df['ema_20'] * 100
    
    upper, middle, lower = talib.BBANDS(c, 20, 2, 2)
    df['bb_width'] = (upper - lower) / middle
    df['bb_position'] = (c - lower) / (upper - lower + 1e-8)
    
    df['volume_ma'] = talib.SMA(v, 20)
    df['volume_ratio'] = v / df['volume_ma'].replace(0, 1)
    
    df['atr'] = talib.ATR(h, l, c, 14)
    df['volatility'] = df['atr'] / c
    
    o = df['open']
    df['hammer'] = ((h - l) > 3 * (o - c)) & ((c - l) > 2 * (o - c)) & (c > o)
    df['engulfing'] = (c > o) & (c.shift(1) < o.shift(1)) & (c > o.shift(1)) & (o < c.shift(1))
    
    return df

def label_true_bottoms(df):
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['low_7d'] = df['low'].rolling(window=7*24, min_periods=1).min()
    df['future_low_14d'] = df['low'].shift(-1).rolling(window=14*24, min_periods=1).min()
    
    df['label'] = (
        (df['low'] <= df['low_7d']) & 
        (df['future_low_14d'] >= df['low'] * 0.98)
    ).astype(int)
    
    return df[['timestamp','open','high','low','close','volume',
               'rsi','ema_dev','bb_position','volume_ratio','volatility',
               'hammer','engulfing','label']].dropna()

def clean_and_label():
    print("🧹 Cleaning and labeling data for true bottoms...")
    for file in os.listdir(RAW_DATA_DIR):
        if not file.endswith('.parquet'): continue
        symbol = file.replace('.parquet', '')
        try:
            df = pd.read_parquet(f"{RAW_DATA_DIR}/{file}")
            df = engineer_features(df)
            df = label_true_bottoms(df)
            if df['label'].sum() < 100:
                print(f"  ⚠️ {symbol}: only {df['label'].sum()} bottoms")
                continue
            df.to_parquet(f"{CLEAN_DATA_DIR}/{symbol}.parquet", index=False)
            print(f"  ✅ {symbol}: {len(df)} samples, {df['label'].sum()} bottoms")
        except Exception as e:
            print(f"❌ {symbol}: {e}")
    print(f"✅ Clean data saved to {CLEAN_DATA_DIR}/")

# =============== STEP 3: TRAIN PER-COIN MODELS (WITH VALIDATION) ===============
from sklearn.model_selection import cross_val_score

def train_models():
    print("🧠 Training per-coin bottom-detection models with validation...")
    features = ['rsi','ema_dev','bb_position','volume_ratio','volatility','hammer','engulfing']
    for file in os.listdir(CLEAN_DATA_DIR):
        if not file.endswith('.parquet'): continue
        symbol = file.replace('.parquet', '')
        try:
            df = pd.read_parquet(f"{CLEAN_DATA_DIR}/{file}")
            X = df[features].fillna(0)
            y = df['label']
            if y.sum() < 100 or (y == 0).sum() < 100:  # Need both classes
                print(f"  ⚠️ {symbol}: insufficient class balance")
                continue
            
            # === ADD VALIDATION HERE ===
            # Quick 3-fold CV to check if model has signal
            model_for_cv = LGBMClassifier(
                n_estimators=100,  # lighter for CV
                max_depth=5,
                learning_rate=0.05,
                min_data_in_leaf=50,
                feature_fraction=0.8,
                class_weight='balanced',
                random_state=42,
                verbose=-1
            )
            scores = cross_val_score(model_for_cv, X, y, cv=3, scoring='roc_auc')
            avg_auc = scores.mean()
            
            if avg_auc < 0.55:
                print(f"  ⚠️ {symbol}: LOW SIGNAL (AUC={avg_auc:.3f}) — skipping")
                continue
            
            # Train final model
            final_model = LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                min_data_in_leaf=50,
                feature_fraction=0.8,
                class_weight='balanced',
                random_state=42,
                verbose=-1
            )
            final_model.fit(X, y)
            
            joblib.dump(final_model, f"{MODELS_DIR}/{symbol}_bottom.pkl")
            print(f"✅ {symbol}: AUC={avg_auc:.3f} | samples={len(df)} | bottoms={y.sum()}")
        except Exception as e:
            print(f"❌ {symbol}: {e}")
    print(f"✅ Models saved to {MODELS_DIR}/")

# =============== STEP 4: PREDICTION — LIVE ENTRY SIGNALS ===============
def predict_entry(symbol):
    try:
        model_path = f"{MODELS_DIR}/{symbol}_bottom.pkl"
        if not os.path.exists(model_path):
            return None
        model = joblib.load(model_path)
        
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": "1h", "limit": 100}
        klines = requests.get(url, params=params, timeout=10).json()
        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume',
            'ct','qav','ntr','tbb','tbq','ignore'
        ])
        df = engineer_features(df)
        latest = df.iloc[-1:][[
            'rsi','ema_dev','bb_position','volume_ratio','volatility','hammer','engulfing'
        ]].fillna(0)
        
        prob = model.predict_proba(latest)[0][1]
        if prob >= 0.70:
            entry = df.iloc[-1]['close']
            return {
                'symbol': symbol,
                'entry_price': round(entry, 6),
                'confidence': round(prob, 3)
            }
    except Exception as e:
        print(f"[PRED ERROR] {symbol}: {e}")
    return None

def generate_signals():
    print("\n🔍 LIVE ENTRY SIGNALS (High-Confidence Bottoms)")
    print("-" * 60)
    signals = []
    for file in os.listdir(MODELS_DIR):
        if not file.endswith('.pkl'): continue
        symbol = file.replace('_bottom.pkl', '')
        signal = predict_entry(symbol)
        if signal:
            signals.append(signal)
    
    if signals:
        for s in sorted(signals, key=lambda x: x['confidence'], reverse=True):
            print(f"{s['symbol']:10} | Entry: ${s['entry_price']:<10} | Confidence: {s['confidence']:<5}")
    else:
        print("❌ No high-confidence bottom signals right now.")

# =============== MAIN ===============
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python altcoin_bottom_detector.py [fetch|label|train|predict]")
        exit(1)
    
    mode = sys.argv[1].lower()
    if mode == "fetch":
        save_raw_data()
    elif mode == "label":
        clean_and_label()
    elif mode == "train":
        train_models()
    elif mode == "predict":
        generate_signals()
    else:
        print("Invalid mode. Use: fetch, label, train, or predict")