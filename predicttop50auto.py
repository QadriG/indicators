import os
import requests
import pandas as pd
import numpy as np
import talib
import joblib
from datetime import datetime, timezone
import time

# =============== CONFIG ===============
MODELS_DIR = "top50_bottom_models"
OUTPUT_CSV = "altcoin_signals.csv"
CHECK_INTERVAL = 3600  # Check every 1 hour (in seconds)

# =============== FEATURE ENGINEERING ===============
def engineer_features(df):
    df = df.sort_values('timestamp').reset_index(drop=True)
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    
    c = df['close']
    h = df['high']
    l = df['low']
    v = df['volume']
    
    df['ema_20'] = talib.EMA(c, 20)
    df['ema_dev'] = (df['ema_20'] - c) / df['ema_20'] * 100
    df['rsi'] = talib.RSI(c, 14)
    upper, _, lower = talib.BBANDS(c, 20, 2, 2)
    df['bb_position'] = (c - lower) / (upper - lower + 1e-8)
    df['volume_sma'] = talib.SMA(v, 20)
    df['volume_ratio'] = v / df['volume_sma'].replace(0, 1)
    df['atr'] = talib.ATR(h, l, c, 14)
    df['volatility'] = df['atr'] / c
    
    o = df['open']
    df['hammer'] = ((h - l) > 3 * (o - c)) & ((c - l) > 2 * (o - c)) & (c > o)
    df['engulfing'] = (c > o) & (c.shift(1) < o.shift(1)) & (c > o.shift(1)) & (o < c.shift(1))
    
    return df

# =============== PREDICTION ===============
def predict_all_signals():
    signals = []
    features = ['rsi','ema_dev','bb_position','volume_ratio','volatility','hammer','engulfing']
    
    for filename in os.listdir(MODELS_DIR):
        if not filename.endswith('.pkl'):
            continue
            
        symbol = filename.replace('_bottom.pkl', '')
        try:
            # Load model
            model_path = os.path.join(MODELS_DIR, filename)
            model = joblib.load(model_path)
            
            # Fetch latest 100 candles
            url = "https://api.binance.com/api/v3/klines"  # ✅ FIXED: no trailing space
            params = {"symbol": symbol, "interval": "1h", "limit": 100}
            klines = requests.get(url, params=params, timeout=10).json()
            
            if not klines or len(klines[0]) < 6:
                continue
                
            df = pd.DataFrame(klines, columns=[
                'timestamp','open','high','low','close','volume',
                'ct','qav','ntr','tbb','tbq','ignore'
            ])
            df = engineer_features(df)
            if len(df) == 0:
                continue
                
            # Get latest features
            latest = df.iloc[-1:][features].fillna(0)
            
            # Predict raw probability (no threshold)
            prob = model.predict_proba(latest)[0][1]
            
            signals.append({
                'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': symbol,
                'entry_price': round(df.iloc[-1]['close'], 6),
                'confidence': round(prob, 3)
            })
        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")
            continue
    
    return signals

# =============== MAIN LOOP ===============
def main():
    print("🔁 Starting continuous altcoin bottom signal monitor...")
    print(f"   Model directory: {MODELS_DIR}")
    print(f"   Output CSV: {OUTPUT_CSV}")
    print(f"   Check interval: {CHECK_INTERVAL} seconds\n")
    
    while True:
        try:
            print(f"🔍 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking for signals...")
            signals = predict_all_signals()
            
            if signals:
                df_signals = pd.DataFrame(signals)
                df_signals.to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)
                print(f"✅ {len(signals)} signals saved to {OUTPUT_CSV}")
                print("-" * 60)
                for s in sorted(signals, key=lambda x: x['confidence'], reverse=True):
                    print(f"{s['symbol']:10} | Entry: ${s['entry_price']:<10} | Confidence: {s['confidence']:<5}")
            else:
                print("   ❌ No signals generated.")
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user.")
            break
        except Exception as e:
            print(f"⚠️  Unexpected error: {e}")
        
        print(f"   Sleeping for {CHECK_INTERVAL} seconds...\n")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()