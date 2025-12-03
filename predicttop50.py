import os
import requests
import pandas as pd
import numpy as np
import talib
import joblib
from datetime import datetime, timezone

# =============== CONFIG ===============
MODELS_DIR = "top50_bottom_models"
OUTPUT_CSV = "altcoin_signals.csv"

# =============== FEATURE ENGINEERING ===============
def engineer_features(df):
    df = df.sort_values('timestamp').reset_index(drop=True)
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
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
            model = joblib.load(os.path.join(MODELS_DIR, filename))
            
            url = "https://api.binance.com/api/v3/klines"  # Fixed URL (no trailing space!)
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
                
            latest = df.iloc[-1:][features].fillna(0)
            prob = model.predict_proba(latest)[0][1]
            
            # ✅ NO THRESHOLD — log every prediction
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
# =============== MAIN ===============
if __name__ == "__main__":
    print("🔍 Generating live altcoin bottom signals...")
    signals = predict_all_signals()
    
    if signals:
        # Save to CSV with timestamp
        df_signals = pd.DataFrame(signals)
        df_signals.to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)
        print(f"\n✅ {len(signals)} signals saved to {OUTPUT_CSV}")
        print("-" * 60)
        for s in sorted(signals, key=lambda x: x['confidence'], reverse=True):
            print(f"{s['symbol']:10} | Entry: ${s['entry_price']:<10} | Confidence: {s['confidence']:<5}")
    else:
        print("❌ No high-confidence signals found.")