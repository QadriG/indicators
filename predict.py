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
MODELS_DIR = "models_per_coin"
OUTPUT_FILE = "today_live_predictions.csv"
MIN_CONFIDENCE = 0.75
# ==========================================

def fetch_klines(symbol, days=2):
    """Fetch last 48 hours of 1-hour candles."""
    url = "https://api.binance.com/api/v3/klines"
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = end_time - days * 24 * 60 * 60 * 1000
    params = {
        "symbol": symbol,
        "interval": "1h",
        "startTime": start_time,
        "endTime": end_time,
        "limit": 1000
    }
    res = requests.get(url, params=params, timeout=10)
    return res.json()

def engineer_features(df):
    """Compute real technical features."""
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['close'])
    
    close = df['close']
    volume = df['volume']
    
    # Real features (match training)
    df['ema_9'] = talib.EMA(close, timeperiod=9)
    df['rsi'] = talib.RSI(close, timeperiod=14)
    df['volume_ma'] = talib.SMA(volume, timeperiod=20)
    df['volume_ratio'] = volume / df['volume_ma']
    
    return df

def predict_signal(symbol):
    """Generate prediction using real features and trained model."""
    try:
        # Fetch data
        klines = fetch_klines(symbol, days=2)
        if len(klines) < 25:
            return None
            
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'ct', 'qav', 'ntr', 'tbb', 'tbq', 'ignore'
        ])
        df = engineer_features(df)
        if df.empty:
            return None
            
        # Use latest complete candle (not current open)
        latest = df.iloc[-2]
        
        # Load model
        model_path = f"{MODELS_DIR}/{symbol}.pkl"
        if not os.path.exists(model_path):
            return None
            
        model_data = joblib.load(model_path)
        
        # Prepare features for prediction
        X = np.array([[
            latest['ema_9'],
            latest['rsi'],
            latest['volume_ratio']
        ]])
        
        # Predict TP and hold time
        tp_pred = model_data['tp_model'].predict(X)[0]
        hold_pred = model_data['hold_model'].predict(X)[0]
        
        # Confidence based on model certainty (simplified)
        confidence = np.clip(tp_pred / 10.0, 0.6, 0.95)
        
        if confidence >= MIN_CONFIDENCE and tp_pred >= 2.5:
            return {
                'symbol': symbol,
                'prediction_time': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                'predicted_entry_price': round(latest['close'], 6),
                'optimal_tp_pct': round(tp_pred, 2),
                'predicted_exit_price': round(latest['close'] * (1 + tp_pred / 100), 6),
                'min_hold_minutes': max(20, hold_pred * 0.5),
                'max_hold_minutes': min(240, hold_pred * 1.5),
                'confidence': round(confidence, 2)
            }
    except Exception as e:
        print(f"[ERROR] {symbol}: {e}")
    return None

def main():
    print("🔮 PREDICTING TODAY'S HIGH-CONFIDENCE TRADES (REAL FEATURES)")
    signals = []
    
    for symbol in COINS:
        signal = predict_signal(symbol)
        if signal:
            signals.append(signal)
        time.sleep(0.2)  # Rate limit
    
    if signals:
        df = pd.DataFrame(signals)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ Saved {len(signals)} signals to {OUTPUT_FILE}")
        print("\n🎯 TODAY'S PREDICTED TRADES:")
        print(df.to_string(index=False))
    else:
        print("\n⚠️ No high-confidence signals (confidence ≥ 0.75 and TP ≥ 2.5%)")

if __name__ == "__main__":
    main()