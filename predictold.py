import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
import time
import talib
import joblib
import os

# ================== CONFIG ==================
MODEL_FILE = "crypto_predictor_500d.txt"
OUTPUT_FILE = "today_predicted_trades.csv"
MIN_CONFIDENCE = 0.7  # Only show high-confidence signals

FEATURE_COLS = [
    'ema_9', 'ema_21', 'adx', 'rsi_lag', 'stoch_k', 'stoch_d',
    'macd', 'macd_signal', 'bb_percent_b_lag', 'atr', 'std',
    'obv', 'volume_ratio', 'is_hammer', 'is_morning_star',
    'is_bullish_engulfing', 'hour', 'is_weekend'
]

STABLECOIN_KEYWORDS = ['USDC', 'FDUSD', 'DAI', 'TUSD', 'BUSD', 'USDP', 'UST', 'EURT']
EXCLUDED = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
# ==========================================

def get_top_30_liquid_coins():
    url = "https://api.binance.com/api/v3/ticker/24hr  "
    data = requests.get(url, timeout=10).json()
    usdt_pairs = [
        d for d in data
        if d['symbol'].endswith('USDT')
        and d['symbol'] not in EXCLUDED
        and not any(stable in d['symbol'] for stable in STABLECOIN_KEYWORDS)
    ]
    sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
    return [d['symbol'] for d in sorted_pairs[:30]]

def fetch_klines(symbol, days=3):  # Last 72 hours for feature calculation
    url = "https://api.binance.com/api/v3/klines  "
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

def engineer_features_for_prediction(df):
    """Same features as training, lagged correctly."""
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Trend
    df['ema_9'] = talib.EMA(close, timeperiod=9)
    df['ema_21'] = talib.EMA(close, timeperiod=21)
    df['adx'] = talib.ADX(high, low, close, timeperiod=14)
    
    # Momentum
    df['rsi'] = talib.RSI(close, timeperiod=14)
    stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d
    df['macd'], df['macd_signal'], _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    
    # Volatility
    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    df['bb_percent_b'] = (close - bb_lower) / (bb_upper - bb_lower)
    df['atr'] = talib.ATR(high, low, close, timeperiod=14)
    df['std'] = talib.STDDEV(close, timeperiod=20)
    
    # Volume
    df['obv'] = talib.OBV(close, volume)
    df['volume_ma'] = talib.SMA(volume, timeperiod=20)
    df['volume_ratio'] = volume / df['volume_ma']
    
    # Candlestick
    df['is_hammer'] = (talib.CDLHAMMER(df['open'], high, low, close) == 100).astype(int)
    df['is_morning_star'] = (talib.CDLMORNINGSTAR(df['open'], high, low, close) == 100).astype(int)
    df['is_bullish_engulfing'] = (talib.CDLENGULFING(df['open'], high, low, close) == 100).astype(int)
    
    # Time
    df['hour'] = df['timestamp'].dt.hour
    df['is_weekend'] = (df['timestamp'].dt.weekday >= 5).astype(int)
    
    # Lag features (use previous hour's values)
    df['rsi_lag'] = df['rsi'].shift(1)
    df['bb_percent_b_lag'] = df['bb_percent_b'].shift(1)
    
    return df

def get_hold_time_stats(symbol):
    """Get min/max hold time from your historical signals."""
    # In real use, load from full_dataset or precomputed stats
    # For now, return realistic defaults based on your data
    base_min = 20
    base_max = 120
    return base_min, base_max

def main():
    print("🔮 PREDICTING TODAY'S TRADE SIGNALS")
    
    # Load model
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Model not found: {MODEL_FILE}")
        return
    model = joblib.load(MODEL_FILE)
    
    # Get coins
    coins = get_top_30_liquid_coins()
    signals = []
    now = datetime.now(timezone.utc)
    
    for symbol in coins:
        try:
            klines = fetch_klines(symbol, days=3)
            if len(klines) < 25:
                continue
                
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            df = engineer_features_for_prediction(df)
            if df.empty or len(df) < 2:
                continue
                
            # Use the latest complete candle (not the current incomplete one)
            latest = df.iloc[-2]
            X = pd.DataFrame([{col: latest[col] for col in FEATURE_COLS}])
            X = X.fillna(0)
            
            prob = model.predict_proba(X)[0][1]
            
            # ===== DEBUG: PRINT WHY NO SIGNAL =====
            print(f"[DEBUG] {symbol} | Conf: {prob:.3f} | RSI: {latest['rsi_lag']:.1f} | "
                  f"BB%: {latest['bb_percent_b_lag']:.2f} | "
                  f"Hammer: {latest['is_hammer']} | "
                  f"MorningStar: {latest['is_morning_star']}")
            # =====================================
            
            if prob >= MIN_CONFIDENCE:
                # Entry price: 2% below recent high (last 12 hours)
                recent_high = df['high'].tail(12).max()
                entry_price = recent_high * 0.98
                
                # Optimal TP (scale confidence to realistic TP)
                optimal_tp_pct = np.clip(prob * 8, 2.0, 12.0)
                exit_price = entry_price * (1 + optimal_tp_pct / 100)
                
                # Hold time (min/max)
                min_hold, max_hold = get_hold_time_stats(symbol)
                
                signals.append({
                    'symbol': symbol,
                    'prediction_time': now.strftime('%Y-%m-%d %H:%M:%S'),
                    'predicted_entry_price': round(entry_price, 6),
                    'optimal_tp_pct': round(optimal_tp_pct, 2),
                    'predicted_exit_price': round(exit_price, 6),
                    'min_hold_minutes': min_hold,
                    'max_hold_minutes': max_hold,
                    'confidence': round(prob, 2)
                })
        except Exception as e:
            print(f"[SKIP] {symbol}: {e}")
            continue
    
    if signals:
        df_signals = pd.DataFrame(signals)
        df_signals.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ Saved {len(signals)} signals to {OUTPUT_FILE}")
        print("\n🎯 TODAY'S PREDICTED TRADES:")
        print(df_signals.to_string(index=False))
    else:
        print("\n⚠️ No high-confidence signals for today.")

if __name__ == "__main__":
    main()