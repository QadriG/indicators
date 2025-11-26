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
OUTPUT_DIR = "training_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ==========================================

def get_top_30_liquid_coins():
    """Fetch top 30 USDT pairs, excluding BTC/ETH/BNB and stablecoins."""
    print("📡 Fetching top 30 liquid coins...")
    url = "https://api.binance.com/api/v3/ticker/24hr"
    data = requests.get(url, timeout=10).json()
    
    stablecoin_keywords = ['USDC', 'FDUSD', 'DAI', 'TUSD', 'BUSD', 'USDP', 'UST', 'EURT']
    excluded = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    usdt_pairs = [
        d for d in data
        if d['symbol'].endswith('USDT')
        and d['symbol'] not in excluded
        and not any(stable in d['symbol'] for stable in stablecoin_keywords)
    ]
    
    sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
    return [d['symbol'] for d in sorted_pairs[:30]]

def fetch_klines(symbol, days=500, interval="1h"):
    """Fetch 1-hour klines."""
    url = "https://api.binance.com/api/v3/klines"
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = end_time - days * 24 * 60 * 60 * 1000
    all_klines = []
    current_start = start_time

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": min(current_start + 1000 * 60 * 60 * 1000, end_time),
            "limit": 1000
        }
        try:
            res = requests.get(url, params=params, timeout=10)
            if res.status_code == 429:
                time.sleep(30)
                continue
            klines = res.json()
            if not klines:
                break
            all_klines.extend(klines)
            current_start = klines[-1][0] + 1
            time.sleep(0.2)
        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")
            break
    return all_klines

def engineer_features(df):
    """Generate 30 predictive features."""
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
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
    df['volume_spike'] = (df['volume'] > df['volume_ma'] * 1.5).astype(int)
    
    # Candlestick
    df['is_hammer'] = (talib.CDLHAMMER(df['open'], high, low, close) == 100).astype(int)
    df['is_morning_star'] = (talib.CDLMORNINGSTAR(df['open'], high, low, close) == 100).astype(int)
    df['is_bullish_engulfing'] = (talib.CDLENGULFING(df['open'], high, low, close) == 100).astype(int)
    
    # Time-based
    df['hour'] = df['timestamp'].dt.hour
    df['is_weekend'] = (df['timestamp'].dt.weekday >= 5).astype(int)
    
    # Lag features
    df['rsi_lag'] = df['rsi'].shift(1)
    df['bb_percent_b_lag'] = df['bb_percent_b'].shift(1)
    
    # NEW: Price acceleration
    df['price_accel'] = close.diff(3) / close.shift(3)
    
    return df

def create_forward_labels(df, min_rally=2.0, future_window=12):
    """Label = 1 if price rallies >=2% within next 12 hours (no indicator filters)."""
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Look ahead 12 hours
    df['future_high'] = df['high'].shift(-future_window).rolling(future_window, min_periods=1).max()
    df['max_rally_pct'] = (df['future_high'] - df['close']) / df['close'] * 100
    
    # Label ANY rally >=2%
    df['label'] = (df['max_rally_pct'] >= min_rally).astype(int)
    
    # Optimal TP = 75th percentile of rallies
    rally_vals = df[df['label'] == 1]['max_rally_pct']
    if len(rally_vals) > 0:
        optimal_tp = np.percentile(rally_vals, 75)
        df['optimal_tp'] = np.where(df['label'] == 1, optimal_tp, 0.0)
    else:
        df['optimal_tp'] = 0.0
    
    return df[[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'ema_9', 'ema_21', 'adx', 'rsi_lag', 'stoch_k', 'stoch_d',
        'macd', 'macd_signal', 'bb_percent_b_lag', 'atr', 'std',
        'obv', 'volume_ratio', 'is_hammer', 'is_morning_star',
        'is_bullish_engulfing', 'hour', 'is_weekend', 'volume_spike',
        'price_accel', 'label', 'optimal_tp'
    ]]

def main():
    print("🚀 BUILDING 500-DAY CRYPTO TRAINING DATASET")
    print("Coins: Top 30 liquid (excl. BTC/ETH/BNB/stablecoins)")
    print("Timeframe: 1-hour candles | Lookback: 500 days\n")
    
    coins = get_top_30_liquid_coins()
    all_data = []
    
    for i, symbol in enumerate(coins, 1):
        print(f"[{i}/{len(coins)}] Processing {symbol}...")
        klines = fetch_klines(symbol, DAYS_BACK, INTERVAL)
        if len(klines) < 1000:
            continue
            
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        df = engineer_features(df)
        if df.empty:
            continue
            
        df = create_forward_labels(df)
        df['symbol'] = symbol
        all_data.append(df)
        df.to_parquet(f"{OUTPUT_DIR}/{symbol}.parquet", index=False)
    
    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        full_df.to_parquet(f"{OUTPUT_DIR}/full_dataset.parquet", index=False)
        print(f"\n✅ Full dataset saved: {len(full_df)} samples")
        print(f"✅ Positive labels: {full_df['label'].sum()} ({full_df['label'].mean()*100:.1f}%)")
    else:
        print("❌ No data collected.")

if __name__ == "__main__":
    main()