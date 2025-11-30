import pandas as pd
import numpy as np
import joblib
import os
from lightgbm import LGBMRegressor

# ================== CONFIG ==================
DATA_DIR = "altcoin_data_v3"
MODELS_DIR = "altcoin_models_v3"
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES = [
    'ema_ratio', 'rsi_diff', 'bb_position', 'bb_width',
    'volume_ratio', 'volatility', 'high_low_ratio', 'body_size'
]
# ==========================================

def train_altcoin_model(symbol):
    try:
        df = pd.read_parquet(f"{DATA_DIR}/{symbol}.parquet")
        # ➕ Recompute future rally (24h = 24 candles)
        df['future_rally'] = (
            df['close'].shift(-24).rolling(24, min_periods=1).max() - df['close']
        ) / df['close'] * 100

        # Keep only entries with 25–100% rally
        df = df[(df['label'] == 1) & (df['future_rally'] >= 25.0) & (df['future_rally'] <= 100.0)]

        if len(df) < 30:
            return None
        
        X = df[FEATURES].fillna(0)
        y_rally = df['future_rally']  # ✅ THIS WAS MISSING
        y_min = y_rally
        y_max = y_rally
        
        # Min TP model (25th percentile)
        model_min = LGBMRegressor(
            objective='quantile',
            alpha=0.25,
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            device='gpu',
            random_state=42
        )
        model_min.fit(X, y_min)
        
        # Max TP model (75th percentile)
        model_max = LGBMRegressor(
            objective='quantile',
            alpha=0.75,
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            device='gpu',
            random_state=42
        )
        model_max.fit(X, y_max)
        
        # Hold time models (if your data has these columns)
        if 'min_hold' in df.columns and 'max_hold' in df.columns:
            model_min_hold = LGBMRegressor(n_estimators=200, max_depth=6, device='gpu')
            model_min_hold.fit(X, df['min_hold'])
            
            model_max_hold = LGBMRegressor(n_estimators=200, max_depth=6, device='gpu')
            model_max_hold.fit(X, df['max_hold'])
        else:
            # If hold time columns don't exist, skip or use dummy models
            model_min_hold = None
            model_max_hold = None
        
        joblib.dump({
            'model_min_tp': model_min,
            'model_max_tp': model_max,
            'model_min_hold': model_min_hold,
            'model_max_hold': model_max_hold,
            'avg_min_tp': float(y_min.mean()),
            'avg_max_tp': float(y_max.mean())
        }, f"{MODELS_DIR}/{symbol}.pkl")
        
        return len(df)
    except Exception as e:
        print(f"[ERROR] {symbol}: {e}")
        return None

def main():
    print("🧠 TRAINING ALTCOIN MODELS (MIN/MAX TP + HOLD TIME)")
    
    coins = [f.replace('.parquet', '') for f in os.listdir(DATA_DIR) if f.endswith('.parquet')]
    trained = 0
    
    for symbol in coins:
        count = train_altcoin_model(symbol)
        if count:
            print(f"✅ {symbol}: {count} signals")
            trained += 1
    
    print(f"\n✅ Trained {trained} models in '{MODELS_DIR}/'")

if __name__ == "__main__":
    main()