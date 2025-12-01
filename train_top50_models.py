import pandas as pd
import numpy as np
import joblib
import os
from lightgbm import LGBMClassifier, LGBMRegressor

# ================= CONFIG ==================
DATA_DIR = "top50_data_3y"
MODELS_DIR = "top50_models_3y"
os.makedirs(MODELS_DIR, exist_ok=True)

ALL_FEATURES = (
    FEATURES_1H + FEATURES_2H + FEATURES_4H
)

# ==========================================

def train_coin_model(symbol):
    try:
        df = pd.read_parquet(f"{DATA_DIR}/{symbol}.parquet")
        if len(df) < 100: return None
        
        X = df[ALL_FEATURES].fillna(0)
        y_label = df['label'].astype(int)
        y_tp = df['max_tp']
        y_hold = df['hold_hours']
        
        # Classifier: Will it hit 2%+?
        clf = LGBMClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.03,
            class_weight='balanced',
            random_state=42
        )
        clf.fit(X, y_label)
        
        # Regressor: Max TP (only on positive cases)
        pos_mask = y_label == 1
        if pos_mask.sum() < 50: return None
        
        X_pos = X[pos_mask]
        reg_tp = LGBMRegressor(n_estimators=200, max_depth=6, random_state=42)
        reg_tp.fit(X_pos, y_tp[pos_mask])
        
        reg_hold = LGBMRegressor(n_estimators=150, max_depth=5, random_state=42)
        reg_hold.fit(X_pos, y_hold[pos_mask])
        
        joblib.dump({
            'classifier': clf,
            'regressor_tp': reg_tp,
            'regressor_hold': reg_hold,
            'features': ALL_FEATURES,
            'avg_tp': y_tp[pos_mask].mean(),
            'avg_hold': y_hold[pos_mask].mean()
        }, f"{MODELS_DIR}/{symbol}.pkl")
        
        return len(df)
    except Exception as e:
        print(f"[ERROR] {symbol}: {e}")
        return None

def main():
    print("🧠 Training coin-specific models...")
    coins = [f.replace('.parquet', '') for f in os.listdir(DATA_DIR) if f.endswith('.parquet')]
    trained = 0
    for symbol in coins:
        count = train_coin_model(symbol)
        if count:
            print(f"✅ {symbol}: {count} setups")
            trained += 1
    print(f"\n✅ Trained {trained} models in '{MODELS_DIR}/'")

if __name__ == "__main__":
    main()