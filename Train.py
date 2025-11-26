import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import os
import warnings
import joblib
warnings.filterwarnings('ignore')

# ================== CONFIG ==================
DATA_DIR = "training_data"
OUTPUT_MODEL = "crypto_predictor_500d.txt"
FEATURE_COLS = [
    'ema_9', 'ema_21', 'adx', 'rsi_lag', 'stoch_k', 'stoch_d',
    'macd', 'macd_signal', 'bb_percent_b_lag', 'atr', 'std',
    'obv', 'volume_ratio', 'is_hammer', 'is_morning_star',
    'is_bullish_engulfing', 'hour', 'is_weekend', 'volume_spike',
    'price_accel'
]
# ==========================================

def load_full_dataset():
    return pd.read_parquet(f"{DATA_DIR}/full_dataset.parquet")

def simple_train_test(df, test_size=30*24):
    df = df.sort_values('timestamp').reset_index(drop=True)
    test_df = df.tail(test_size)
    train_df = df.iloc[:-test_size]
    
    X_train = train_df[FEATURE_COLS].fillna(0)
    y_train = train_df['label']
    X_test = test_df[FEATURE_COLS].fillna(0)
    y_test = test_df['label']
    
    print(f"Train: {len(train_df)} samples ({y_train.mean()*100:.1f}% pos)")
    print(f"Test:  {len(test_df)} samples ({y_test.mean()*100:.1f}% pos)")
    
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.005,
        max_depth=6,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n✅ RESULTS ON LAST 30 DAYS:")
    print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | AUC: {auc:.3f}")
    return model, (precision, recall, auc)

def train_final_model(df):
    X = df[FEATURE_COLS].fillna(0)
    y = df['label']
    
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.005,
        max_depth=6,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X, y)
    joblib.dump(model, OUTPUT_MODEL)
    return model

def main():
    print("🧠 TRAINING MODEL ON 500-DAY CRYPTO DATASET")
    df = load_full_dataset()
    actual_pos_rate = df['label'].mean() * 100
    print(f"Dataset: 500 days | Positive rate: {actual_pos_rate:.1f}%\n")
    print(f"Loaded {len(df)} samples")
    
    model, metrics = simple_train_test(df)
    precision, recall, _ = metrics
    
    if precision > 0.65:
        print("\n🚀 Model is reliable! Training final model on all data...")
        final_model = train_final_model(df)
        print(f"✅ Final model saved to: {OUTPUT_MODEL}")
    else:
        print(f"\n⚠️ Precision too low ({precision:.3f}). Consider extending data.")

if __name__ == "__main__":
    main()