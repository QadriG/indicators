# validate_top10.py
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model
import pandas as pd

models_dir = 'data/models/top10'
results = []

top10_coins = ["PEPE", "DOGE", "SHIB", "BONK", "LUNC", "STRK", "SOON", "ULTIMA", "ATH", "RON"]

for coin in top10_coins:
    npz_path = f"data/processed/{coin}_processed.npz"
    model_path = f"{models_dir}/{coin}_model.h5"
    if not os.path.exists(model_path):
        print(f"Missing model for {coin}")
        continue

    data = np.load(npz_path)
    X_test = data['X_test']
    y_test = data['y_test']
    scaler_X = data['scaler_X']
    model = load_model(model_path)

    # 2025 backtest subset (assume last 20% is 2025 data)
    recent_split = int(0.5 * len(X_test))  # Last half as proxy for 2025
    X_2025, y_2025 = X_test[recent_split:], y_test[recent_split:]
    y_dir_2025 = np.where(y_2025 > y_2025[0], 1, 0)

    _, p_dir = model.predict(X_2025)
    auc = roc_auc_score(y_dir_2025, p_dir)
    acc = np.mean((p_dir > 0.5) == y_dir_2025) * 100

    # Simulate trades (buy if p_dir > 0.6, sell < 0.4)
    signals = (p_dir > 0.6).astype(int) - (p_dir < 0.4).astype(int)
    returns = np.diff(y_2025) * signals[:-1]
    sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

    results.append({'Coin': coin, 'AUC': auc, 'Accuracy %': acc, 'Sharpe': sharpe})
    print(f"{coin}: AUC {auc:.3f}, Acc {acc:.1f}%, Sharpe {sharpe:.2f}")

df_results = pd.DataFrame(results)
df_results.to_csv('top10_validation.csv', index=False)
print("\nValidation Summary:")
print(df_results)