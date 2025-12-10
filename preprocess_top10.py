# preprocess_top10.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

os.makedirs('data/processed', exist_ok=True)

def create_sequences(X, time_steps=60):
    Xs = []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
    return np.array(Xs)

top10_coins = ["PEPE", "DOGE", "SHIB", "BONK", "LUNC", "STRK", "ZEC", "HYPER", "FLOKI", "WLD"]

for coin in top10_coins:
    csv_path = f"data/{coin}_data.csv"
    if not os.path.exists(csv_path):
        print(f"Missing data for {coin}")
        continue
    print(f"Preprocessing {coin}...")
    df = pd.read_csv(csv_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    df = df.dropna()

    # Features (OHLCV + sentiment for memes)
    features = ['open', 'high', 'low', 'close', 'volume', 'sentiment']
    X = df[features].values
    y = df['close'].shift(-1).values[:-1]
    X = X[:-1]

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    X_seq = create_sequences(X_scaled)
    y_seq = y_scaled[60:]

    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    np.savez(f"data/processed/{coin}_processed.npz",
             X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
             scaler_X=scaler_X, scaler_y=scaler_y)
    print(f"Processed {coin}: {len(X_train)} train / {len(X_test)} test sequences")

print("Preprocessing complete!")