# train_top10.py
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

models_dir = 'data/models/top10'
os.makedirs(models_dir, exist_ok=True)

top10_coins = ["PEPE", "DOGE", "SHIB", "BONK", "LUNC", "STRK", "SOON", "ULTIMA", "ATH", "RON"]

for coin in top10_coins:
    npz_path = f"data/processed/{coin}_processed.npz"
    if not os.path.exists(npz_path):
        print(f"Missing processed data for {coin}")
        continue

    print(f"\nTraining {coin}...")
    data = np.load(npz_path)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    scaler_X, scaler_y = data['scaler_X'], data['scaler_y']

    # 12h direction labels
    y_dir_train = np.where(y_train > y_train[0], 1, 0)  # Binary: up/down
    y_dir_test = np.where(y_test > y_test[0], 1, 0)

    inputs = Input(shape=(60, 6))  # 6 features (OHLCV + sentiment)
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    att = Attention()([x, x])
    x = LSTM(80)(x)
    x = Dropout(0.3)(x)
    price_out = Dense(1, name='price')(x)
    dir_out = Dense(1, activation='sigmoid', name='direction')(x)

    model = Model(inputs, [price_out, dir_out])
    model.compile(optimizer=Adam(0.0008),
                  loss={'price': MeanSquaredError(), 'direction': BinaryCrossentropy()},
                  loss_weights={'price': 0.05, 'direction': 0.95},
                  metrics={'direction': ['accuracy', 'AUC']})

    model.fit(X_train, {'price': y_train, 'direction': y_dir_train},
              validation_split=0.1, epochs=150, batch_size=32,
              callbacks=[EarlyStopping(monitor='val_direction_AUC', mode='max', patience=20, restore_best_weights=True)],
              verbose=1)

    p_price, p_dir = model.predict(X_test)
    auc = roc_auc_score(y_dir_test, p_dir)
    print(f"{coin} AUC: {auc:.3f} (target 0.55-0.65)")

    model.save(f"{models_dir}/{coin}_model.h5")
    np.savez(f"{models_dir}/{coin}_scalers.npz", scaler_X=scaler_X, scaler_y=scaler_y)

print("Training complete!")