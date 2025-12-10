# collect_top10.py
import ccxt
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import time
from datetime import datetime, timedelta

exchange = ccxt.binance({'enableRateLimit': True})
analyzer = SentimentIntensityAnalyzer()
os.makedirs('data', exist_ok=True)

top10_coins = ["PEPE", "DOGE", "SHIB", "BONK", "LUNC", "STRK", "ZEC", "HYPER", "FLOKI", "WLD"]
meme_coins = ["PEPE", "DOGE", "SHIB", "BONK"]  # Sentiment target

def fetch_ohlcv(symbol, since_days=730):  # 2+ years
    since = exchange.milliseconds() - (since_days * 24 * 60 * 60 * 1000)
    data = []
    while since < exchange.milliseconds():
        batch = exchange.fetch_ohlcv(f"{symbol}/USDT", '1h', since, limit=1000)
        if not batch: break
        data.extend(batch)
        since = batch[-1][0] + 1
        time.sleep(1)  # Rate limit
    return data

def add_sentiment(df, symbol):
    if symbol not in meme_coins:
        df['sentiment'] = 0
        return df
    # Proxy sentiment from price (replace with real tweets via X API)
    df['sentiment'] = df['close'].pct_change().apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
    return df.fillna(0)

for coin in top10_coins:
    print(f"Collecting {coin}...")
    ohlcv = fetch_ohlcv(coin)
    if len(ohlcv) < 10000:  # Min ~1 year
        print(f"Insufficient data for {coin}")
        continue
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = add_sentiment(df, coin)
    df.to_csv(f"data/{coin}_data.csv", index=False)
    print(f"Saved {len(df)} rows for {coin}")

print("Collection complete!")