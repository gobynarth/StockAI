"""
Fetch daily OHLCV data from CoinGecko and save as CSV compatible with full_backtest.py
Usage: python fetch_coingecko.py TAO
       python fetch_coingecko.py DOGE
"""
import sys
import time
import pandas as pd
from pycoingecko import CoinGeckoAPI

# Map ticker -> CoinGecko coin ID
COIN_IDS = {
    "TAO":  "bittensor",
    "DOGE": "dogecoin",
    "XRP":  "ripple",
    "SOL":  "solana",
    "BTC":  "bitcoin",
    "ETH":  "ethereum",
    "ADA":  "cardano",
    "AVAX": "avalanche-2",
    "MATIC":"matic-network",
    "DOT":  "polkadot",
}

TICKER = sys.argv[1].upper() if len(sys.argv) > 1 else "TAO"
coin_id = COIN_IDS.get(TICKER)
if not coin_id:
    print(f"Unknown ticker {TICKER}. Add it to COIN_IDS dict.")
    sys.exit(1)

cg = CoinGeckoAPI()
print(f"Fetching {TICKER} ({coin_id}) from CoinGecko...")

# Get max available OHLC data (free tier: daily candles for up to 180 days, weekly beyond)
# Use market_chart for full history with volume
data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency="usd", days=365, interval="daily")

prices  = pd.DataFrame(data["prices"],        columns=["ts", "close"])
volumes = pd.DataFrame(data["total_volumes"], columns=["ts", "volume"])

df = prices.merge(volumes, on="ts")
df["timestamps"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
df = df.drop(columns="ts")

# CoinGecko daily chart doesn't give OHLC — approximate with close only
# For proper OHLC use the /ohlc endpoint (max 180 days on free tier)
# Fetch OHLC and merge
print("Fetching OHLC data (may be limited to 180 days on free tier)...")
try:
    ohlc_raw = cg.get_coin_ohlc_by_id(id=coin_id, vs_currency="usd", days="max")
    ohlc = pd.DataFrame(ohlc_raw, columns=["ts", "open", "high", "low", "close_ohlc"])
    ohlc["timestamps"] = pd.to_datetime(ohlc["ts"], unit="ms").dt.normalize()
    ohlc = ohlc.drop(columns=["ts", "close_ohlc"])
    df = df.merge(ohlc, on="timestamps", how="left")
except Exception as e:
    print(f"OHLC fetch failed ({e}), using close as open/high/low")
    df["open"] = df["close"]
    df["high"] = df["close"]
    df["low"]  = df["close"]

df["amount"] = df["close"] * df["volume"]
df = df[["timestamps", "open", "high", "low", "close", "volume", "amount"]].dropna()
df = df.sort_values("timestamps").reset_index(drop=True)

out = f"C:/Users/Dream/StockAI/data/{TICKER}.csv"
import os; os.makedirs("C:/Users/Dream/StockAI/data", exist_ok=True)
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows to {out}")
print(f"Date range: {df['timestamps'].min()} to {df['timestamps'].max()}")
