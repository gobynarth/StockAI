"""
Fetch and save daily OHLCV CSVs for all edge tickers.
Run this before uploading to RunPod.
"""
import yfinance as yf
import pandas as pd
import os

TICKERS = ["RIVN", "ENVX", "TSLA", "COIN"]
OUT_DIR = "C:/Users/Dream/StockAI/data"
os.makedirs(OUT_DIR, exist_ok=True)

for ticker in TICKERS:
    print(f"Fetching {ticker}...")
    raw = yf.download(ticker, period="5y", interval="1d",
                      auto_adjust=True, progress=False, timeout=30)
    raw = raw.reset_index()
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                   for c in raw.columns]
    date_col = "date" if "date" in raw.columns else "datetime"
    raw = raw.rename(columns={date_col: "timestamps"})
    raw["amount"] = raw["close"] * raw["volume"]
    raw = raw[["timestamps","open","high","low","close","volume","amount"]].dropna()
    out = os.path.join(OUT_DIR, f"{ticker}.csv")
    raw.to_csv(out, index=False)
    print(f"  Saved {len(raw)} rows -> {out}")

print("Done.")
