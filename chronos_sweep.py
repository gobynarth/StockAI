"""
Chronos (Amazon) backtest: same methodology as Kronos sweep.
Tests direction accuracy on edge tickers at h=40/60/90.
"""
import sys, os
import pandas as pd
import numpy as np
import torch
import yfinance as yf
from datetime import timedelta

TICKERS = ["RIVN", "COIN", "ENVX", "NIO", "RIOT", "SMCI", "CRWD", "MARA",
           "PLTR", "HOOD"]
HORIZONS = [40, 60, 90]
LOOKBACK = 200
VALIDATION_N = 400
CKPT_DIR = "/workspace/chronos_checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)


def fetch_data(ticker):
    raw = yf.download(ticker, period="5y", interval="1d",
                      auto_adjust=True, progress=False, timeout=30)
    raw = raw.reset_index()
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    date_col = "date" if "date" in raw.columns else "datetime"
    raw = raw.rename(columns={date_col: "timestamps"})
    raw["amount"] = raw["close"] * raw["volume"]
    raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)
    return raw[["timestamps", "open", "high", "low", "close", "volume", "amount"]].dropna()


from chronos import ChronosPipeline

print("Loading Chronos-T5-Base...")
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-base",
    device_map="cuda",
    torch_dtype=torch.float32,
)

all_results = {}

for ticker in TICKERS:
    print("")
    print("=" * 60)
    print(ticker)
    print("=" * 60)
    try:
        raw = fetch_data(ticker)
    except Exception as e:
        print(f"  FAILED: {e}")
        continue
    print(f"  {len(raw)} rows loaded")
    if len(raw) < 500:
        print("  SKIPPING -- too few rows")
        continue

    for horizon in HORIZONS:
        cfg_key = f"{ticker}_h{horizon}"
        ckpt_path = f"{CKPT_DIR}/{cfg_key}.csv"

        max_possible = len(raw) - LOOKBACK - horizon
        if max_possible < VALIDATION_N:
            print(f"  [h={horizon}] not enough data, skipping")
            continue
        actual_n = min(600 + VALIDATION_N, max_possible)
        test_start = len(raw) - actual_n - horizon

        if os.path.exists(ckpt_path):
            existing = pd.read_csv(ckpt_path, parse_dates=["date"])
            if len(existing) >= actual_n:
                print(f"  [h={horizon}] already done ({len(existing)} windows)")
                all_results[cfg_key] = existing
                continue
            rows = existing.to_dict("records")
            start_i = len(existing)
            print(f"  [h={horizon}] resuming {start_i}/{actual_n}")
        else:
            rows = []
            start_i = 0

        print(f"  [h={horizon}] running {actual_n} windows...")
        for i in range(start_i, actual_n):
            idx = test_start + i
            entry_close = float(raw.iloc[idx]["close"])
            entry_date = raw.iloc[idx]["timestamps"]

            context = torch.tensor(
                raw.iloc[idx - LOOKBACK:idx]["close"].values, dtype=torch.float32
            )
            forecast = pipeline.predict(context.unsqueeze(0), horizon)
            pred_close = float(forecast.median(dim=1).values[0, -1].item())

            actual_idx = idx + horizon
            if actual_idx >= len(raw):
                continue
            actual_close = float(raw.iloc[actual_idx]["close"])
            rows.append({
                "date": entry_date,
                "entry_close": entry_close,
                "pred_close": pred_close,
                "actual_close": actual_close,
                "correct": (pred_close > entry_close) == (actual_close > entry_close),
                "error_pct": (pred_close - actual_close) / actual_close * 100,
            })

            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{actual_n}")
                pd.DataFrame(rows).to_csv(ckpt_path, index=False)

        df_out = pd.DataFrame(rows)
        df_out.to_csv(ckpt_path, index=False)
        all_results[cfg_key] = df_out
        print(f"  [h={horizon}] done.")

print("")
print("")
print("=" * 60)
print(f"CHRONOS RESULTS (base, lb={LOOKBACK})")
print("=" * 60)
print("Ticker    Horizon   OOS Acc   Sel Acc  Verdict")
print("-" * 55)

for ticker in TICKERS:
    for horizon in HORIZONS:
        cfg_key = f"{ticker}_h{horizon}"
        if cfg_key not in all_results:
            continue
        df = all_results[cfg_key]
        n = len(df)
        if n < VALIDATION_N:
            continue
        oos = df.tail(VALIDATION_N)["correct"].mean() * 100
        sel_df = df.iloc[:max(0, n - VALIDATION_N)]
        sel = sel_df["correct"].mean() * 100 if len(sel_df) > 0 else 0
        if oos > 55:
            verdict = "** EDGE **"
        elif oos > 50:
            verdict = "weak"
        else:
            verdict = "no edge"
        print(f"{ticker:<8} h={horizon:>3}    {oos:>7.1f}%  {sel:>7.1f}%  {verdict}")
