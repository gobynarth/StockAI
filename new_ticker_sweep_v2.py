"""
New ticker sweep — runs h=40/60/90 backtest on candidate tickers.
Designed for RunPod (4090). Resumable via checkpoints.
"""
import sys, os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta

sys.path.insert(0, "/workspace/Kronos")
from model import Kronos, KronosTokenizer, KronosPredictor

# Tickers to test (X-trending + user-requested)
TICKERS = ["ASTS", "LUNR", "RKLB", "IREN", "CAR", "DJT"]

HORIZONS    = [40, 60, 90]
TEMPERATURE = 1.0
LOOKBACK    = 200
SELECTION_N = 600
VALIDATION_N = 400

CKPT_DIR = "/workspace/new_ticker_checkpoints"
DATA_DIR = "/workspace/data"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_data(ticker):
    csv_path = f"{DATA_DIR}/{ticker}.csv"
    if os.path.exists(csv_path):
        raw = pd.read_csv(csv_path, parse_dates=["timestamps"])
        raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)
    else:
        raw = yf.download(ticker, period="5y", interval="1d",
                          auto_adjust=True, progress=False, timeout=60)
        raw = raw.reset_index()
        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
        date_col = "date" if "date" in raw.columns else "datetime"
        raw = raw.rename(columns={date_col: "timestamps"})
        raw["amount"] = raw["close"] * raw["volume"]
        raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)
        raw[["timestamps","open","high","low","close","volume","amount"]].dropna().to_csv(csv_path, index=False)
    raw = raw[["timestamps","open","high","low","close","volume","amount"]].dropna()
    return raw


print(f"Loading Kronos-base...")
tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
mdl = Kronos.from_pretrained("NeoQuasar/Kronos-base")
predictor = KronosPredictor(mdl, tok, max_context=512)
print("Model loaded.")

for ticker in TICKERS:
    print(f"\n{'='*70}\n{ticker}\n{'='*70}")
    try:
        raw = fetch_data(ticker)
    except Exception as e:
        print(f"  Skipping {ticker}: {e}")
        continue
    print(f"  {len(raw)} rows loaded ({raw['timestamps'].iloc[0].date()} to {raw['timestamps'].iloc[-1].date()})")

    if len(raw) < LOOKBACK + max(HORIZONS) + 100:
        print(f"  Not enough data, skipping.")
        continue

    for horizon in HORIZONS:
        cfg_key = f"h{horizon}"
        ckpt = f"{CKPT_DIR}/{ticker}_{cfg_key}.csv"

        max_n = len(raw) - LOOKBACK - horizon
        if max_n < 200:
            print(f"  [h={horizon}] not enough data ({max_n} windows), skipping")
            continue
        actual_n = min(SELECTION_N + VALIDATION_N, max_n)
        test_start = len(raw) - actual_n - horizon

        if os.path.exists(ckpt):
            existing = pd.read_csv(ckpt, parse_dates=["date"])
            if len(existing) >= actual_n:
                print(f"  [h={horizon}] complete, skipping")
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
            x_df = raw.iloc[idx - LOOKBACK: idx][["open","high","low","close","volume","amount"]].reset_index(drop=True)
            x_ts = raw.iloc[idx - LOOKBACK: idx]["timestamps"].reset_index(drop=True)
            entry_close = raw.iloc[idx - 1]["close"]
            entry_date  = raw.iloc[idx - 1]["timestamps"]
            actual_close = raw.iloc[idx + horizon - 1]["close"]
            y_dates = pd.bdate_range(start=entry_date + timedelta(days=1), periods=horizon)
            y_ts = pd.Series(y_dates)

            try:
                pred = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                                         pred_len=horizon, T=TEMPERATURE, top_p=0.9,
                                         sample_count=1, verbose=False)
                pred_close = pred["close"].iloc[-1]
            except Exception as e:
                print(f"    [{i}] error: {e}")
                continue

            correct = (pred_close > entry_close) == (actual_close > entry_close)
            rows.append({
                "date": str(entry_date.date()),
                "entry_close": round(float(entry_close), 4),
                "pred_close":  round(float(pred_close),  4),
                "actual_close":round(float(actual_close), 4),
                "correct": int(correct),
            })

            if (i + 1) % 50 == 0 or i == actual_n - 1:
                pd.DataFrame(rows).to_csv(ckpt, index=False)
                acc = np.mean([r["correct"] for r in rows]) * 100
                print(f"    [{i+1}/{actual_n}] acc={acc:.1f}%")

        pd.DataFrame(rows).to_csv(ckpt, index=False)
        print(f"  [h={horizon}] done.")

# Summary
print(f"\n{'='*70}\nNEW TICKER RESULTS\n{'='*70}")
print(f"{'Ticker':<7} {'Horizon':>8} {'Total':>7} {'Sel Acc':>9} {'Val Acc':>9}")
print(f"{'-'*55}")
for ticker in TICKERS:
    for h in HORIZONS:
        ckpt = f"{CKPT_DIR}/{ticker}_h{h}.csv"
        if not os.path.exists(ckpt):
            continue
        df = pd.read_csv(ckpt)
        n = len(df)
        if n < 100:
            continue
        sel = df.iloc[:n//2 if n < SELECTION_N+VALIDATION_N else SELECTION_N]
        val = df.iloc[len(sel):]
        sel_acc = sel["correct"].mean() * 100
        val_acc = val["correct"].mean() * 100 if len(val) > 0 else float('nan')
        print(f"{ticker:<7} h={h:<6} {n:>7} {sel_acc:>8.1f}% {val_acc:>8.1f}%")
