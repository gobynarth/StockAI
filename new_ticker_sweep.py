"""
Extended horizon sweep for new candidate tickers.
Uses winning recipe: Kronos-base, T=1.0, lb=200, h=[40,60,90]
"""
import sys, os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
sys.path.insert(0, "/workspace/Kronos")
from model import Kronos, KronosTokenizer, KronosPredictor

TICKERS = ["MARA", "RIOT", "HOOD", "SOFI", "LCID", "NIO", "IONQ", "SMCI", "PLTR"]
HORIZONS = [40, 60, 90]
TEMPERATURE = 1.0
LOOKBACK = 200
SELECTION_N = 600
VALIDATION_N = 400
BATCH_SIZE = 16
CKPT_DIR = "/workspace/StockAI/new_ticker_checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

def next_dates(from_date, n):
    return pd.bdate_range(start=from_date + timedelta(days=1), periods=n)

def fetch_data(ticker):
    csv_path = f"/workspace/StockAI/data/{ticker}.csv"
    if os.path.exists(csv_path):
        raw = pd.read_csv(csv_path, parse_dates=["timestamps"])
        raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)
    else:
        raw = yf.download(ticker, period="5y", interval="1d",
                          auto_adjust=True, progress=False, timeout=30)
        raw = raw.reset_index()
        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
        date_col = "date" if "date" in raw.columns else "datetime"
        raw = raw.rename(columns={date_col: "timestamps"})
        raw["amount"] = raw["close"] * raw["volume"]
        raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)
        os.makedirs("/workspace/StockAI/data", exist_ok=True)
        raw.to_csv(csv_path, index=False)
    return raw[["timestamps","open","high","low","close","volume","amount"]].dropna()

print("Loading Kronos-base...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model_obj = Kronos.from_pretrained("NeoQuasar/Kronos-base")
predictor = KronosPredictor(model_obj, tokenizer, max_context=512)

all_results = {}

for ticker in TICKERS:
    print(f"\n{'='*60}")
    print(f"{ticker}")
    print(f"{'='*60}")
    try:
        raw = fetch_data(ticker)
    except Exception as e:
        print(f"  FAILED to fetch data: {e}")
        continue
    print(f"  {len(raw)} rows loaded")

    for horizon in HORIZONS:
        cfg_key = f"{ticker}_h{horizon}"
        ckpt_path = f"{CKPT_DIR}/{cfg_key}.csv"

        max_possible = len(raw) - LOOKBACK - horizon
        if max_possible < SELECTION_N + 50:
            actual_n = max(max_possible, VALIDATION_N + 50)
            if actual_n < VALIDATION_N:
                print(f"  [h={horizon}] not enough data ({max_possible} windows), skipping")
                continue
        else:
            actual_n = min(SELECTION_N + VALIDATION_N, max_possible)

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
        lb = min(LOOKBACK, 512)
        batch_x_dfs, batch_x_ts_list, batch_y_ts_list, batch_meta = [], [], [], []

        def flush_batch():
            if not batch_x_dfs: return
            preds = predictor.predict_batch(
                df_list=batch_x_dfs, x_timestamp_list=batch_x_ts_list,
                y_timestamp_list=batch_y_ts_list, pred_len=horizon,
                T=TEMPERATURE, top_p=0.9, sample_count=1, verbose=False)
            for j, (bidx, bentry_close, bentry_date) in enumerate(batch_meta):
                actual_idx = bidx + horizon
                if actual_idx >= len(raw): continue
                pred_close = preds[j]["close"].iloc[horizon - 1]
                actual_close = raw.iloc[actual_idx]["close"]
                rows.append({
                    "date": bentry_date, "entry_close": bentry_close,
                    "pred_close": pred_close, "actual_close": actual_close,
                    "correct": (pred_close > bentry_close) == (actual_close > bentry_close),
                    "error_pct": (pred_close - actual_close) / actual_close * 100,
                })
            batch_x_dfs.clear(); batch_x_ts_list.clear()
            batch_y_ts_list.clear(); batch_meta.clear()

        for i in range(start_i, actual_n):
            idx = test_start + i
            entry_close = raw.iloc[idx]["close"]
            entry_date = raw.iloc[idx]["timestamps"]
            x_df = raw.iloc[idx-lb:idx][["open","high","low","close","volume","amount"]].reset_index(drop=True)
            x_ts = raw.iloc[idx-lb:idx]["timestamps"].reset_index(drop=True)
            y_ts = pd.Series(next_dates(entry_date, horizon))
            batch_x_dfs.append(x_df); batch_x_ts_list.append(x_ts)
            batch_y_ts_list.append(y_ts); batch_meta.append((idx, entry_close, entry_date))
            if len(batch_x_dfs) >= BATCH_SIZE or i == actual_n - 1:
                flush_batch()
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{actual_n}")
                pd.DataFrame(rows).to_csv(ckpt_path, index=False)

        df_out = pd.DataFrame(rows)
        df_out.to_csv(ckpt_path, index=False)
        all_results[cfg_key] = df_out
        print(f"  [h={horizon}] done.")

# Final summary
print(f"\n\n{'='*60}")
print(f"RESULTS SUMMARY (base, T=1.0, lb=200)")
print(f"{'='*60}")
print(f"{'Ticker':<8} {'Horizon':>8} {'OOS Acc':>9} {'Sel Acc':>9}  Verdict")
print(f"{'-'*50}")

winners = []
for ticker in TICKERS:
    for horizon in HORIZONS:
        cfg_key = f"{ticker}_h{horizon}"
        if cfg_key not in all_results: continue
        df = all_results[cfg_key]
        n = len(df)
        if n < VALIDATION_N:
            oos = df["correct"].mean() * 100
            sel = 0
        else:
            oos_df = df.tail(VALIDATION_N)
            sel_df = df.iloc[:max(0, n - VALIDATION_N)]
            oos = oos_df["correct"].mean() * 100
            sel = sel_df["correct"].mean() * 100 if len(sel_df) > 0 else 0

        if oos > 55:
            verdict = "** EDGE **"
            winners.append((ticker, horizon, oos))
        elif oos > 50:
            verdict = "weak"
        else:
            verdict = "no edge"
        print(f"{ticker:<8} h={horizon:>3}    {oos:>7.1f}%  {sel:>7.1f}%  {verdict}")

if winners:
    print(f"\nWINNERS (>55% OOS):")
    for t, h, acc in sorted(winners, key=lambda x: -x[2]):
        print(f"  {t} h={h}: {acc:.1f}%")
