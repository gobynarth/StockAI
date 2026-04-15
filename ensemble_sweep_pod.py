"""
Ensemble sweep for screener Tier 1 candidates.
Runs mini/small/base at h=60 on 6 tickers, saves checkpoints for ensemble_filter.py.
Designed to run on RunPod with RTX 4090.
"""
import sys, os, time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
from env_paths import add_kronos_to_path, base_path

add_kronos_to_path()
from model import Kronos, KronosTokenizer, KronosPredictor

TICKERS = ["ITW", "HD", "NCLH", "HON", "WM", "PATH"]
N_WINDOWS = 1000
LOOKBACK = 200
HORIZONS = [60]
CHECKPOINT_DIR = base_path("ensemble_checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODELS = [
    {"name": "mini",  "model_id": "NeoQuasar/Kronos-mini",  "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",  "max_context": 2048, "params": "4.1M"},
    {"name": "small", "model_id": "NeoQuasar/Kronos-small", "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base","max_context": 512,  "params": "24.7M"},
    {"name": "base",  "model_id": "NeoQuasar/Kronos-base",  "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base","max_context": 512,  "params": "102.3M"},
]

def next_biz_dates(from_date, n):
    return pd.bdate_range(start=from_date + timedelta(days=1), periods=n)

def run_ticker(ticker):
    print(f"\n{'='*60}")
    print(f"  {ticker}")
    print(f"{'='*60}")

    print(f"Fetching {ticker}...")
    raw = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False, timeout=30)
    raw = raw.reset_index()
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    date_col = "date" if "date" in raw.columns else "datetime"
    raw = raw.rename(columns={date_col: "timestamps"})
    raw["amount"] = raw["close"] * raw["volume"]
    raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)
    raw = raw[["timestamps", "open", "high", "low", "close", "volume", "amount"]].dropna()

    max_h = max(HORIZONS)
    n_windows = N_WINDOWS
    required = LOOKBACK + max_h + n_windows
    if len(raw) < required:
        n_windows = len(raw) - LOOKBACK - max_h
        print(f"  Only {len(raw)} rows, reducing to {n_windows} windows")
    if n_windows <= 0:
        print(f"  SKIP {ticker}: not enough data")
        return

    test_start = len(raw) - n_windows - max_h
    all_results = {}

    for cfg in MODELS:
        checkpoint_path = f"{CHECKPOINT_DIR}/{ticker}_{cfg['name']}.csv"

        if os.path.exists(checkpoint_path):
            existing = pd.read_csv(checkpoint_path, parse_dates=["date"])
            completed = len(existing) // len(HORIZONS)
            print(f"\n  [{cfg['name']}] Resuming from checkpoint ({completed}/{n_windows} done)")
            for h in HORIZONS:
                all_results[(cfg["name"], h)] = existing[existing["horizon"] == h].drop(columns="horizon").reset_index(drop=True)
            if completed >= n_windows:
                print(f"  [{cfg['name']}] Already complete, skipping.")
                continue
            start_i = completed
        else:
            start_i = 0

        print(f"  Loading Kronos-{cfg['name']} ({cfg['params']})...")
        tokenizer = KronosTokenizer.from_pretrained(cfg["tokenizer_id"])
        model_obj = Kronos.from_pretrained(cfg["model_id"])
        predictor = KronosPredictor(model_obj, tokenizer, max_context=cfg["max_context"])

        rows = {h: (all_results[(cfg["name"], h)].to_dict("records") if (cfg["name"], h) in all_results else []) for h in HORIZONS}
        lb = min(LOOKBACK, cfg["max_context"])
        BATCH_SIZE = 16

        batch_x_dfs, batch_x_ts_list, batch_y_ts_list, batch_meta = [], [], [], []

        def flush_batch():
            if not batch_x_dfs:
                return
            preds = predictor.predict_batch(
                df_list=batch_x_dfs, x_timestamp_list=batch_x_ts_list,
                y_timestamp_list=batch_y_ts_list, pred_len=max_h,
                T=1.0, top_p=0.9, sample_count=1, verbose=False)
            for j, (bi, bidx, bentry_close, bentry_date) in enumerate(batch_meta):
                for h in HORIZONS:
                    actual_idx = bidx + h
                    if actual_idx >= len(raw): continue
                    pred_close   = preds[j]["close"].iloc[h - 1]
                    actual_close = raw.iloc[actual_idx]["close"]
                    rows[h].append({
                        "date": bentry_date, "entry_close": bentry_close,
                        "pred_close": pred_close, "actual_close": actual_close,
                        "correct": (pred_close > bentry_close) == (actual_close > bentry_close),
                        "error_pct": (pred_close - actual_close) / actual_close * 100,
                    })
            batch_x_dfs.clear(); batch_x_ts_list.clear()
            batch_y_ts_list.clear(); batch_meta.clear()

        t0 = time.time()
        for i in range(start_i, n_windows):
            idx = test_start + i
            entry_close = raw.iloc[idx]["close"]
            entry_date  = raw.iloc[idx]["timestamps"]

            x_df = raw.iloc[idx - lb:idx][["open","high","low","close","volume","amount"]].reset_index(drop=True)
            x_ts = raw.iloc[idx - lb:idx]["timestamps"].reset_index(drop=True)
            y_ts = pd.Series(next_biz_dates(entry_date, max_h))

            batch_x_dfs.append(x_df); batch_x_ts_list.append(x_ts)
            batch_y_ts_list.append(y_ts); batch_meta.append((i, idx, entry_close, entry_date))

            if len(batch_x_dfs) >= BATCH_SIZE or i == n_windows - 1:
                flush_batch()

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1 - start_i) / elapsed
                remaining = (n_windows - i - 1) / rate if rate > 0 else 0
                print(f"  [{cfg['name']}] {i+1}/{n_windows} ({rate:.1f} win/s, ~{remaining/60:.1f}m left)")
                frames = []
                for h in HORIZONS:
                    df_h = pd.DataFrame(rows[h])
                    df_h["horizon"] = h
                    frames.append(df_h)
                pd.concat(frames).to_csv(checkpoint_path, index=False)

        for h in HORIZONS:
            all_results[(cfg["name"], h)] = pd.DataFrame(rows[h])

        frames = []
        for h in HORIZONS:
            df_h = pd.DataFrame(all_results[(cfg["name"], h)])
            df_h["horizon"] = h
            frames.append(df_h)
        pd.concat(frames).to_csv(checkpoint_path, index=False)
        print(f"  [{cfg['name']}] Done. Checkpoint saved.")

        del model_obj, predictor
        import torch; torch.cuda.empty_cache()

    # Quick summary
    print(f"\n  {ticker} SUMMARY (last 400 windows):")
    for cfg in MODELS:
        for h in HORIZONS:
            key = (cfg["name"], h)
            if key not in all_results: continue
            df = all_results[key]
            val_df = df.iloc[-400:] if len(df) >= 400 else df
            acc = val_df["correct"].mean() * 100
            print(f"    {cfg['name']:<8} h={h:<4} OOS Acc: {acc:.1f}% ({len(val_df)} windows)")

for ticker in TICKERS:
    run_ticker(ticker)

print(f"\n{'='*60}")
print(f"ALL DONE. Checkpoints in {CHECKPOINT_DIR}/")
print(f"{'='*60}")
