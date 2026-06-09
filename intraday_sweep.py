"""
Level 6: Intraday sweep using 1-hour candles.
Tests if the edge holds at intraday timeframes (h=5,10,20 hours = 1-4 trading days).
Uses Kronos-base with best params (T=1.0, lb=100 hourly bars).

yfinance gives ~730 days of hourly data (market hours only: 9:30-16:00 ET).
We use market-hours-only bars, so no overnight gaps pollute sequences.

Usage:
    python intraday_sweep.py RIVN
    python intraday_sweep.py NIO
    python intraday_sweep.py RIVN --model small   (to test mini/small too)

Output: intraday_checkpoints/<TICKER>_h<N>_t10_lb100.csv
"""
import sys, os, argparse
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta, datetime

sys.path.append("C:/Users/Dream/Kronos")
from model import Kronos, KronosTokenizer, KronosPredictor

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument("ticker", nargs="?", default="RIVN")
parser.add_argument("--model", default="base", choices=["mini", "small", "base"])
parser.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 20])
parser.add_argument("--lookback", type=int, default=100)
parser.add_argument("--temp", type=float, default=1.0)
parser.add_argument("--n_windows", type=int, default=400,
                    help="Total rolling windows to evaluate (split 50/50 selection/validation)")
args = parser.parse_args()

TICKER      = args.ticker.upper()
MODEL_NAME  = args.model
HORIZONS    = args.horizons
LOOKBACK    = args.lookback
TEMP        = args.temp
N_WINDOWS   = args.n_windows
SELECTION_N = N_WINDOWS // 2
VALIDATION_N = N_WINDOWS - SELECTION_N

CHECKPOINT_DIR = "C:/Users/Dream/Projects/StockAI/intraday_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODEL_CONFIGS = {
    "mini":  {"model_id": "NeoQuasar/Kronos-mini",  "tok_id": "NeoQuasar/Kronos-Tokenizer-2k",   "max_ctx": 2048},
    "small": {"model_id": "NeoQuasar/Kronos-small", "tok_id": "NeoQuasar/Kronos-Tokenizer-base", "max_ctx": 512},
    "base":  {"model_id": "NeoQuasar/Kronos-base",  "tok_id": "NeoQuasar/Kronos-Tokenizer-base", "max_ctx": 512},
}
cfg = MODEL_CONFIGS[MODEL_NAME]

# --- Load hourly data ---
print(f"Downloading hourly data for {TICKER}...")
raw = yf.download(TICKER, period="730d", interval="1h",
                  auto_adjust=True, progress=False, timeout=60)
raw = raw.reset_index()
raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
date_col = "datetime" if "datetime" in raw.columns else "date"
raw = raw.rename(columns={date_col: "timestamps"})
raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)
raw["amount"] = raw["close"] * raw["volume"]
raw = raw[["timestamps", "open", "high", "low", "close", "volume", "amount"]].dropna()

# Keep only market hours: 9:30–16:00 ET Mon-Fri
raw = raw[raw["timestamps"].dt.weekday < 5]
raw = raw[raw["timestamps"].dt.time >= pd.Timestamp("09:30").time()]
raw = raw[raw["timestamps"].dt.time <= pd.Timestamp("16:00").time()]
raw = raw.reset_index(drop=True)

print(f"Loaded {len(raw)} hourly bars for {TICKER} (market hours only)")
print(f"  Date range: {raw['timestamps'].iloc[0].date()} to {raw['timestamps'].iloc[-1].date()}")

if len(raw) < LOOKBACK + max(HORIZONS) + 50:
    print("Not enough hourly data. Exiting.")
    sys.exit(1)

# --- Load model ---
print(f"\nLoading Kronos-{MODEL_NAME}...")
tok = KronosTokenizer.from_pretrained(cfg["tok_id"])
mdl = Kronos.from_pretrained(cfg["model_id"])
predictor = KronosPredictor(mdl, tok, max_context=cfg["max_ctx"])
print("Model loaded.")

# --- Sweep ---
all_results = {}

for horizon in HORIZONS:
    cfg_key   = f"h{horizon}_t{int(TEMP*10)}_lb{LOOKBACK}"
    ckpt_path = f"{CHECKPOINT_DIR}/{TICKER}_{MODEL_NAME}_{cfg_key}.csv"

    max_possible = len(raw) - LOOKBACK - horizon
    if max_possible < SELECTION_N + 10:
        print(f"\n[{cfg_key}] Not enough data ({max_possible} windows), skipping.")
        continue

    actual_n  = min(N_WINDOWS, max_possible)
    test_start = len(raw) - actual_n - horizon

    if os.path.exists(ckpt_path):
        existing = pd.read_csv(ckpt_path, parse_dates=["timestamp"])
        if len(existing) >= actual_n:
            print(f"\n[{cfg_key}] Already complete ({len(existing)} rows), loading...")
            all_results[cfg_key] = existing
            continue
        rows    = existing.to_dict("records")
        start_i = len(existing)
        print(f"\n[{cfg_key}] Resuming from {start_i}/{actual_n}")
    else:
        rows    = []
        start_i = 0

    print(f"\n[{cfg_key}] Running {actual_n} windows (horizon={horizon}h)...")

    for i in range(start_i, actual_n):
        idx = test_start + i
        x_df = raw.iloc[idx - LOOKBACK: idx][["open","high","low","close","volume","amount"]].reset_index(drop=True)
        x_ts = raw.iloc[idx - LOOKBACK: idx]["timestamps"].reset_index(drop=True)

        # Future y timestamps: next `horizon` hourly bars from raw
        if idx + horizon > len(raw):
            break  # not enough future data
        y_ts  = raw.iloc[idx: idx + horizon]["timestamps"].reset_index(drop=True)

        entry_close  = raw.iloc[idx - 1]["close"]
        actual_close = raw.iloc[idx + horizon - 1]["close"]
        entry_ts     = raw.iloc[idx - 1]["timestamps"]
        actual_ts    = raw.iloc[idx + horizon - 1]["timestamps"]

        try:
            pred = predictor.predict(
                df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                pred_len=horizon, T=TEMP, top_p=0.9,
                sample_count=1, verbose=False
            )
            pred_close = pred["close"].iloc[-1]
        except Exception as e:
            print(f"  [{i}] predict error: {e}")
            continue

        pred_dir   = pred_close > entry_close
        actual_dir = actual_close > entry_close
        correct    = pred_dir == actual_dir
        pct_pred   = (pred_close   - entry_close) / entry_close * 100
        pct_actual = (actual_close - entry_close) / entry_close * 100

        rows.append({
            "timestamp":     str(entry_ts),
            "actual_ts":     str(actual_ts),
            "entry_close":   round(float(entry_close),   4),
            "pred_close":    round(float(pred_close),    4),
            "actual_close":  round(float(actual_close),  4),
            "pred_direction":  int(pred_dir),
            "actual_direction":int(actual_dir),
            "correct":         int(correct),
            "pct_pred":        round(float(pct_pred),    2),
            "pct_actual":      round(float(pct_actual),  2),
        })

        if (i + 1) % 20 == 0 or i == actual_n - 1:
            pd.DataFrame(rows).to_csv(ckpt_path, index=False)
            acc_so_far = np.mean([r["correct"] for r in rows]) * 100
            print(f"  [{i+1}/{actual_n}] acc={acc_so_far:.1f}%  saved.")

    all_results[cfg_key] = pd.DataFrame(rows)
    print(f"  [{cfg_key}] Done. {len(rows)} windows saved to {ckpt_path}")

# --- Results summary ---
print(f"\n{'='*70}")
print(f"INTRADAY RESULTS: {TICKER} (Kronos-{MODEL_NAME})")
print(f"{'='*70}")
print(f"{'Config':<25} {'N':>5} {'Sel Acc':>8} {'Val Acc':>8} {'Val UP%':>8} {'Val DN%':>8}")
print(f"{'-'*65}")

for cfg_key, df in sorted(all_results.items()):
    if df.empty:
        continue
    n = len(df)
    sel = df.iloc[:SELECTION_N]
    val = df.iloc[SELECTION_N:]
    if val.empty:
        continue

    sel_acc = sel["correct"].mean() * 100
    val_acc = val["correct"].mean() * 100

    # Direction breakdown
    val_up = val[val["actual_direction"] == 1]
    val_dn = val[val["actual_direction"] == 0]
    val_up_acc = val_up["correct"].mean() * 100 if len(val_up) > 0 else float("nan")
    val_dn_acc = val_dn["correct"].mean() * 100 if len(val_dn) > 0 else float("nan")

    print(f"{cfg_key:<25} {n:>5} {sel_acc:>7.1f}% {val_acc:>7.1f}% {val_up_acc:>7.1f}% {val_dn_acc:>7.1f}%")

print(f"\nBaseline (random): 50.0%")
print(f"Edge threshold: >55% val acc = weak edge, >60% = strong edge")
print(f"\nCheckpoints saved to: {CHECKPOINT_DIR}/")

# Comparison vs daily
print(f"\n--- Daily benchmark (from prior research) ---")
DAILY_BENCHMARKS = {
    "RIVN": {"h=40": 75.0, "h=90": None},
    "NIO":  {"h=90": 69.5},
    "TSLA": {"h=90": 57.2},
    "COIN": {"h=60": 71.8},
}
if TICKER in DAILY_BENCHMARKS:
    for hkey, acc in DAILY_BENCHMARKS[TICKER].items():
        if acc:
            print(f"  {TICKER} daily {hkey}: {acc:.1f}% OOS")
