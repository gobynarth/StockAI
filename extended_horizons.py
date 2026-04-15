"""
Extended horizon sweep: test h=40 and h=60 on confirmed edge tickers.
Uses Kronos-base with best known params (T=1.0, lb=200).
Usage: python extended_horizons.py RIVN
"""
import sys, os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
from env_paths import add_kronos_to_path, base_path

add_kronos_to_path()
from model import Kronos, KronosTokenizer, KronosPredictor

TICKER = sys.argv[1].upper() if len(sys.argv) > 1 else "RIVN"
SELECTION_N  = 600
VALIDATION_N = 400
HORIZONS     = [30, 40, 60, 90]
TEMPERATURE  = 1.0
LOOKBACK     = 200

CHECKPOINT_DIR = base_path("ext_checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CRYPTO = {"BTC", "SOL", "TAO", "ETH", "DOGE", "XRP"}
yf_ticker = f"{TICKER}-USD" if TICKER.upper() in CRYPTO else TICKER
is_crypto = TICKER.upper() in CRYPTO

def next_dates(from_date, n):
    if is_crypto:
        return pd.date_range(start=from_date + timedelta(days=1), periods=n)
    return pd.bdate_range(start=from_date + timedelta(days=1), periods=n)

csv_path = base_path("data", f"{TICKER}.csv")
if os.path.exists(csv_path):
    raw = pd.read_csv(csv_path, parse_dates=["timestamps"])
    raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)
else:
    raw = yf.download(yf_ticker, period="5y", interval="1d", auto_adjust=True, progress=False, timeout=30)
    raw = raw.reset_index()
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    date_col = "date" if "date" in raw.columns else "datetime"
    raw = raw.rename(columns={date_col: "timestamps"})
    raw["amount"] = raw["close"] * raw["volume"]
    raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)

raw = raw[["timestamps","open","high","low","close","volume","amount"]].dropna()
print(f"Loaded {len(raw)} rows for {TICKER}")

print(f"Loading Kronos-base...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model_obj  = Kronos.from_pretrained("NeoQuasar/Kronos-base")
predictor  = KronosPredictor(model_obj, tokenizer, max_context=512)

all_results = {}

for horizon in HORIZONS:
    cfg_key   = f"h{horizon}_t10_lb{LOOKBACK}"
    ckpt_path = f"{CHECKPOINT_DIR}/{TICKER}_{cfg_key}.csv"

    max_possible = len(raw) - LOOKBACK - horizon
    if max_possible < SELECTION_N + 50:
        print(f"  [{cfg_key}] not enough data, skipping")
        continue
    actual_n  = min(SELECTION_N + VALIDATION_N, max_possible)
    test_start = len(raw) - actual_n - horizon

    if os.path.exists(ckpt_path):
        existing = pd.read_csv(ckpt_path, parse_dates=["date"])
        if len(existing) >= actual_n:
            print(f"  [{cfg_key}] already complete, loading...")
            all_results[cfg_key] = existing
            continue
        rows   = existing.to_dict("records")
        start_i = len(existing)
        print(f"  [{cfg_key}] resuming {start_i}/{actual_n}")
    else:
        rows    = []
        start_i = 0

    print(f"  [{cfg_key}] running {actual_n} windows (h={horizon})...")
    lb = min(LOOKBACK, 512)
    BATCH_SIZE = 16
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
            pred_close   = preds[j]["close"].iloc[horizon - 1]
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
        entry_date  = raw.iloc[idx]["timestamps"]

        x_df = raw.iloc[idx-lb:idx][["open","high","low","close","volume","amount"]].reset_index(drop=True)
        x_ts = raw.iloc[idx-lb:idx]["timestamps"].reset_index(drop=True)
        y_ts = pd.Series(next_dates(entry_date, horizon))

        batch_x_dfs.append(x_df); batch_x_ts_list.append(x_ts)
        batch_y_ts_list.append(y_ts); batch_meta.append((idx, entry_close, entry_date))

        if len(batch_x_dfs) >= BATCH_SIZE or i == actual_n - 1:
            flush_batch()

        if (i + 1) % 100 == 0:
            print(f"    [{cfg_key}] {i+1}/{actual_n}")
            pd.DataFrame(rows).to_csv(ckpt_path, index=False)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(ckpt_path, index=False)
    all_results[cfg_key] = df_out
    print(f"  [{cfg_key}] done.")

print(f"\n{'='*60}")
print(f"EXTENDED HORIZONS: {TICKER} (base, T=1.0, lb={LOOKBACK})")
print(f"{'='*60}")
print(f"{'Config':<20} {'Sel Acc':>8} {'OOS Acc':>8}  Verdict")
print(f"{'-'*50}")
for horizon in HORIZONS:
    cfg_key = f"h{horizon}_t10_lb{LOOKBACK}"
    if cfg_key not in all_results: continue
    df = all_results[cfg_key]
    if len(df) < SELECTION_N + 50: continue
    sel = df.iloc[:SELECTION_N]["correct"].mean() * 100
    oos = df.iloc[-VALIDATION_N:]["correct"].mean() * 100
    verdict = "EDGE" if oos > 55 else ("WEAK" if oos > 50 else "NO EDGE")
    print(f"h={horizon} T=1.0 lb={LOOKBACK:<8} {sel:>7.1f}% {oos:>7.1f}%  {verdict}")
