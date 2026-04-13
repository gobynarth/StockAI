"""
Sample count sweep: test averaging N predictions per window to reduce variance.
Uses best known config: Kronos-base, h=30, T=1.0, lb=200.
Tests sample_count = 1, 5, 10 on edge tickers.
Usage: python sample_count_sweep.py RIVN
"""
import sys, os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
sys.path.append("C:/Users/Dream/Kronos")
from model import Kronos, KronosTokenizer, KronosPredictor

TICKER      = sys.argv[1].upper() if len(sys.argv) > 1 else "RIVN"
HORIZON     = 30
TEMPERATURE = 1.0
LOOKBACK    = 200
SELECTION_N = 600
VALIDATION_N= 400
SAMPLE_COUNTS = [5, 10]   # 20 too slow; 1 already done in hyperparam sweep

CHECKPOINT_DIR = "C:/Users/Dream/StockAI/sample_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CRYPTO = {"BTC", "SOL", "TAO", "ETH", "DOGE", "XRP", "COIN"}
yf_ticker = f"{TICKER}-USD" if TICKER.upper() in CRYPTO else TICKER
is_crypto = TICKER.upper() in CRYPTO

def next_dates(from_date, n):
    if is_crypto:
        return pd.date_range(start=from_date + timedelta(days=1), periods=n)
    return pd.bdate_range(start=from_date + timedelta(days=1), periods=n)

csv_path = f"C:/Users/Dream/StockAI/data/{TICKER}.csv"
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

max_possible = len(raw) - LOOKBACK - HORIZON
actual_n     = min(SELECTION_N + VALIDATION_N, max_possible)
test_start   = len(raw) - actual_n - HORIZON
print(f"Windows: {actual_n}  test_start: {test_start}")

print(f"Loading Kronos-base...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model_obj  = Kronos.from_pretrained("NeoQuasar/Kronos-base")
predictor  = KronosPredictor(model_obj, tokenizer, max_context=512)

# Load baseline (sample_count=1) from sweep checkpoints if available
sweep_ckpt = f"C:/Users/Dream/StockAI/sweep_checkpoints/{TICKER}_h{HORIZON}_t10_lb{LOOKBACK}.csv"
if os.path.exists(sweep_ckpt):
    base_df = pd.read_csv(sweep_ckpt, parse_dates=["date"])
    base_oos = base_df.iloc[-VALIDATION_N:]["correct"].mean() * 100
    print(f"\nBaseline (sample=1): {base_oos:.1f}% OOS")
else:
    print("\nBaseline not found — run hyperparam_sweep.py first")

all_results = {}

for sc in SAMPLE_COUNTS:
    cfg_key   = f"h{HORIZON}_t10_lb{LOOKBACK}_sc{sc}"
    ckpt_path = f"{CHECKPOINT_DIR}/{TICKER}_{cfg_key}.csv"

    if os.path.exists(ckpt_path):
        existing = pd.read_csv(ckpt_path, parse_dates=["date"])
        if len(existing) >= actual_n:
            print(f"  [{cfg_key}] already complete, loading...")
            all_results[cfg_key] = existing
            continue
        rows    = existing.to_dict("records")
        start_i = len(existing)
        print(f"  [{cfg_key}] resuming {start_i}/{actual_n}")
    else:
        rows    = []
        start_i = 0

    print(f"  [{cfg_key}] running {actual_n} windows (sample_count={sc})...")
    lb = min(LOOKBACK, 512)
    BATCH_SIZE = 4  # smaller batch for higher sample counts to avoid VRAM pressure
    batch_x_dfs, batch_x_ts_list, batch_y_ts_list, batch_meta = [], [], [], []

    def flush_batch():
        if not batch_x_dfs: return
        preds = predictor.predict_batch(
            df_list=batch_x_dfs, x_timestamp_list=batch_x_ts_list,
            y_timestamp_list=batch_y_ts_list, pred_len=HORIZON,
            T=TEMPERATURE, top_p=0.9, sample_count=sc, verbose=False)
        for j, (bidx, bentry_close, bentry_date) in enumerate(batch_meta):
            actual_idx = bidx + HORIZON
            if actual_idx >= len(raw): continue
            pred_close   = preds[j]["close"].iloc[HORIZON - 1]
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
        y_ts = pd.Series(next_dates(entry_date, HORIZON))

        batch_x_dfs.append(x_df); batch_x_ts_list.append(x_ts)
        batch_y_ts_list.append(y_ts); batch_meta.append((idx, entry_close, entry_date))

        if len(batch_x_dfs) >= BATCH_SIZE or i == actual_n - 1:
            flush_batch()

        if (i + 1) % 50 == 0:
            print(f"    [{cfg_key}] {i+1}/{actual_n}")
            pd.DataFrame(rows).to_csv(ckpt_path, index=False)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(ckpt_path, index=False)
    all_results[cfg_key] = df_out
    print(f"  [{cfg_key}] done.")

print(f"\n{'='*55}")
print(f"SAMPLE COUNT RESULTS: {TICKER} (base, h={HORIZON}, T={TEMPERATURE}, lb={LOOKBACK})")
print(f"{'='*55}")
if os.path.exists(sweep_ckpt):
    print(f"  sample=1 :  {base_oos:.1f}% OOS  (baseline)")
for sc in SAMPLE_COUNTS:
    cfg_key = f"h{HORIZON}_t10_lb{LOOKBACK}_sc{sc}"
    if cfg_key not in all_results: continue
    df = all_results[cfg_key]
    oos = df.iloc[-VALIDATION_N:]["correct"].mean() * 100
    sel = df.iloc[:SELECTION_N]["correct"].mean() * 100
    print(f"  sample={sc:<2}: {oos:.1f}% OOS  (sel={sel:.1f}%)")
