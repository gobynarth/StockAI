"""
Compare finetuned vs zero-shot Kronos on OOS windows.
Runs both models on identical data, prints accuracy side-by-side.
"""
import sys, os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
sys.path.insert(0, "/workspace/Kronos")
from model import Kronos, KronosTokenizer, KronosPredictor

TICKERS = {
    "RIVN": {"horizon": 40, "lookback": 200, "temp": 1.0, "ft_dir": "RIVN_h40"},
    "ENVX": {"horizon": 90, "lookback": 200, "temp": 1.0, "ft_dir": "ENVX_h90"},
    "TSLA": {"horizon": 90, "lookback": 200, "temp": 0.5, "ft_dir": "TSLA_h90"},
    "COIN": {"horizon": 60, "lookback": 200, "temp": 1.0, "ft_dir": "COIN_h60"},
}

VALIDATION_N = 400
BATCH_SIZE = 16
FT_BASE = "/workspace/StockAI/finetuned"
CKPT_DIR = "/workspace/StockAI/ft_comparison"
os.makedirs(CKPT_DIR, exist_ok=True)

def next_dates(from_date, n):
    return pd.bdate_range(start=from_date + timedelta(days=1), periods=n)

def run_backtest(predictor, raw, horizon, lookback, temp, n_windows):
    lb = min(lookback, 512)
    max_possible = len(raw) - lb - horizon
    actual_n = min(n_windows, max_possible)
    test_start = len(raw) - actual_n - horizon

    rows = []
    batch_x_dfs, batch_x_ts_list, batch_y_ts_list, batch_meta = [], [], [], []

    def flush_batch():
        if not batch_x_dfs:
            return
        preds = predictor.predict_batch(
            df_list=batch_x_dfs, x_timestamp_list=batch_x_ts_list,
            y_timestamp_list=batch_y_ts_list, pred_len=horizon,
            T=temp, top_p=0.9, sample_count=1, verbose=False)
        for j, (bidx, bentry_close, bentry_date) in enumerate(batch_meta):
            actual_idx = bidx + horizon
            if actual_idx >= len(raw):
                continue
            pred_close = preds[j]["close"].iloc[horizon - 1]
            actual_close = raw.iloc[actual_idx]["close"]
            rows.append({
                "date": bentry_date, "entry_close": bentry_close,
                "pred_close": pred_close, "actual_close": actual_close,
                "correct": (pred_close > bentry_close) == (actual_close > bentry_close),
            })
        batch_x_dfs.clear(); batch_x_ts_list.clear()
        batch_y_ts_list.clear(); batch_meta.clear()

    for i in range(actual_n):
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
    return pd.DataFrame(rows)

results = {}

for ticker, cfg in TICKERS.items():
    print(f"\n{'='*60}")
    print(f"{ticker} (h={cfg['horizon']}, lb={cfg['lookback']}, T={cfg['temp']})")
    print(f"{'='*60}")

    csv_path = f"/workspace/StockAI/data/{ticker}.csv"
    raw = pd.read_csv(csv_path, parse_dates=["timestamps"])
    raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)
    raw = raw[["timestamps","open","high","low","close","volume","amount"]].dropna()
    print(f"  Loaded {len(raw)} rows")

    ft_tok_path = f"{FT_BASE}/{cfg['ft_dir']}/tokenizer/best_model"
    ft_mod_path = f"{FT_BASE}/{cfg['ft_dir']}/basemodel/best_model"

    # --- Zero-shot ---
    ckpt_zs = f"{CKPT_DIR}/{ticker}_zeroshot.csv"
    if os.path.exists(ckpt_zs):
        df_zs = pd.read_csv(ckpt_zs, parse_dates=["date"])
        print(f"  [zero-shot] loaded from checkpoint ({len(df_zs)} windows)")
    else:
        print(f"  [zero-shot] running...")
        tok_zs = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        mod_zs = Kronos.from_pretrained("NeoQuasar/Kronos-base")
        pred_zs = KronosPredictor(mod_zs, tok_zs, max_context=512)
        df_zs = run_backtest(pred_zs, raw, cfg["horizon"], cfg["lookback"], cfg["temp"], VALIDATION_N)
        df_zs.to_csv(ckpt_zs, index=False)
        del tok_zs, mod_zs, pred_zs
        print(f"  [zero-shot] done ({len(df_zs)} windows)")

    # --- Finetuned ---
    ckpt_ft = f"{CKPT_DIR}/{ticker}_finetuned.csv"
    if os.path.exists(ckpt_ft):
        df_ft = pd.read_csv(ckpt_ft, parse_dates=["date"])
        print(f"  [finetuned] loaded from checkpoint ({len(df_ft)} windows)")
    else:
        print(f"  [finetuned] running...")
        tok_ft = KronosTokenizer.from_pretrained(ft_tok_path)
        mod_ft = Kronos.from_pretrained(ft_mod_path)
        pred_ft = KronosPredictor(mod_ft, tok_ft, max_context=512)
        df_ft = run_backtest(pred_ft, raw, cfg["horizon"], cfg["lookback"], cfg["temp"], VALIDATION_N)
        df_ft.to_csv(ckpt_ft, index=False)
        del tok_ft, mod_ft, pred_ft
        print(f"  [finetuned] done ({len(df_ft)} windows)")

    zs_acc = df_zs["correct"].mean() * 100
    ft_acc = df_ft["correct"].mean() * 100
    delta = ft_acc - zs_acc
    results[ticker] = {"zeroshot": zs_acc, "finetuned": ft_acc, "delta": delta}
    print(f"  Zero-shot: {zs_acc:.1f}%  |  Finetuned: {ft_acc:.1f}%  |  Delta: {delta:+.1f}%")

print(f"\n\n{'='*60}")
print(f"FINAL COMPARISON: Finetuned vs Zero-Shot")
print(f"{'='*60}")
print(f"{'Ticker':<8} {'Zero-Shot':>10} {'Finetuned':>10} {'Delta':>8}  Verdict")
print(f"{'-'*50}")
for ticker, r in results.items():
    v = "FT WINS" if r["delta"] > 2 else ("TIE" if abs(r["delta"]) <= 2 else "ZS WINS")
    print(f"{ticker:<8} {r['zeroshot']:>9.1f}% {r['finetuned']:>9.1f}% {r['delta']:>+7.1f}%  {v}")
