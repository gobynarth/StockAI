"""
Ensemble filter: analyze existing checkpoints to find windows where
mini + small + base all agree on direction. Reports accuracy on:
  - All signals
  - Agree-only signals (high confidence)
Usage: python ensemble_filter.py [TICKER]
       python ensemble_filter.py  (runs all edge tickers)
"""
import sys, os
import pandas as pd
import numpy as np

CHECKPOINTS  = "C:/Users/Dream/StockAI/checkpoints"
MODELS       = ["mini", "small", "base"]
HORIZONS     = [1, 5, 10]
SELECTION_N  = 600
VALIDATION_N = 400
EDGE_TICKERS = ["ENVX", "RIVN", "TSLA", "COIN"]

tickers = [sys.argv[1].upper()] if len(sys.argv) > 1 else EDGE_TICKERS

print(f"{'='*72}")
print(f"ENSEMBLE FILTER ANALYSIS")
print(f"{'='*72}")

summary_rows = []

for ticker in tickers:
    # Load all models x horizons
    data = {}
    ok = True
    for model in MODELS:
        path = f"{CHECKPOINTS}/{ticker}_{model}.csv"
        if not os.path.exists(path):
            print(f"\n[{ticker}] missing {model} checkpoint, skipping")
            ok = False; break
        df = pd.read_csv(path, parse_dates=["date"])
        for h in HORIZONS:
            sub = df[df["horizon"] == h][["date","entry_close","pred_close","actual_close","correct"]].copy()
            sub = sub.rename(columns={
                "pred_close": f"pred_{model}",
                "correct":    f"correct_{model}",
            })
            data[(model, h)] = sub
    if not ok:
        continue

    print(f"\n{'-'*72}")
    print(f"  {ticker}")
    print(f"{'-'*72}")
    print(f"  {'Horizon':<10} {'All N':>6} {'All Acc':>8} {'Agree N':>8} {'Agree%':>7} {'Agree Acc':>10} {'Lift':>6}")
    print(f"  {'-'*65}")

    for h in HORIZONS:
        # Merge all 3 models on date
        base_df = data[("mini", h)][["date","entry_close","actual_close","correct_mini","pred_mini"]].copy()
        for model in ["small", "base"]:
            m_df = data[(model, h)][["date", f"pred_{model}", f"correct_{model}"]]
            base_df = base_df.merge(m_df, on="date", how="inner")

        if len(base_df) < SELECTION_N + 50:
            continue

        # Determine predicted direction for each model
        for model in MODELS:
            base_df[f"dir_{model}"] = base_df[f"pred_{model}"] > base_df["entry_close"]

        # Consensus: all 3 agree
        base_df["all_agree"] = (
            (base_df["dir_mini"] == base_df["dir_small"]) &
            (base_df["dir_small"] == base_df["dir_base"])
        )
        base_df["actual_dir"] = base_df["actual_close"] > base_df["entry_close"]
        base_df["ensemble_correct"] = base_df["dir_mini"] == base_df["actual_dir"]  # same for all when agree

        # OOS slice (last VALIDATION_N)
        oos = base_df.iloc[-VALIDATION_N:].copy()
        agree_oos = oos[oos["all_agree"]]

        all_acc   = oos["correct_mini"].mean() * 100  # use any model for baseline
        # average of all 3 for "all signals" accuracy
        all_acc   = oos[["correct_mini","correct_small","correct_base"]].mean(axis=1).mean() * 100
        agree_acc = agree_oos["ensemble_correct"].mean() * 100 if len(agree_oos) > 0 else 0
        agree_pct = len(agree_oos) / len(oos) * 100
        lift      = agree_acc - all_acc

        print(f"  {h}-day{'':<6} {len(oos):>6} {all_acc:>7.1f}% {len(agree_oos):>8} {agree_pct:>6.1f}% {agree_acc:>9.1f}% {lift:>+5.1f}%")

        summary_rows.append({
            "ticker": ticker, "horizon": h,
            "all_n": len(oos), "all_acc": all_acc,
            "agree_n": len(agree_oos), "agree_pct": agree_pct,
            "agree_acc": agree_acc, "lift": lift,
        })

    # Also show best horizon by agree_acc for this ticker
    ticker_rows = [r for r in summary_rows if r["ticker"] == ticker and r["agree_n"] >= 50]
    if ticker_rows:
        best = max(ticker_rows, key=lambda x: x["agree_acc"])
        print(f"\n  >> Best: {best['horizon']}-day agree-only  {best['agree_acc']:.1f}% on {best['agree_n']} signals ({best['agree_pct']:.0f}% of windows)")

# ---- Overall summary ----
if summary_rows:
    print(f"\n{'='*72}")
    print(f"SUMMARY: Best agree-only configs per ticker")
    print(f"{'='*72}")
    print(f"{'Ticker':<8} {'Horizon':<10} {'All Acc':>8} {'Agree N':>8} {'Agree%':>7} {'Agree Acc':>10} {'Lift':>6}")
    print(f"{'-'*60}")
    for ticker in tickers:
        rows = [r for r in summary_rows if r["ticker"] == ticker and r["agree_n"] >= 50]
        if not rows: continue
        best = max(rows, key=lambda x: x["agree_acc"])
        print(f"{ticker:<8} {best['horizon']}-day{'':<6} {best['all_acc']:>7.1f}% {best['agree_n']:>8} {best['agree_pct']:>6.1f}% {best['agree_acc']:>9.1f}% {best['lift']:>+5.1f}%")
