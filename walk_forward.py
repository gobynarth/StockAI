"""
Walk-forward validation: rolling 6-month test windows across the full checkpoint.
Confirms edge is stable over time, not concentrated in one lucky period.
No GPU needed — pure analysis on existing checkpoint CSVs.
"""
import os
import pandas as pd
import numpy as np
from datetime import timedelta

EXT = "C:/Users/Dream/Projects/StockAI/ext_checkpoints"

CONFIGS = {
    "RIVN": ("RIVN_h40_t10_lb200.csv", 40),
    "COIN": ("COIN_h60_t10_lb200.csv", 60),
    "ENVX": ("ENVX_h90_t10_lb200.csv", 90),
    "TSLA": ("TSLA_h90_t10_lb200.csv", 90),
}

WINDOW_DAYS = 126   # ~6 months of trading days per test window
STEP_DAYS   = 63    # step 3 months each iteration
MIN_WINDOW  = 30

print(f"{'='*70}")
print("WALK-FORWARD VALIDATION  (6-month windows, 3-month step)")
print(f"{'='*70}")

for ticker, (fname, horizon) in CONFIGS.items():
    df = pd.read_csv(os.path.join(EXT, fname), parse_dates=["date"])
    df["pred_up"]   = df["pred_close"]  > df["entry_close"]
    df["actual_up"] = df["actual_close"] > df["entry_close"]
    df["correct"]   = df["pred_up"] == df["actual_up"]
    df = df.sort_values("date").reset_index(drop=True)

    dates = df["date"]
    start, end = dates.iloc[0], dates.iloc[-1]

    print(f"\n--- {ticker}  ({start.date()} to {end.date()}) ---")
    print(f"  {'Window':<28} {'N':>5} {'Acc':>7} {'LongWR':>9} {'LongN':>7}  {'Edge?':>6}")
    print(f"  {'-'*62}")

    accs, long_wrs = [], []
    cur = start
    while cur + timedelta(days=WINDOW_DAYS * 1.5) <= end + timedelta(days=1):
        win_end = cur + timedelta(days=WINDOW_DAYS)
        sub     = df[(dates >= cur) & (dates < win_end)]
        if len(sub) < MIN_WINDOW:
            cur += timedelta(days=STEP_DAYS)
            continue

        acc    = sub["correct"].mean() * 100
        longs  = sub[sub["pred_up"]]
        long_wr = longs["actual_up"].mean() * 100 if len(longs) > 0 else float("nan")
        edge   = "YES" if acc > 55 else ("WEAK" if acc > 52 else "NO")
        lwr_s  = f"{long_wr:>8.1f}%" if not np.isnan(long_wr) else "     n/a"
        label  = f"{cur.date()} - {min(win_end, end).date()}"
        print(f"  {label:<28} {len(sub):>5} {acc:>6.1f}% {lwr_s} {len(longs):>7}  {edge:>6}")

        accs.append(acc)
        if not np.isnan(long_wr):
            long_wrs.append(long_wr)
        cur += timedelta(days=STEP_DAYS)

    if accs:
        pos = sum(a > 55 for a in accs)
        print(f"\n  avg acc={np.mean(accs):.1f}%  |  avg long WR={np.mean(long_wrs):.1f}%  |  "
              f"edge in {pos}/{len(accs)} windows ({pos/len(accs)*100:.0f}%)")
