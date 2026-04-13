"""
Exit Optimizer: sweep take-profit and stop-loss thresholds.
Uses NON-OVERLAPPING windows: enter -> wait for TP/SL/horizon -> enter next.
Same methodology as pnl_backtest.py.

Usage: python exit_optimizer.py
       python exit_optimizer.py RIVN
"""
import sys, os
import pandas as pd
import numpy as np
import yfinance as yf

TICKERS = sys.argv[1:] if len(sys.argv) > 1 else ["RIVN", "ENVX", "TSLA", "COIN"]

BEST_CONFIGS = {
    "RIVN": ("ext_checkpoints/RIVN_h40_t10_lb200.csv", 40),
    "ENVX": ("ext_checkpoints/ENVX_h90_t10_lb200.csv", 90),
    "TSLA": ("ext_checkpoints/TSLA_h90_t10_lb200.csv", 90),
    "COIN": ("ext_checkpoints/COIN_h60_t10_lb200.csv", 60),
}

VALIDATION_N = 400
BASE = "C:/Users/Dream/StockAI"
TAKE_PROFITS = [0.02, 0.05, 0.10, 0.15, 0.20]
STOP_LOSSES  = [0.02, 0.05, 0.10]

def fetch_ohlc(ticker):
    csv_path = os.path.join(BASE, "data", f"{ticker}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["timestamps"])
        df = df.rename(columns={"timestamps": "date"})
    else:
        raw = yf.download(ticker, period="5y", interval="1d",
                          auto_adjust=True, progress=False, timeout=30)
        raw = raw.reset_index()
        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                       for c in raw.columns]
        date_col = "date" if "date" in raw.columns else "datetime"
        df = raw.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.set_index("date")[["open","high","low","close"]].sort_index()


def simulate_non_overlapping(oos_df, ohlc, horizon, tp, sl):
    """
    Walk through OOS windows in order.
    After each trade exits, find the next OOS entry that starts on/after exit date.
    Returns list of trade returns.
    """
    # Sort OOS by date, build lookup
    oos_sorted = oos_df.sort_values("date").reset_index(drop=True)
    oos_sorted["direction"] = np.where(
        oos_sorted["pred_close"] > oos_sorted["entry_close"], "UP", "DOWN")

    trades = []
    i = 0
    while i < len(oos_sorted):
        row       = oos_sorted.iloc[i]
        entry_date = row["date"]
        direction  = row["direction"]
        entry_close = row["entry_close"]

        # Get future price path
        try:
            future = ohlc.loc[entry_date:].iloc[1:horizon+1]
        except Exception:
            i += 1
            continue

        if len(future) == 0:
            i += 1
            continue

        exit_date   = future.index[-1]   # default: hold to end
        trade_ret   = 0.0
        exit_reason = "HOLD"

        for date, row_p in future.iterrows():
            close = row_p["close"]
            ret   = (close - entry_close) / entry_close
            if direction == "UP":
                if ret >= tp:
                    trade_ret   = tp
                    exit_date   = date
                    exit_reason = "TP"
                    break
                if ret <= -sl:
                    trade_ret   = -sl
                    exit_date   = date
                    exit_reason = "SL"
                    break
            else:  # SHORT
                ret_s = -ret
                if ret_s >= tp:
                    trade_ret   = tp
                    exit_date   = date
                    exit_reason = "TP"
                    break
                if ret_s <= -sl:
                    trade_ret   = -sl
                    exit_date   = date
                    exit_reason = "SL"
                    break
        else:
            # held full horizon
            final_close = future["close"].iloc[-1]
            trade_ret   = (final_close - entry_close) / entry_close
            if direction == "DOWN":
                trade_ret = -trade_ret

        trades.append((trade_ret, exit_date, exit_reason))

        # advance to next OOS entry on or after exit_date
        later = oos_sorted[oos_sorted["date"] > exit_date]
        if later.empty:
            break
        i = later.index[0]

    return trades


def stats(trades):
    if not trades:
        return None
    rets = np.array([t[0] for t in trades])
    mean = rets.mean()
    std  = rets.std()
    sharpe = (mean / std * np.sqrt(252)) if std > 0 else 0.0
    total  = float((1 + rets).prod() - 1)
    cum = np.cumprod(1 + rets)
    roll_max = np.maximum.accumulate(cum)
    max_dd = float(((cum - roll_max) / roll_max).min())
    win_rate = float((rets > 0).mean() * 100)
    tp_hits = sum(1 for t in trades if t[2] == "TP")
    sl_hits = sum(1 for t in trades if t[2] == "SL")
    return {
        "n_trades": len(trades), "win_rate": round(win_rate, 1),
        "avg_ret": round(mean * 100, 2), "total_ret": round(total * 100, 1),
        "sharpe": round(sharpe, 2), "max_dd": round(max_dd * 100, 1),
        "tp_hits": tp_hits, "sl_hits": sl_hits,
    }


print("EXIT OPTIMIZER (non-overlapping windows)")
print("=" * 72)

all_best = {}

for ticker in TICKERS:
    if ticker not in BEST_CONFIGS:
        print(f"No config for {ticker}, skipping.")
        continue

    ckpt_rel, horizon = BEST_CONFIGS[ticker]
    ckpt_full = os.path.join(BASE, ckpt_rel)
    if not os.path.exists(ckpt_full):
        print(f"Missing: {ckpt_full}")
        continue

    print(f"\n{ticker}  (h={horizon})")
    print("-" * 62)

    ckpt = pd.read_csv(ckpt_full)
    ckpt["date"] = pd.to_datetime(ckpt["date"]).dt.normalize()
    oos  = ckpt.tail(VALIDATION_N).copy()
    ohlc = fetch_ohlc(ticker)

    # Baseline: non-overlapping, hold full horizon
    baseline_trades = simulate_non_overlapping(oos, ohlc, horizon, tp=999, sl=999)
    b = stats(baseline_trades)
    print(f"  Baseline hold (no TP/SL): n={b['n_trades']}  "
          f"Sharpe {b['sharpe']:.2f}  Total {b['total_ret']:.1f}%  "
          f"Win% {b['win_rate']:.1f}%  MaxDD {b['max_dd']:.1f}%")

    # Sweep
    results = []
    for tp in TAKE_PROFITS:
        for sl in STOP_LOSSES:
            trades = simulate_non_overlapping(oos, ohlc, horizon, tp, sl)
            s = stats(trades)
            if s:
                s["tp"] = tp
                s["sl"] = sl
                results.append(s)

    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)

    print(f"\n  Top 5 by Sharpe:")
    print(f"  {'TP':>5} {'SL':>5} {'n':>5} {'Sharpe':>7} {'Total%':>8} {'Win%':>6} {'MaxDD%':>8}")
    for _, r in df.head(5).iterrows():
        print(f"  {r['tp']*100:>4.0f}% {r['sl']*100:>4.0f}%  "
              f"{r['n_trades']:>4}  {r['sharpe']:>7.2f}  "
              f"{r['total_ret']:>7.1f}%  {r['win_rate']:>5.1f}%  {r['max_dd']:>7.1f}%")

    best = df.iloc[0]
    all_best[ticker] = best
    print(f"\n  Best: TP={best['tp']*100:.0f}%  SL={best['sl']*100:.0f}%  "
          f"Sharpe={best['sharpe']:.2f}  Total={best['total_ret']:.1f}%")

    out = os.path.join(BASE, f"exit_optimizer_{ticker}.csv")
    df.to_csv(out, index=False)

print("\n" + "=" * 72)
print("SUMMARY")
print(f"{'Ticker':<8} {'TP':>5} {'SL':>5} {'n':>5} {'Sharpe':>8} {'Total%':>10} {'Win%':>7} {'MaxDD%':>8}")
print("-" * 60)
for ticker, best in all_best.items():
    _, horizon = BEST_CONFIGS[ticker]
    print(f"{ticker:<8} {best['tp']*100:>4.0f}% {best['sl']*100:>4.0f}%  "
          f"{best['n_trades']:>4}  {best['sharpe']:>7.2f}  "
          f"{best['total_ret']:>9.1f}%  {best['win_rate']:>6.1f}%  {best['max_dd']:>7.1f}%")
