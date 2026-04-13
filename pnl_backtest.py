"""
P&L backtest: converts Kronos direction signals into actual trading returns.
Uses existing checkpoint data — no GPU needed.
Strategy: if Kronos predicts up -> long for H days, if down -> skip (long-only)
         or long/short version.
Applies 0.1% slippage each way.
Usage: python pnl_backtest.py [TICKER]
       python pnl_backtest.py  (runs all edge tickers)
"""
import sys, os
import pandas as pd
import numpy as np

CHECKPOINTS  = "C:/Users/Dream/StockAI/checkpoints"
MODELS       = ["mini", "small", "base"]
HORIZONS     = [1, 5, 10]
VALIDATION_N = 400
SLIPPAGE     = 0.001  # 0.1% each way
EDGE_TICKERS = ["ENVX", "RIVN", "TSLA", "COIN"]

args = sys.argv[1:]
LONG_SHORT = "--long-short" in args
args = [a for a in args if not a.startswith("--")]
tickers = [args[0].upper()] if args else EDGE_TICKERS

def simulate(signals_df, horizon, slippage=0.001, long_short=False):
    """
    signals_df: rows with entry_close, actual_close, pred_direction (True=up)
    Returns dict of strategy metrics.
    """
    portfolio = 1.0
    bah = 1.0  # buy and hold
    trades = []
    equity = [1.0]

    # Non-overlapping windows: step by horizon
    i = 0
    rows = signals_df.reset_index(drop=True)
    while i < len(rows) - 1:
        row = rows.iloc[i]
        entry = row["entry_close"] * (1 + slippage)
        exit_ = row["actual_close"] * (1 - slippage)
        raw_ret = exit_ / row["entry_close"] - 1  # actual price move

        if row["pred_direction"]:  # long
            trade_ret = (exit_ / entry) - 1
        elif long_short:           # short
            trade_ret = (entry / (row["actual_close"] * (1 + slippage))) - 1
        else:
            trade_ret = 0.0        # skip

        portfolio *= (1 + trade_ret)
        bah       *= (1 + raw_ret)
        equity.append(portfolio)
        trades.append(trade_ret)
        i += horizon  # non-overlapping

    if not trades:
        return None

    trades = np.array(trades)
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak
    max_dd = drawdowns.min() * 100

    # Annualized (assume 252 trading days, horizon in days)
    n_trades = len(trades)
    periods_per_year = 252 / horizon
    ann_ret = (portfolio ** (periods_per_year / n_trades) - 1) * 100 if n_trades > 0 else 0
    ann_vol = trades.std() * np.sqrt(periods_per_year) * 100 if n_trades > 1 else 0
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0

    return {
        "total_ret":  (portfolio - 1) * 100,
        "bah_ret":    (bah - 1) * 100,
        "ann_ret":    ann_ret,
        "sharpe":     sharpe,
        "max_dd":     max_dd,
        "n_trades":   n_trades,
        "win_rate":   (trades > 0).mean() * 100,
    }

mode_str = "long/short" if LONG_SHORT else "long-only"
print(f"{'='*80}")
print(f"P&L BACKTEST (0.1% slippage, {mode_str}, non-overlapping windows)")
print(f"OOS period only (last {VALIDATION_N} windows)")
print(f"{'='*80}")

for ticker in tickers:
    print(f"\n--- {ticker} ---")
    print(f"{'Model':<8} {'H':>4} {'TotRet':>8} {'B&H':>8} {'AnnRet':>8} {'Sharpe':>7} {'MaxDD':>7} {'Trades':>7}")
    print(f"{'-'*65}")

    best_sharpe = -999
    best_row = None

    for model in MODELS:
        path = f"{CHECKPOINTS}/{ticker}_{model}.csv"
        if not os.path.exists(path): continue
        df = pd.read_csv(path, parse_dates=["date"])

        for h in HORIZONS:
            sub = df[df["horizon"] == h].reset_index(drop=True)
            if len(sub) < VALIDATION_N + 50: continue
            oos = sub.iloc[-VALIDATION_N:].copy()
            oos["pred_direction"] = oos["pred_close"] > oos["entry_close"]

            r = simulate(oos, h, SLIPPAGE, long_short=LONG_SHORT)
            if r is None: continue

            flag = " *" if r["sharpe"] > best_sharpe else ""
            if r["sharpe"] > best_sharpe:
                best_sharpe = r["sharpe"]
                best_row = (model, h, r)

            print(f"{model:<8} {h:>4} {r['total_ret']:>7.1f}% {r['bah_ret']:>7.1f}% "
                  f"{r['ann_ret']:>7.1f}% {r['sharpe']:>6.2f} {r['max_dd']:>6.1f}% "
                  f"{r['n_trades']:>7}{flag}")

    if best_row:
        m, h, r = best_row
        print(f"\n  >> Best: {m} {h}-day  Sharpe={r['sharpe']:.2f}  "
              f"Total={r['total_ret']:.1f}%  vs B&H={r['bah_ret']:.1f}%")

    # Ensemble filter version for best horizon
    print(f"\n  Ensemble filter (agree-only, best horizons):")
    all_data = {}
    ok = True
    for model in MODELS:
        path = f"{CHECKPOINTS}/{ticker}_{model}.csv"
        if not os.path.exists(path): ok = False; break
        df_m = pd.read_csv(path, parse_dates=["date"])
        for h in HORIZONS:
            sub = df_m[df_m["horizon"] == h][["date","entry_close","pred_close","actual_close"]].copy()
            sub = sub.rename(columns={"pred_close": f"pred_{model}"})
            all_data[(model, h)] = sub

    if ok:
        for h in HORIZONS:
            base = all_data[("mini", h)][["date","entry_close","actual_close","pred_mini"]].copy()
            for model in ["small", "base"]:
                m_df = all_data[(model, h)][["date", f"pred_{model}"]]
                base = base.merge(m_df, on="date", how="inner")
            if len(base) < VALIDATION_N + 50: continue
            oos = base.iloc[-VALIDATION_N:].copy()
            for model in MODELS:
                oos[f"dir_{model}"] = oos[f"pred_{model}"] > oos["entry_close"]
            oos["all_agree"] = (oos["dir_mini"] == oos["dir_small"]) & (oos["dir_small"] == oos["dir_base"])
            oos["pred_direction"] = oos["dir_mini"]
            agree_oos = oos[oos["all_agree"]].copy()
            if len(agree_oos) < 20: continue
            r = simulate(agree_oos, h, SLIPPAGE, long_short=False)
            if r is None: continue
            print(f"  {h}-day agree-only ({r['n_trades']} trades):  "
                  f"Sharpe={r['sharpe']:.2f}  Total={r['total_ret']:.1f}%  "
                  f"vs B&H={r['bah_ret']:.1f}%  MaxDD={r['max_dd']:.1f}%")

print(f"\n{'='*80}")
print("Done.")
