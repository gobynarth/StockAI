"""
P&L backtest using extended-horizon checkpoints (h=40/60/90).
No GPU needed — pure analysis on existing checkpoint data.
"""
import sys, os
import pandas as pd
import numpy as np

BASE = "/workspace/StockAI"
VALIDATION_N = 400
SLIPPAGE = 0.001

BEST_CONFIGS = {
    "RIVN": ("ext_checkpoints/RIVN_h40_t10_lb200.csv", 40),
    "ENVX": ("ext_checkpoints/ENVX_h90_t10_lb200.csv", 90),
    "TSLA": ("ext_checkpoints/TSLA_h90_t10_lb200.csv", 90),
    "COIN": ("ext_checkpoints/COIN_h60_t10_lb200.csv", 60),
}

def simulate(signals_df, horizon, slippage=0.001, long_short=False):
    portfolio = 1.0
    bah = 1.0
    trades = []
    equity = [1.0]
    i = 0
    rows = signals_df.reset_index(drop=True)
    while i < len(rows) - 1:
        row = rows.iloc[i]
        entry = row["entry_close"] * (1 + slippage)
        exit_ = row["actual_close"] * (1 - slippage)
        raw_ret = exit_ / row["entry_close"] - 1
        if row["pred_direction"]:
            trade_ret = (exit_ / entry) - 1
        elif long_short:
            trade_ret = (entry / (row["actual_close"] * (1 + slippage))) - 1
        else:
            trade_ret = 0.0
        portfolio *= (1 + trade_ret)
        bah *= (1 + raw_ret)
        equity.append(portfolio)
        trades.append(trade_ret)
        i += horizon
    if not trades:
        return None
    trades = np.array(trades)
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak
    max_dd = drawdowns.min() * 100
    n_trades = len(trades)
    periods_per_year = 252 / horizon
    ann_ret = (portfolio ** (periods_per_year / n_trades) - 1) * 100 if n_trades > 0 else 0
    ann_vol = trades.std() * np.sqrt(periods_per_year) * 100 if n_trades > 1 else 0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    return {
        "total_ret": (portfolio - 1) * 100, "bah_ret": (bah - 1) * 100,
        "ann_ret": ann_ret, "sharpe": sharpe, "max_dd": max_dd,
        "n_trades": n_trades, "win_rate": (trades > 0).mean() * 100,
    }

for mode in ["long-only", "long/short"]:
    ls = mode == "long/short"
    print(f"\n{'='*70}")
    print(f"P&L BACKTEST -- EXT HORIZONS ({mode}, 0.1% slippage)")
    print(f"{'='*70}")
    print(f"{'Ticker':<8} {'Config':<22} {'TotRet':>8} {'B&H':>8} {'Sharpe':>7} {'MaxDD':>7} {'Trades':>7} {'Win%':>6}")
    print("-" * 75)

    for ticker, (ckpt_rel, horizon) in BEST_CONFIGS.items():
        ckpt_path = os.path.join(BASE, ckpt_rel)
        if not os.path.exists(ckpt_path):
            print(f"{ticker}: missing {ckpt_path}")
            continue
        df = pd.read_csv(ckpt_path)
        if len(df) < VALIDATION_N:
            print(f"{ticker}: only {len(df)} rows, need {VALIDATION_N}")
            continue
        oos = df.tail(VALIDATION_N).copy()
        oos["pred_direction"] = oos["pred_close"] > oos["entry_close"]
        r = simulate(oos, horizon, SLIPPAGE, long_short=ls)
        if r:
            print(f"{ticker:<8} h={horizon} T=1.0 lb=200    {r['total_ret']:>7.1f}% {r['bah_ret']:>7.1f}% "
                  f"{r['sharpe']:>6.2f} {r['max_dd']:>6.1f}% {r['n_trades']:>7} {r['win_rate']:>5.1f}%")
