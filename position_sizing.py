"""
Kelly criterion position sizing for Kronos strategy.
Uses exit optimizer results (win rate + TP/SL) to compute optimal bet size.

Outputs:
  - Per-ticker Kelly fraction (full, half, quarter)
  - Simulated equity curves at each Kelly level on validation data
  - Multi-position portfolio sizing (concurrent open trades)
  - Sensitivity table: what if win rate is 5% lower than backtest?

Usage: python position_sizing.py
"""
import os
import pandas as pd
import numpy as np

BASE = "C:/Users/Dream/Projects/StockAI"
EXT  = os.path.join(BASE, "ext_checkpoints")

# Best confirmed configs per ticker (from exit optimizer + extended horizons)
CONFIGS = {
    "RIVN": {"horizon": 40, "ckpt": "RIVN_h40_t10_lb200.csv", "tp": 0.15, "sl": 0.02},
    "COIN": {"horizon": 60, "ckpt": "COIN_h60_t10_lb200.csv", "tp": 0.20, "sl": 0.10},
    "ENVX": {"horizon": 90, "ckpt": "ENVX_h90_t10_lb200.csv", "tp": 0.15, "sl": 0.02},
    "TSLA": {"horizon": 90, "ckpt": "TSLA_h90_t10_lb200.csv", "tp": 0.20, "sl": 0.02},
}

VALIDATION_N = 400
SLIPPAGE     = 0.001


def load_val(ckpt_file):
    path = os.path.join(EXT, ckpt_file)
    df = pd.read_csv(path, parse_dates=["date"])
    df["pred_direction"] = df["correct"].astype(bool) == (df["pred_close"] > df["entry_close"])
    # Recompute pred_direction cleanly
    df["pred_up"]    = df["pred_close"] > df["entry_close"]
    df["actual_up"]  = df["actual_close"] > df["entry_close"]
    df["correct"]    = df["pred_up"] == df["actual_up"]
    return df.iloc[-VALIDATION_N:].reset_index(drop=True)


def kelly_fraction(win_rate, tp, sl):
    """Standard Kelly: f* = (p*b - q) / b where b = avg_win/avg_loss."""
    p = win_rate
    q = 1 - p
    b = tp / sl        # payoff ratio (e.g. TP15/SL2 = 7.5x)
    f = (p * b - q) / b
    return max(0.0, f)


def simulate_kelly(df, horizon, tp, sl, kelly_f, slippage=0.001):
    """
    Simulate non-overlapping trades with TP/SL exits and Kelly position sizing.
    kelly_f IS the fraction of portfolio to allocate per trade.
    Kelly already accounts for TP/SL — no extra division needed.
    """
    portfolio = 1.0
    equity = [1.0]
    trades = []

    rows = df.reset_index(drop=True)
    i = 0
    while i < len(rows) - 1:
        row = rows.iloc[i]
        entry  = row["entry_close"] * (1 + slippage)
        actual = row["actual_close"]

        if row["pred_up"]:  # LONG
            raw_ret = (actual - entry) / entry
            if raw_ret >= tp:
                trade_ret = tp      # TP hit
            elif raw_ret <= -sl:
                trade_ret = -sl     # SL hit
            else:
                trade_ret = raw_ret # held to horizon
        else:
            trade_ret = 0.0  # skip (long-only)

        # Kelly fraction = fraction of portfolio allocated to this position
        portfolio_ret  = trade_ret * kelly_f
        portfolio     *= (1 + portfolio_ret)
        equity.append(portfolio)
        trades.append(portfolio_ret)
        i += horizon

    if not trades:
        return None

    trades  = np.array(trades)
    equity  = np.array(equity)
    peak    = np.maximum.accumulate(equity)
    dd      = (equity - peak) / peak
    n       = len(trades)
    active  = trades[trades != 0]
    n_years = (n * horizon) / 252

    return {
        "n_trades":  n,
        "n_active":  len(active),
        "win_rate":  (active > 0).mean() * 100 if len(active) else 0,
        "total_ret": (portfolio - 1) * 100,
        "cagr":      ((portfolio ** (1 / n_years)) - 1) * 100 if n_years > 0 else 0,
        "sharpe":    active.mean() / active.std() * np.sqrt(252 / horizon) if len(active) > 1 and active.std() > 0 else 0,
        "max_dd":    dd.min() * 100,
        "equity":    equity,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*75}")
print("KELLY CRITERION POSITION SIZING")
print(f"{'='*75}")

results = {}

for ticker, cfg in CONFIGS.items():
    df  = load_val(cfg["ckpt"])
    tp, sl, h = cfg["tp"], cfg["sl"], cfg["horizon"]

    # Use EXIT OPTIMIZER win rate (TP/SL trade outcomes), NOT directional accuracy.
    # Directional accuracy (82% RIVN) != TP/SL win rate (32.6% RIVN).
    EXIT_WIN_RATES = {"RIVN": 0.326, "COIN": 0.524, "ENVX": 0.272, "TSLA": 0.244}
    win_rate = EXIT_WIN_RATES[ticker]

    full_k  = kelly_fraction(win_rate, tp, sl)
    half_k  = full_k / 2
    qtr_k   = full_k / 4

    print(f"\n{'─'*60}")
    print(f"  {ticker}  |  h={h}d  |  TP={tp*100:.0f}%  SL={sl*100:.0f}%  |  "
          f"Payoff ratio {tp/sl:.1f}x")
    print(f"  Exit optimizer win rate (TP/SL): {win_rate*100:.1f}%")
    print(f"  Full Kelly:    {full_k*100:.1f}%  of portfolio risked per trade")
    print(f"  Half Kelly:    {half_k*100:.1f}%")
    print(f"  Quarter Kelly: {qtr_k*100:.1f}%")

    print(f"\n  {'Kelly':>14} {'N trades':>9} {'Win%':>7} {'Total%':>8} {'CAGR%':>7} {'Sharpe':>7} {'MaxDD%':>7}")
    print(f"  {'-'*65}")

    ticker_res = {}
    # Use all data for simulation (selection + validation) but Kelly params from validation
    df_full = pd.read_csv(os.path.join(EXT, cfg["ckpt"]), parse_dates=["date"])
    df_full["pred_up"]   = df_full["pred_close"]   > df_full["entry_close"]
    df_full["actual_up"] = df_full["actual_close"]  > df_full["entry_close"]

    for label, kf in [("Full Kelly", full_k), ("Half Kelly", half_k), ("Qtr Kelly", qtr_k), ("5% fixed", 0.05)]:
        if kf <= 0:
            print(f"  {label:>14}  No edge (Kelly <= 0)")
            continue
        r = simulate_kelly(df_full, h, tp, sl, kf)
        if r:
            print(f"  {label:>14} {r['n_active']:>9} {r['win_rate']:>6.1f}% "
                  f"{r['total_ret']:>7.1f}% {r['cagr']:>6.1f}% "
                  f"{r['sharpe']:>6.2f}  {r['max_dd']:>6.1f}%")
            ticker_res[label] = r

    results[ticker] = {"cfg": cfg, "win_rate": win_rate, "full_k": full_k,
                       "half_k": half_k, "qtr_k": qtr_k, "sims": ticker_res}

# ── Sensitivity: what if win rate is 5% worse? ────────────────────────────────
print(f"\n\n{'='*75}")
print("SENSITIVITY: Win rate −5% (model overfit / regime change)")
print(f"{'='*75}")
print(f"  {'Ticker':>6} {'Real WR':>8} {'Stress WR':>10} {'Full K (real)':>14} {'Full K (stress)':>16} {'Still edge?':>12}")
print(f"  {'-'*70}")
for ticker, r in results.items():
    cfg = r["cfg"]
    wrs = r["win_rate"] - 0.05
    fks = kelly_fraction(wrs, cfg["tp"], cfg["sl"])
    edge = "YES" if fks > 0.02 else ("THIN" if fks > 0 else "NO")
    print(f"  {ticker:>6} {r['win_rate']*100:>7.1f}% {wrs*100:>9.1f}% "
          f"{r['full_k']*100:>13.1f}% {fks*100:>15.1f}%  {edge:>12}")

# ── Multi-position portfolio sizing ───────────────────────────────────────────
print(f"\n\n{'='*75}")
print("MULTI-POSITION PORTFOLIO SIZING")
print("(When multiple tickers are open simultaneously)")
print(f"{'='*75}")

tickers_open = list(CONFIGS.keys())
n_concurrent = len(tickers_open)

print(f"\n  Assumption: up to {n_concurrent} positions open at once.")
print(f"  Strategy: allocate half-Kelly per ticker, scaled by 1/n_concurrent\n")
print(f"  {'Ticker':>6} {'Half-K':>8} {'Scaled (÷{:d})'.format(n_concurrent):>12} {'Position size':>14} {'Max loss/trade':>15}")
print(f"  {'-'*60}")

total_risk = 0
for ticker, r in results.items():
    sl       = CONFIGS[ticker]["sl"]
    hk       = r["half_k"]
    scaled   = hk / n_concurrent          # scale down for concurrent positions
    max_loss   = scaled * sl * 100        # % portfolio loss if SL hits (scaled IS the position size)
    total_risk += max_loss
    print(f"  {ticker:>6} {hk*100:>7.1f}% {scaled*100:>11.1f}% {scaled*100:>13.1f}%  {max_loss:>13.2f}%")

print(f"\n  Worst case (all 4 SL same day): {total_risk:.1f}% portfolio loss")
print(f"  Recommended: use half-Kelly scaled, review after every 20 closed trades.")

print(f"\n{'='*75}")
print("RECOMMENDATION")
print(f"{'='*75}")
for ticker, r in results.items():
    hk      = r["half_k"]
    scaled  = hk / n_concurrent
    sl      = CONFIGS[ticker]["sl"]
    max_loss = scaled * sl * 100
    print(f"  {ticker}: allocate {scaled*100:.0f}% of portfolio per trade  "
          f"| max loss if SL hits: {max_loss:.1f}%  "
          f"(half-Kelly / {n_concurrent} concurrent)")
