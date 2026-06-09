"""
Trailing stop backtest — compare fixed TP/SL vs trailing variants.

Loads daily OHLC + checkpoint entry dates, simulates N-day trades with
different exit strategies:
  A. FIXED       - current baseline (TP 15%, SL 2%)
  B. BE_MOVE     - at +5% move SL to entry (breakeven)
  C. TRAIL_3     - trail SL at peak - 3%
  D. TRAIL_5     - trail SL at peak - 5%
  E. TRAIL_7     - trail SL at peak - 7%
  F. BE_THEN_5   - at +5% move SL to entry, then trail at peak - 5%

Run: python trailing_stop_backtest.py RIVN
"""
import sys, os
import pandas as pd
import numpy as np
import yfinance as yf

TICKERS = {
    "RIVN": {"horizon": 40, "tp": 0.15, "sl": 0.02},
    "ENVX": {"horizon": 90, "tp": 0.15, "sl": 0.02},
    "TSLA": {"horizon": 90, "tp": 0.20, "sl": 0.02},
    "COIN": {"horizon": 60, "tp": 0.20, "sl": 0.10},
}
EXT = "C:/Users/Dream/Projects/StockAI/ext_checkpoints"


def load_ticker(ticker, horizon):
    ckpt_name = f"{ticker}_h{horizon}_t10_lb200.csv"
    ckpt = pd.read_csv(os.path.join(EXT, ckpt_name), parse_dates=["date"])
    ckpt["pred_up"] = ckpt["pred_close"] > ckpt["entry_close"]

    raw = yf.download(ticker, period="5y", interval="1d",
                      auto_adjust=True, progress=False)
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index).tz_localize(None) if raw.index.tz else pd.to_datetime(raw.index)
    return ckpt, raw


def simulate_trade(bars, entry_price, strategy, tp, sl, horizon_days):
    """
    Simulate a LONG trade over `bars` (daily OHLC), applying the given exit strategy.
    Returns (exit_price, exit_reason, days_held).
    """
    peak = entry_price
    # Initial SL
    current_sl = entry_price * (1 - sl)
    moved_to_be = False

    for i, (date, bar) in enumerate(bars.iterrows()):
        high   = bar["high"]
        low    = bar["low"]
        close  = bar["close"]

        # Update peak
        if high > peak:
            peak = high

        # Update SL based on strategy
        if strategy == "FIXED":
            pass  # fixed SL, never moves
        elif strategy == "BE_MOVE":
            if peak >= entry_price * 1.05 and not moved_to_be:
                current_sl = entry_price  # move SL to breakeven
                moved_to_be = True
        elif strategy in ("TRAIL_3", "TRAIL_5", "TRAIL_7"):
            trail_pct = {"TRAIL_3": 0.03, "TRAIL_5": 0.05, "TRAIL_7": 0.07}[strategy]
            candidate = peak * (1 - trail_pct)
            if candidate > current_sl:
                current_sl = candidate
        elif strategy == "BE_THEN_5":
            if peak >= entry_price * 1.05 and not moved_to_be:
                current_sl = max(current_sl, entry_price)
                moved_to_be = True
            if moved_to_be:
                candidate = peak * 0.95
                if candidate > current_sl:
                    current_sl = candidate

        # Check exits (SL first then TP, intraday order unknown so use bar extremes)
        # SL hit if low <= current_sl
        if low <= current_sl:
            return current_sl, "SL", i + 1
        # TP hit if high >= entry_price * (1 + tp)
        if high >= entry_price * (1 + tp):
            return entry_price * (1 + tp), "TP", i + 1

    # End of horizon, exit at close
    return bars.iloc[-1]["close"], "EXPIRY", len(bars)


def backtest_strategy(ticker, horizon, tp, sl, strategy):
    ckpt, raw = load_ticker(ticker, horizon)
    longs = ckpt[ckpt["pred_up"]].copy()

    returns = []
    wins = 0
    tp_hits = 0
    sl_hits = 0
    expiries = 0

    for _, row in longs.iterrows():
        entry_date = row["date"]
        entry_price = row["entry_close"]

        # Get next `horizon` trading days
        future_bars = raw.loc[raw.index > entry_date].head(horizon)
        if len(future_bars) < 3:
            continue

        exit_price, reason, days = simulate_trade(
            future_bars, entry_price, strategy, tp, sl, horizon)

        ret = (exit_price - entry_price) / entry_price
        returns.append(ret)
        if ret > 0: wins += 1
        if reason == "TP": tp_hits += 1
        elif reason == "SL": sl_hits += 1
        else: expiries += 1

    if not returns:
        return None
    r = np.array(returns)
    return {
        "strategy": strategy,
        "n":        len(r),
        "win_rate": (r > 0).mean() * 100,
        "avg_ret":  r.mean() * 100,
        "total":    (np.prod(1 + r) - 1) * 100,  # compounded
        "sharpe":   (r.mean() / r.std() * np.sqrt(252/horizon)) if r.std() > 0 else 0,
        "max_win":  r.max() * 100,
        "max_loss": r.min() * 100,
        "tp_hits":  tp_hits,
        "sl_hits":  sl_hits,
        "expiries": expiries,
    }


STRATEGIES = ["FIXED", "BE_MOVE", "TRAIL_3", "TRAIL_5", "TRAIL_7", "BE_THEN_5"]

ticker_arg = sys.argv[1].upper() if len(sys.argv) > 1 else None
tickers = [ticker_arg] if ticker_arg else list(TICKERS.keys())

for ticker in tickers:
    cfg = TICKERS[ticker]
    print(f"\n{'='*80}")
    print(f"TRAILING STOP BACKTEST - {ticker} h={cfg['horizon']}d  (TP={cfg['tp']*100:.0f}% base SL={cfg['sl']*100:.0f}%)")
    print(f"{'='*80}")
    print(f"{'Strategy':<12} {'N':>5} {'Win%':>7} {'AvgRet%':>9} {'Total%':>10} {'Sharpe':>7} {'Max+':>7} {'Max-':>7} {'TP':>5} {'SL':>5} {'Exp':>5}")
    print('-'*90)

    rows = []
    for strat in STRATEGIES:
        r = backtest_strategy(ticker, cfg["horizon"], cfg["tp"], cfg["sl"], strat)
        if r is None:
            continue
        rows.append(r)
        print(f"{r['strategy']:<12} {r['n']:>5} {r['win_rate']:>6.1f}% {r['avg_ret']:>8.2f}% "
              f"{r['total']:>9.1f}% {r['sharpe']:>6.2f} {r['max_win']:>6.1f}% "
              f"{r['max_loss']:>6.1f}% {r['tp_hits']:>5} {r['sl_hits']:>5} {r['expiries']:>5}")

    if rows:
        best = max(rows, key=lambda x: x["sharpe"])
        base = next(r for r in rows if r["strategy"] == "FIXED")
        print(f"\n  Best by Sharpe: {best['strategy']} ({best['sharpe']:.2f}) vs FIXED ({base['sharpe']:.2f})")
        delta = best["total"] - base["total"]
        print(f"  Total return delta vs FIXED: {delta:+.1f}%")
