"""
Phase 2 validation for Tier 1 ensemble candidates: PATH, ITW, HON, NCLH.
Uses agree-only signals (mini+small+base all agree on direction).
Runs: Correlation -> Walk-forward -> Exit optimizer -> Trailing stops -> VIX -> Earnings.
"""
import os
import pandas as pd
import numpy as np
import yfinance as yf
from env_paths import base_path

BASE = base_path()
CKPT_DIR = f"{BASE}/ensemble_checkpoints"
TICKERS = ["PATH", "ITW", "HON", "NCLH"]
EXISTING = ["RIVN", "ENVX", "TSLA", "BITF"]
MODELS = ["mini", "small", "base"]
HORIZON = 60
VALIDATION_N = 400


def build_ensemble(ticker):
    """Load checkpoints, merge, return DataFrame with agree-only signals."""
    dfs = {}
    for model in MODELS:
        path = f"{CKPT_DIR}/{ticker}_{model}.csv"
        df = pd.read_csv(path, parse_dates=["date"])
        df = df[df["horizon"] == HORIZON].reset_index(drop=True)
        dfs[model] = df

    merged = dfs["mini"][["date", "entry_close", "actual_close", "correct"]].copy()
    merged = merged.rename(columns={"correct": "correct_mini"})
    merged["pred_mini"] = dfs["mini"]["pred_close"]
    for model in ["small", "base"]:
        merged[f"pred_{model}"] = dfs[model]["pred_close"]
        merged[f"correct_{model}"] = dfs[model]["correct"]

    for model in MODELS:
        merged[f"dir_{model}"] = merged[f"pred_{model}"] > merged["entry_close"]

    merged["all_agree"] = (
        (merged["dir_mini"] == merged["dir_small"]) &
        (merged["dir_small"] == merged["dir_base"])
    )
    merged["pred_up"] = merged["dir_mini"]  # direction when all agree
    merged["actual_up"] = merged["actual_close"] > merged["entry_close"]
    return merged


def simulate_trade(bars, entry_price, tp, sl, strategy="FIXED"):
    peak = entry_price
    current_sl = entry_price * (1 - sl)
    moved_be = False
    for _, bar in bars.iterrows():
        high = bar.get("high", bar["close"])
        low = bar.get("low", bar["close"])
        if high > peak:
            peak = high
        if strategy == "BE_MOVE" and peak >= entry_price * 1.05 and not moved_be:
            current_sl = entry_price; moved_be = True
        elif strategy == "BE_THEN_5":
            if peak >= entry_price * 1.05 and not moved_be:
                current_sl = max(current_sl, entry_price); moved_be = True
            if moved_be:
                c = peak * 0.95
                if c > current_sl:
                    current_sl = c
        elif strategy in ("TRAIL_3", "TRAIL_5", "TRAIL_7"):
            tpct = {"TRAIL_3": 0.03, "TRAIL_5": 0.05, "TRAIL_7": 0.07}[strategy]
            c = peak * (1 - tpct)
            if c > current_sl:
                current_sl = c
        if low <= current_sl:
            return (current_sl - entry_price) / entry_price
        if strategy in ("FIXED", "BE_MOVE") and high >= entry_price * (1 + tp):
            return tp
    return (bars.iloc[-1]["close"] - entry_price) / entry_price


def backtest_ensemble(ticker, merged, raw, tp, sl, strategy="FIXED"):
    agree = merged[merged["all_agree"]].iloc[-VALIDATION_N:]
    longs = agree[agree["pred_up"]]
    if len(longs) < 10:
        return None

    returns = []
    for _, row in longs.iterrows():
        entry_date = row["date"]
        entry_price = row["entry_close"]
        future = raw.loc[raw.index > entry_date].head(HORIZON)
        if len(future) < 3:
            continue
        ret = simulate_trade(future, entry_price, tp, sl, strategy)
        returns.append(ret)
    if len(returns) < 10:
        return None
    r = np.array(returns)
    return {
        "n": len(r),
        "win_rate": (r > 0).mean() * 100,
        "avg_ret": r.mean() * 100,
        "total_ret": (np.prod(1 + r) - 1) * 100,
        "sharpe": (r.mean() / r.std() * np.sqrt(252 / HORIZON)) if r.std() > 0 else 0,
        "long_wr": (r > 0).mean() * 100,
    }


# ── 1. Correlation ──────────────────────────────────────────────────────────
print("=" * 60)
print("[1] CORRELATION CHECK")
print("=" * 60)
symbols = TICKERS + EXISTING
closes = {}
for t in symbols:
    try:
        df = yf.download(t, period="3y", interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            continue
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        closes[t] = df["close"]
    except Exception as e:
        print(f"  {t}: failed ({e})")

prices = pd.DataFrame(closes).dropna()
returns = prices.pct_change().dropna()
corr = returns.corr()

print("\nCorrelation matrix (new vs existing):")
for t in TICKERS:
    row = []
    for e in EXISTING + TICKERS:
        if t == e:
            continue
        if t in corr.index and e in corr.columns:
            c = corr.loc[t, e]
            flag = " ***" if abs(c) >= 0.70 else ""
            row.append(f"{e}={c:.2f}{flag}")
    print(f"  {t}: {', '.join(row)}")

# Check new tickers against each other
print("\nNew ticker cross-correlation:")
for i, t1 in enumerate(TICKERS):
    for t2 in TICKERS[i+1:]:
        if t1 in corr.index and t2 in corr.columns:
            c = corr.loc[t1, t2]
            flag = " *** HIGH" if abs(c) >= 0.70 else ""
            print(f"  {t1}/{t2}: {c:.2f}{flag}")


# ── 2. Walk-forward ─────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("[2] WALK-FORWARD STABILITY")
print("=" * 60)

walk_results = {}
for t in TICKERS:
    merged = build_ensemble(t)
    agree = merged[merged["all_agree"]].reset_index(drop=True)
    if len(agree) < 100:
        walk_results[t] = {"stable": False, "quarters": []}
        continue
    q_size = len(agree) // 4
    q_accs = []
    for i in range(4):
        start = i * q_size
        end = (i + 1) * q_size if i < 3 else len(agree)
        q = agree.iloc[start:end]
        acc = (q["pred_up"] == q["actual_up"]).mean() * 100
        q_accs.append(acc)
    good_qs = sum(a >= 55 for a in q_accs)
    walk_results[t] = {"stable": good_qs >= 3, "quarters": q_accs}
    print(f"  {t}: quarters={[round(a, 1) for a in q_accs]} stable={good_qs >= 3} ({good_qs}/4 >= 55%)")


# ── 3. Exit optimizer ───────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("[3] EXIT OPTIMIZER (TP/SL scan, agree-only longs)")
print("=" * 60)

TP_grid = [0.10, 0.15, 0.20, 0.25]
SL_grid = [0.02, 0.05, 0.10]

exit_opt = {}
raw_cache = {}
for t in TICKERS:
    merged = build_ensemble(t)
    try:
        raw = yf.download(t, period="5y", interval="1d", auto_adjust=True, progress=False)
        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
        raw.index = pd.to_datetime(raw.index).tz_localize(None) if raw.index.tz else pd.to_datetime(raw.index)
        raw_cache[t] = raw
    except:
        print(f"  {t}: failed to download price data")
        continue

    best = {"sharpe": -999}
    print(f"  {t}:")
    for tp in TP_grid:
        for sl in SL_grid:
            r = backtest_ensemble(t, merged, raw, tp, sl, "FIXED")
            if r and r["sharpe"] > best["sharpe"]:
                best = {**r, "tp": tp, "sl": sl}
            if r:
                print(f"    TP={tp*100:.0f}% SL={sl*100:.0f}%: Sharpe={r['sharpe']:.2f} WR={r['win_rate']:.1f}% Total={r['total_ret']:.1f}%")
    if best["sharpe"] > -999:
        exit_opt[t] = best
        print(f"  >> {t} BEST: TP={best['tp']*100:.0f}% SL={best['sl']*100:.0f}% Sharpe={best['sharpe']:.2f} WR={best['win_rate']:.1f}%")


# ── 4. Trailing stop comparison ─────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("[4] TRAILING STOP COMPARISON")
print("=" * 60)

strategies = ["FIXED", "TRAIL_3", "TRAIL_5", "TRAIL_7", "BE_MOVE", "BE_THEN_5"]
best_strategy = {}
for t, cfg in exit_opt.items():
    merged = build_ensemble(t)
    raw = raw_cache[t]
    results = {}
    for strat in strategies:
        r = backtest_ensemble(t, merged, raw, cfg["tp"], cfg["sl"], strat)
        if r:
            results[strat] = {"sharpe": r["sharpe"], "wr": r["win_rate"], "total": r["total_ret"]}
    if results:
        winner = max(results, key=lambda s: results[s]["sharpe"])
        best_strategy[t] = {"strategy": winner, **results[winner]}
        print(f"  {t}:")
        for strat, res in sorted(results.items(), key=lambda x: -x[1]["sharpe"]):
            flag = " <<<" if strat == winner else ""
            print(f"    {strat:<12} Sharpe={res['sharpe']:.2f} WR={res['wr']:.1f}% Total={res['total']:.1f}%{flag}")


# ── 5. VIX regime ───────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("[5] VIX REGIME FILTER")
print("=" * 60)

vix_df = yf.download("^VIX", period="5y", interval="1d", auto_adjust=True, progress=False)
vix_df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in vix_df.columns]
vix_df.index = pd.to_datetime(vix_df.index).tz_localize(None) if vix_df.index.tz else pd.to_datetime(vix_df.index)
vix = vix_df[["close"]].rename(columns={"close": "vix"})

needs_vix = {}
for t in best_strategy:
    merged = build_ensemble(t)
    agree = merged[merged["all_agree"]].copy()
    longs = agree[agree["pred_up"]].copy()
    longs["date"] = longs["date"].dt.normalize()
    longs = longs.merge(vix, left_on="date", right_index=True, how="left")
    longs["vix"] = longs["vix"].ffill()

    low = longs[longs["vix"] < 15]
    med = longs[(longs["vix"] >= 15) & (longs["vix"] < 25)]
    high = longs[longs["vix"] >= 25]

    low_wr = low["actual_up"].mean() * 100 if len(low) >= 5 else float("nan")
    med_wr = med["actual_up"].mean() * 100 if len(med) >= 5 else float("nan")
    high_wr = high["actual_up"].mean() * 100 if len(high) >= 5 else float("nan")

    if not np.isnan(low_wr) and not np.isnan(med_wr):
        needs_vix[t] = (med_wr - low_wr) > 15
    else:
        needs_vix[t] = False

    print(f"  {t}: low_vix(<15)={low_wr:.1f}% ({len(low)}n) med(15-25)={med_wr:.1f}% ({len(med)}n) high(25+)={high_wr:.1f}% ({len(high)}n) filter={needs_vix[t]}")


# ── 6. Earnings proximity ───────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("[6] EARNINGS PROXIMITY FILTER")
print("=" * 60)

needs_earnings = {}
for t in best_strategy:
    try:
        ed = yf.Ticker(t).earnings_dates
        if ed is None or ed.empty:
            needs_earnings[t] = False
            print(f"  {t}: no earnings data")
            continue
        edates = pd.to_datetime(ed.index).tz_localize(None).normalize()
    except Exception:
        needs_earnings[t] = False
        print(f"  {t}: no earnings data")
        continue

    merged = build_ensemble(t)
    agree = merged[merged["all_agree"]].copy()
    longs = agree[agree["pred_up"]].copy()
    longs["date"] = longs["date"].dt.normalize()

    def near(d):
        return any(abs((d - ed_).days) <= 14 for ed_ in edates)

    longs["near_earn"] = longs["date"].apply(near)
    near_longs = longs[longs["near_earn"]]
    clear_longs = longs[~longs["near_earn"]]
    near_wr = near_longs["actual_up"].mean() * 100 if len(near_longs) >= 5 else float("nan")
    clear_wr = clear_longs["actual_up"].mean() * 100 if len(clear_longs) >= 5 else float("nan")

    if not np.isnan(near_wr) and not np.isnan(clear_wr):
        needs_earnings[t] = (clear_wr - near_wr) > 15
        print(f"  {t}: near_earn={near_wr:.1f}% ({len(near_longs)}n) clear={clear_wr:.1f}% ({len(clear_longs)}n) filter={needs_earnings[t]}")
    else:
        needs_earnings[t] = False
        print(f"  {t}: insufficient data for earnings test")


# ── FINAL SUMMARY ───────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("FINAL SUMMARY — TIER 1 VALIDATION")
print("=" * 60)
print(f"{'Ticker':<8} {'Agree Acc':>10} {'Walk-Fwd':>10} {'Strategy':>12} {'TP':>5} {'SL':>5} {'Sharpe':>8} {'WR':>6} {'VIX':>8} {'Earn':>8}")
print("-" * 90)
for t in TICKERS:
    merged = build_ensemble(t)
    agree = merged[merged["all_agree"]].iloc[-VALIDATION_N:]
    agree_acc = (agree["pred_up"] == agree["actual_up"]).mean() * 100

    wf = walk_results.get(t, {})
    wf_str = "YES" if wf.get("stable") else "NO"

    if t in best_strategy:
        bs = best_strategy[t]
        eo = exit_opt[t]
        print(f"{t:<8} {agree_acc:>9.1f}% {wf_str:>10} {bs['strategy']:>12} {eo['tp']*100:>4.0f}% {eo['sl']*100:>4.0f}% {bs['sharpe']:>7.2f} {bs['wr']:>5.1f}% {'skip<15' if needs_vix.get(t) else 'none':>8} {'skip' if needs_earnings.get(t) else 'none':>8}")
    else:
        print(f"{t:<8} {agree_acc:>9.1f}% {wf_str:>10} {'N/A':>12}")
