"""
Phase 1: Full validation on top 20 screener survivors.
Correlation → Walk-forward → Exit optimizer → Trailing stops → VIX → Earnings.
Output: C:/Users/Dream/Projects/StockAI/screener_survivors.csv
"""
import os, sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
from env_paths import base_path

BASE = base_path()
RESULTS = pd.read_csv(f"{BASE}/screener_results.csv")
CKPT_DIR = f"{BASE}/screener_checkpoints"
EXISTING = ["RIVN", "ENVX", "TSLA"]

# Top 20 by val_acc with long_n >= 30
top = RESULTS[RESULTS["long_n"] >= 30].nlargest(20, "val_acc").reset_index(drop=True)
tickers = top["ticker"].tolist()
print(f"Top 20 survivors: {tickers}")

# ── 1a. Correlation ──────────────────────────────────────────────────────────
print("\n[1a] Downloading 3yr returns for correlation...")
symbols = tickers + EXISTING + ["BTC-USD", "NVDA"]
closes = {}
for t in symbols:
    try:
        df = yf.download(t, period="3y", interval="1d", auto_adjust=True, progress=False)
        if df.empty: continue
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        closes[t] = df["close"]
    except Exception as e:
        print(f"  {t}: failed ({e})")

prices = pd.DataFrame(closes).dropna()
returns = prices.pct_change().dropna()
corr = returns.corr()

# Group correlated tickers (>= 0.70)
groups = {}
group_id = 0
assigned = set()
for t in tickers:
    if t in assigned or t not in corr.index: continue
    group = [t]
    for other in tickers:
        if other == t or other in assigned or other not in corr.columns: continue
        if corr.loc[t, other] >= 0.70:
            group.append(other)
    if len(group) > 1:
        for g in group: assigned.add(g); groups.setdefault(f"g{group_id}", []).append(g)
        group_id += 1
    else:
        assigned.add(t)

# For each correlated group, keep highest-val_acc
to_drop_corr = set()
for gid, members in groups.items():
    best = max(members, key=lambda t: top[top["ticker"]==t]["val_acc"].values[0])
    for m in members:
        if m != best: to_drop_corr.add(m)
    print(f"  Group {gid}: {members} → keep {best}")

# Also drop any correlating >=0.70 with RIVN/ENVX/TSLA
to_drop_existing = set()
for t in tickers:
    if t not in corr.index: continue
    for e in EXISTING:
        if e in corr.columns and corr.loc[t, e] >= 0.70:
            to_drop_existing.add(t)
            print(f"  {t}: corr={corr.loc[t, e]:.2f} with {e} → drop")
            break

# Ticker -> corr_group label
def corr_group_label(t):
    for gid, members in groups.items():
        if t in members:
            return gid
    return "-"

# ── 1b. Walk-forward ─────────────────────────────────────────────────────────
print("\n[1b] Walk-forward (4 quarters per ticker)...")
walk_stable = {}
for t in tickers:
    ckpt = f"{CKPT_DIR}/{t}.csv"
    if not os.path.exists(ckpt):
        walk_stable[t] = False; continue
    df = pd.read_csv(ckpt, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if len(df) < 100:
        walk_stable[t] = False; continue
    q_size = len(df) // 4
    q_accs = []
    for i in range(4):
        start = i * q_size
        end = (i+1) * q_size if i < 3 else len(df)
        q = df.iloc[start:end]
        q_accs.append(q["correct"].mean() * 100)
    good_qs = sum(a >= 52 for a in q_accs)
    walk_stable[t] = good_qs >= 2  # >= 50% of quarters (not strictly >50)
    print(f"  {t}: q_accs={[round(a,1) for a in q_accs]} stable={walk_stable[t]}")

# ── 1c. Exit optimizer ───────────────────────────────────────────────────────
print("\n[1c] Exit optimizer (TP/SL scan)...")
TP_grid = [0.10, 0.15, 0.20, 0.25]
SL_grid = [0.02, 0.05, 0.10]

def simulate_trade(bars, entry_price, tp, sl, strategy="FIXED"):
    """Return list of trade returns with given strategy applied."""
    peak = entry_price
    current_sl = entry_price * (1 - sl)
    moved_be = False
    for i, (date, bar) in enumerate(bars.iterrows()):
        high = bar.get("high", bar["close"])
        low  = bar.get("low",  bar["close"])
        if high > peak: peak = high
        if strategy == "BE_MOVE" and peak >= entry_price * 1.05 and not moved_be:
            current_sl = entry_price; moved_be = True
        elif strategy == "BE_THEN_5":
            if peak >= entry_price * 1.05 and not moved_be:
                current_sl = max(current_sl, entry_price); moved_be = True
            if moved_be:
                c = peak * 0.95
                if c > current_sl: current_sl = c
        elif strategy in ("TRAIL_3","TRAIL_5","TRAIL_7"):
            tpct = {"TRAIL_3":0.03,"TRAIL_5":0.05,"TRAIL_7":0.07}[strategy]
            c = peak * (1 - tpct)
            if c > current_sl: current_sl = c
        if low <= current_sl:
            return (current_sl - entry_price) / entry_price
        if strategy in ("FIXED","BE_MOVE") and high >= entry_price * (1 + tp):
            return tp
    return (bars.iloc[-1]["close"] - entry_price) / entry_price

def backtest(t, tp, sl, strategy="FIXED", horizon=60):
    ckpt = f"{CKPT_DIR}/{t}.csv"
    if not os.path.exists(ckpt): return None
    df = pd.read_csv(ckpt, parse_dates=["date"])
    df["pred_up"] = df["pred_close"] > df["entry_close"]
    longs = df[df["pred_up"]]
    if len(longs) < 20: return None

    try:
        raw = yf.download(t, period="5y", interval="1d", auto_adjust=True, progress=False)
        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
        raw.index = pd.to_datetime(raw.index).tz_localize(None) if raw.index.tz else pd.to_datetime(raw.index)
    except:
        return None

    returns = []
    for _, row in longs.iterrows():
        entry_date = row["date"]
        entry_price = row["entry_close"]
        future = raw.loc[raw.index > entry_date].head(horizon)
        if len(future) < 3: continue
        ret = simulate_trade(future, entry_price, tp, sl, strategy)
        returns.append(ret)
    if len(returns) < 10: return None
    r = np.array(returns)
    return {
        "n": len(r),
        "win_rate": (r > 0).mean() * 100,
        "avg_ret": r.mean() * 100,
        "total_ret": (np.prod(1+r)-1) * 100,
        "sharpe": (r.mean()/r.std() * np.sqrt(252/horizon)) if r.std() > 0 else 0,
    }

exit_opt = {}
for t in tickers:
    if not walk_stable.get(t) or t in to_drop_corr or t in to_drop_existing:
        print(f"  {t}: skipped (pre-filtered)")
        continue
    best = {"sharpe": -999}
    for tp in TP_grid:
        for sl in SL_grid:
            r = backtest(t, tp, sl, "FIXED")
            if r and r["sharpe"] > best["sharpe"]:
                best = {**r, "tp": tp, "sl": sl}
    if best["sharpe"] > -999:
        exit_opt[t] = best
        print(f"  {t}: best TP={best['tp']*100:.0f}% SL={best['sl']*100:.0f}% Sharpe={best['sharpe']:.2f}")

# ── 1d. Trailing stop ────────────────────────────────────────────────────────
print("\n[1d] Trailing stop comparison...")
strategies = ["FIXED","TRAIL_3","TRAIL_5","TRAIL_7","BE_MOVE","BE_THEN_5"]
best_strategy = {}
for t, cfg in exit_opt.items():
    results = {}
    for strat in strategies:
        r = backtest(t, cfg["tp"], cfg["sl"], strat)
        if r: results[strat] = r["sharpe"]
    if results:
        winner = max(results, key=results.get)
        best_strategy[t] = {"strategy": winner, "sharpe": results[winner]}
        print(f"  {t}: {winner} (Sharpe {results[winner]:.2f}) vs FIXED ({results.get('FIXED',0):.2f})")

# ── 1e. VIX regime ───────────────────────────────────────────────────────────
print("\n[1e] VIX regime per ticker...")
vix_df = yf.download("^VIX", period="5y", interval="1d", auto_adjust=True, progress=False)
vix_df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in vix_df.columns]
vix_df.index = pd.to_datetime(vix_df.index).tz_localize(None) if vix_df.index.tz else pd.to_datetime(vix_df.index)
vix = vix_df[["close"]].rename(columns={"close":"vix"})

needs_vix = {}
for t in best_strategy:
    ckpt = f"{CKPT_DIR}/{t}.csv"
    df = pd.read_csv(ckpt, parse_dates=["date"])
    df["date"] = df["date"].dt.normalize()
    df["pred_up"] = df["pred_close"] > df["entry_close"]
    df["actual_up"] = df["actual_close"] > df["entry_close"]
    df = df.merge(vix, left_on="date", right_index=True, how="left")
    df["vix"] = df["vix"].ffill()
    longs = df[df["pred_up"]]
    if len(longs) < 30:
        needs_vix[t] = False; continue
    low = longs[longs["vix"] < 15]
    med = longs[(longs["vix"] >= 15) & (longs["vix"] < 25)]
    low_wr = low["actual_up"].mean() * 100 if len(low) >= 5 else float("nan")
    med_wr = med["actual_up"].mean() * 100 if len(med) >= 5 else float("nan")
    if not np.isnan(low_wr) and not np.isnan(med_wr):
        needs_vix[t] = (med_wr - low_wr) > 15
        print(f"  {t}: low_vix={low_wr:.1f}% med_vix={med_wr:.1f}% needs_filter={needs_vix[t]}")
    else:
        needs_vix[t] = False

# ── 1f. Earnings proximity ───────────────────────────────────────────────────
print("\n[1f] Earnings proximity per ticker...")
needs_earnings = {}
for t in best_strategy:
    try:
        ed = yf.Ticker(t).earnings_dates
        if ed is None or ed.empty: needs_earnings[t] = False; continue
        edates = pd.to_datetime(ed.index).tz_localize(None).normalize()
    except Exception:
        needs_earnings[t] = False; continue
    ckpt = f"{CKPT_DIR}/{t}.csv"
    df = pd.read_csv(ckpt, parse_dates=["date"])
    df["date"] = df["date"].dt.normalize()
    df["pred_up"] = df["pred_close"] > df["entry_close"]
    df["actual_up"] = df["actual_close"] > df["entry_close"]
    def near(d):
        return any(abs((d - ed_).days) <= 14 for ed_ in edates)
    df["near_earn"] = df["date"].apply(near)
    longs = df[df["pred_up"]]
    near_longs = longs[longs["near_earn"]]
    clear_longs = longs[~longs["near_earn"]]
    near_wr = near_longs["actual_up"].mean() * 100 if len(near_longs) >= 5 else float("nan")
    clear_wr = clear_longs["actual_up"].mean() * 100 if len(clear_longs) >= 5 else float("nan")
    if not np.isnan(near_wr) and not np.isnan(clear_wr):
        needs_earnings[t] = (clear_wr - near_wr) > 15
        print(f"  {t}: near={near_wr:.1f}% clear={clear_wr:.1f}% needs_filter={needs_earnings[t]}")
    else:
        needs_earnings[t] = False

# ── 1g. Save survivors ───────────────────────────────────────────────────────
rows = []
for t in tickers:
    row = top[top["ticker"] == t].iloc[0]
    dropped = (not walk_stable.get(t)) or t in to_drop_corr or t in to_drop_existing
    verdict = "DROP"
    if not dropped and t in best_strategy:
        verdict = "KEEP"
    rows.append({
        "ticker": t,
        "val_acc": row["val_acc"],
        "long_wr": row["long_wr"],
        "long_n": row["long_n"],
        "walk_forward_stable": walk_stable.get(t, False),
        "corr_group": corr_group_label(t),
        "corr_with_existing": t in to_drop_existing,
        "best_tp": exit_opt.get(t, {}).get("tp", ""),
        "best_sl": exit_opt.get(t, {}).get("sl", ""),
        "best_strategy": best_strategy.get(t, {}).get("strategy", ""),
        "sharpe": best_strategy.get(t, {}).get("sharpe", ""),
        "needs_vix_filter": needs_vix.get(t, False),
        "needs_earnings_filter": needs_earnings.get(t, False),
        "verdict": verdict,
    })

out = pd.DataFrame(rows).sort_values(["verdict","sharpe"], ascending=[True, False])
out.to_csv(f"{BASE}/screener_survivors.csv", index=False)
print(f"\nSaved → {BASE}/screener_survivors.csv")
print(out.to_string(index=False))
