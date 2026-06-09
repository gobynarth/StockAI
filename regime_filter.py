"""
Volatility regime filter: does edge break down in high VIX / near earnings?
Loads existing ext_checkpoints, cross-references with VIX and earnings dates.
No GPU needed.
"""
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta

EXT = "C:/Users/Dream/Projects/StockAI/ext_checkpoints"

CONFIGS = {
    "RIVN": "RIVN_h40_t10_lb200.csv",
    "COIN": "COIN_h60_t10_lb200.csv",
    "ENVX": "ENVX_h90_t10_lb200.csv",
    "TSLA": "TSLA_h90_t10_lb200.csv",
}

VIX_BUCKETS = [(0, 15, "Low <15"), (15, 25, "Med 15-25"), (25, 35, "High 25-35"), (35, 999, "Extreme >35")]

# Download VIX
print("Downloading VIX...")
vix = yf.download("^VIX", period="10y", interval="1d", auto_adjust=True, progress=False)
vix = vix.reset_index()
vix.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in vix.columns]
vix["date"] = pd.to_datetime(vix["date"]).dt.tz_localize(None).dt.normalize()
vix = vix[["date", "close"]].rename(columns={"close": "vix"}).set_index("date")

# Download earnings dates
print("Fetching earnings calendars...")
earnings = {}
for ticker in CONFIGS:
    try:
        t = yf.Ticker(ticker)
        hist = t.earnings_dates
        if hist is not None and not hist.empty:
            dates = pd.to_datetime(hist.index).tz_localize(None).normalize()
            earnings[ticker] = sorted(dates)
            print(f"  {ticker}: {len(dates)} earnings dates")
        else:
            earnings[ticker] = []
    except Exception as e:
        earnings[ticker] = []
        print(f"  {ticker}: no earnings data ({e})")


def near_earnings(date, ticker, window=14):
    """True if date is within `window` days of an earnings announcement."""
    for ed in earnings.get(ticker, []):
        if abs((date - ed).days) <= window:
            return True
    return False


print(f"\n{'='*75}")
print("VOLATILITY REGIME ANALYSIS")
print(f"{'='*75}")

all_results = []

for ticker, fname in CONFIGS.items():
    df = pd.read_csv(os.path.join(EXT, fname), parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["pred_up"]   = df["pred_close"]  > df["entry_close"]
    df["actual_up"] = df["actual_close"] > df["entry_close"]
    df["correct"]   = df["pred_up"] == df["actual_up"]

    # Merge VIX
    df = df.join(vix, on="date", how="left")
    df["vix"] = df["vix"].ffill()

    # Earnings flag
    df["near_earnings"] = df["date"].apply(lambda d: near_earnings(d, ticker))

    val = df.iloc[-400:].copy()

    print(f"\n--- {ticker} ---")
    print(f"  VIX coverage: {val['vix'].notna().sum()}/{len(val)} rows")
    print(f"  Near-earnings rows: {val['near_earnings'].sum()}")

    # VIX regime breakdown
    print(f"\n  {'Regime':<15} {'N':>5} {'Acc':>7} {'Long WR':>9} {'Long N':>8}")
    print(f"  {'-'*48}")
    for lo, hi, label in VIX_BUCKETS:
        mask = val["vix"].between(lo, hi)
        sub  = val[mask]
        if len(sub) < 5:
            continue
        acc     = sub["correct"].mean() * 100
        longs   = sub[sub["pred_up"]]
        long_wr = longs["actual_up"].mean() * 100 if len(longs) > 0 else float("nan")
        print(f"  {label:<15} {len(sub):>5} {acc:>6.1f}% {long_wr:>8.1f}% {len(longs):>8}")
        all_results.append({"ticker": ticker, "regime": label, "n": len(sub),
                            "acc": acc, "long_wr": long_wr, "long_n": len(longs)})

    # Earnings filter
    near  = val[val["near_earnings"]]
    clear = val[~val["near_earnings"]]
    print(f"\n  Earnings proximity (±14 days):")
    if len(near) > 0:
        print(f"    Near earnings  : n={len(near):<4} acc={near['correct'].mean()*100:.1f}%  "
              f"long WR={near[near['pred_up']]['actual_up'].mean()*100:.1f}%" if near['pred_up'].sum() > 0 else "    Near earnings  : n={len(near)} (no longs)")
    print(f"    Clear of earnings: n={len(clear):<4} acc={clear['correct'].mean()*100:.1f}%  "
          f"long WR={clear[clear['pred_up']]['actual_up'].mean()*100:.1f}%" if clear['pred_up'].sum() > 0 else "")

# Summary: best regimes
print(f"\n\n{'='*75}")
print("SUMMARY: Where does the edge concentrate?")
print(f"{'='*75}")
rdf = pd.DataFrame(all_results)
if not rdf.empty:
    pivot = rdf.pivot_table(index="regime", columns="ticker", values="long_wr", aggfunc="mean")
    print(pivot.round(1).to_string())
    print(f"\nRecommendation:")
    for ticker in CONFIGS:
        sub = rdf[rdf["ticker"] == ticker].dropna(subset=["long_wr"])
        if sub.empty:
            continue
        best  = sub.loc[sub["long_wr"].idxmax(), "regime"]
        worst = sub.loc[sub["long_wr"].idxmin(), "regime"]
        print(f"  {ticker}: best regime={best}, worst regime={worst}")
