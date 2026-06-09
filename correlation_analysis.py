"""
Correlation analysis: how correlated are our watchlist tickers?
Revises the worst-case portfolio loss assuming correlated moves.
No GPU needed.
"""
import pandas as pd
import numpy as np
import yfinance as yf

WATCHLIST = {
    "RIVN": {"alloc": 0.10, "sl": 0.02},
    "TSLA": {"alloc": 0.12, "sl": 0.02},
    "COIN": {"alloc": 0.06, "sl": 0.10},
    "ENVX": {"alloc": 0.06, "sl": 0.02},
    "NIO":  {"alloc": 0.06, "sl": 0.02},
    "RIOT": {"alloc": 0.06, "sl": 0.02},
    "SMCI": {"alloc": 0.06, "sl": 0.05},
    "MARA": {"alloc": 0.06, "sl": 0.02},
}

print("Downloading price data...")
prices = {}
for ticker in WATCHLIST:
    df = yf.download(ticker, period="3y", interval="1d",
                     auto_adjust=True, progress=False)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    prices[ticker] = df["close"]

price_df  = pd.DataFrame(prices).dropna()
returns   = price_df.pct_change().dropna()

print(f"\n{'='*75}")
print("RETURN CORRELATION MATRIX (3-year daily returns)")
print(f"{'='*75}")
corr = returns.corr().round(2)
print(corr.to_string())

print(f"\n\n{'='*75}")
print("CLUSTER ANALYSIS: Which tickers move together?")
print(f"{'='*75}")
tickers = list(WATCHLIST.keys())
for i, t1 in enumerate(tickers):
    for t2 in tickers[i+1:]:
        if t1 in corr.columns and t2 in corr.columns:
            c = corr.loc[t1, t2]
            if abs(c) > 0.6:
                label = "HIGHLY CORRELATED" if c > 0.6 else "HIGHLY INVERSE"
                print(f"  {t1} <-> {t2}: {c:.2f}  {label}")

print(f"\n\n{'='*75}")
print("PORTFOLIO WORST-CASE LOSS (revised for correlation)")
print(f"{'='*75}")

# Assume on a bad day, correlated tickers all hit SL simultaneously
# Estimate: find historical days where multiple tickers dropped hard

bad_days = []
for date, row in returns.iterrows():
    sl_hits = 0
    loss = 0.0
    for ticker, cfg in WATCHLIST.items():
        if ticker not in row.index:
            continue
        ret = row[ticker]
        if ret <= -cfg["sl"]:
            sl_hits += 1
            loss += cfg["alloc"] * cfg["sl"]
    bad_days.append({"date": date, "sl_hits": sl_hits, "portfolio_loss": loss})

bad_df = pd.DataFrame(bad_days).sort_values("portfolio_loss", ascending=False)

print(f"\n  Worst single days (by portfolio loss):")
print(f"  {'Date':<12} {'SL hits':>8} {'Portfolio loss':>15}")
print(f"  {'-'*40}")
for _, row in bad_df.head(10).iterrows():
    print(f"  {str(row['date'].date()):<12} {row['sl_hits']:>8} {row['portfolio_loss']*100:>14.2f}%")

naive_worst = sum(cfg["alloc"] * cfg["sl"] for cfg in WATCHLIST.values())
actual_worst = bad_df["portfolio_loss"].max()
pct_99 = bad_df["portfolio_loss"].quantile(0.99)

print(f"\n  Naive worst case (all SL independent): {naive_worst*100:.2f}%")
print(f"  Historical worst day (actual):         {actual_worst*100:.2f}%")
print(f"  99th percentile bad day:               {pct_99*100:.2f}%")

print(f"\n  Days with 3+ simultaneous SL hits: {(bad_df['sl_hits'] >= 3).sum()}")
print(f"  Days with 4+ simultaneous SL hits: {(bad_df['sl_hits'] >= 4).sum()}")

# Correlation on signal dates specifically
print(f"\n\n{'='*75}")
print("CONDITIONAL CORRELATION: On days RIVN drops >2%, what do others do?")
print(f"{'='*75}")
rivn_bad = returns[returns["RIVN"] < -0.02]
print(f"\n  RIVN dropped >2%: {len(rivn_bad)} days")
if len(rivn_bad) > 10:
    print(f"  Average return of other tickers on those days:")
    for t in tickers:
        if t == "RIVN" or t not in rivn_bad.columns:
            continue
        avg = rivn_bad[t].mean() * 100
        pct_down = (rivn_bad[t] < 0).mean() * 100
        print(f"    {t:<6}: avg {avg:+.1f}%  |  {pct_down:.0f}% of days also negative")
