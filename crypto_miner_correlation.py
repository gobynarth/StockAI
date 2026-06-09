"""
Crypto miner correlation analysis.
Checks if 'crypto miners' are still pure BTC plays or have decoupled via AI pivot.

Compares:
  - Static 3yr correlation (all miners vs each other, vs BTC, vs AI datacenter proxies)
  - Rolling 90-day correlation (shows AI pivot decoupling over time)
  - Beta to BTC and NVDA (AI proxy)
"""
import pandas as pd
import numpy as np
import yfinance as yf

MINERS = ["RIOT", "MARA", "CLSK", "BITF", "CAN", "IREN", "HUT", "BTBT", "GREE", "WULF", "CIFR"]
PROXIES = {
    "BTC": "BTC-USD",     # Bitcoin
    "NVDA": "NVDA",       # AI datacenter proxy
    "SMCI": "SMCI",       # AI server proxy
    "APPLE-DC": "APLD",   # Applied Digital (dedicated AI datacenter REIT)
    "COIN": "COIN",       # Crypto exchange proxy
    "SPY": "SPY",         # Market
}

tickers = MINERS + list(PROXIES.values())
print(f"Downloading data for {len(tickers)} tickers...")
closes = {}
for t in tickers:
    df = yf.download(t, period="3y", interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        print(f"  {t}: no data")
        continue
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None) if df.index.tz else pd.to_datetime(df.index)
    closes[t] = df["close"]

prices = pd.DataFrame(closes).dropna()
returns = prices.pct_change().dropna()
print(f"  {len(returns)} common trading days")

# ── Static correlation matrix ──────────────────────────────────────────────
print(f"\n{'='*75}")
print(f"STATIC CORRELATION (3yr daily returns)")
print(f"{'='*75}")
# Miners vs proxies
rel_cols = MINERS + list(PROXIES.values())
rel_cols = [c for c in rel_cols if c in returns.columns]
corr = returns[rel_cols].corr()

# Focused matrix: miners as rows, proxies as cols
print(f"\n{'Miner':<7}", end="")
for p_name, p_tk in PROXIES.items():
    if p_tk in corr.columns:
        print(f"{p_name:>10}", end="")
print()
print("-" * 70)
for m in MINERS:
    if m not in corr.index:
        continue
    print(f"{m:<7}", end="")
    for p_name, p_tk in PROXIES.items():
        if p_tk in corr.columns:
            v = corr.loc[m, p_tk]
            print(f"{v:>10.2f}", end="")
    print()

# ── Miner-to-miner correlation ─────────────────────────────────────────────
print(f"\n{'='*75}")
print(f"MINER-TO-MINER CORRELATION")
print(f"{'='*75}\n")
miner_corr = returns[[m for m in MINERS if m in returns.columns]].corr()
print(miner_corr.round(2).to_string())

# ── Beta to BTC and NVDA ───────────────────────────────────────────────────
print(f"\n{'='*75}")
print(f"BETA: each miner vs BTC and NVDA  (beta>1 = moves more, <1 = less)")
print(f"{'='*75}")
print(f"  {'Miner':<7} {'Beta BTC':>10} {'Beta NVDA':>11} {'Alpha vs BTC':>14}")
print("-" * 50)
for m in MINERS:
    if m not in returns.columns:
        continue
    btc_var = returns["BTC-USD"].var()
    nvda_var = returns["NVDA"].var()
    beta_btc  = returns[[m, "BTC-USD"]].cov().iloc[0, 1] / btc_var
    beta_nvda = returns[[m, "NVDA"]].cov().iloc[0, 1] / nvda_var
    # Alpha: excess return after subtracting BTC-beta-weighted BTC return
    daily_alpha = (returns[m] - beta_btc * returns["BTC-USD"]).mean() * 252 * 100
    print(f"  {m:<7} {beta_btc:>9.2f}x {beta_nvda:>10.2f}x {daily_alpha:>12.1f}%")

# ── Rolling 90-day correlation to BTC — has it decoupled? ──────────────────
print(f"\n{'='*75}")
print(f"ROLLING 90-DAY CORRELATION TO BTC (has it decoupled over time?)")
print(f"{'='*75}")
print(f"\nShows correlation at 3 points in time:")
print(f"  {'Miner':<7} {'~2yr ago':>10} {'~1yr ago':>10} {'Recent':>10} {'Delta':>10}")
print("-" * 55)
for m in MINERS:
    if m not in returns.columns:
        continue
    roll = returns[m].rolling(90).corr(returns["BTC-USD"])
    roll = roll.dropna()
    if len(roll) < 200:
        continue
    early  = roll.iloc[:60].mean()     # earliest available period
    mid    = roll.iloc[len(roll)//2 - 30:len(roll)//2 + 30].mean()  # middle
    recent = roll.iloc[-60:].mean()    # most recent
    delta  = recent - early
    decoup = " *" if delta < -0.15 else ""
    print(f"  {m:<7} {early:>9.2f}  {mid:>9.2f}  {recent:>9.2f}  {delta:>+9.2f}{decoup}")

print(f"\n* = meaningful decoupling (correlation dropped >0.15)")

# ── Rolling corr to NVDA — has AI pivot shown up? ──────────────────────────
print(f"\n{'='*75}")
print(f"ROLLING 90-DAY CORRELATION TO NVDA (AI proxy — has it increased?)")
print(f"{'='*75}")
print(f"  {'Miner':<7} {'~2yr ago':>10} {'Recent':>10} {'Delta':>10}")
print("-" * 45)
for m in MINERS:
    if m not in returns.columns:
        continue
    roll = returns[m].rolling(90).corr(returns["NVDA"])
    roll = roll.dropna()
    if len(roll) < 200:
        continue
    early = roll.iloc[:60].mean()
    recent = roll.iloc[-60:].mean()
    delta = recent - early
    ai_pivot = " *" if delta > 0.15 else ""
    print(f"  {m:<7} {early:>9.2f}  {recent:>9.2f}  {delta:>+9.2f}{ai_pivot}")

print(f"\n* = rising correlation to NVDA (possible AI pivot effect)")
