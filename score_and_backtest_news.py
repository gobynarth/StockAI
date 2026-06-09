"""
Score news with VADER (free, no GPU) and backtest sentiment overlay.
Joins news sentiment with RIVN h=40 checkpoint to test if filtering by
sentiment improves Kronos's win rate.

Usage: python score_and_backtest_news.py
"""
import os
import pandas as pd
import numpy as np
from datetime import timedelta

# pip install vaderSentiment if missing
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "vaderSentiment"])
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

TICKER = "RIVN"
NEWS_PATH = f"C:/Users/Dream/Projects/StockAI/news_data/{TICKER}.csv"
CHECKPOINT = f"C:/Users/Dream/Projects/StockAI/ext_checkpoints/{TICKER}_h40_t10_lb200.csv"
WINDOW_DAYS = 7   # how many days of news prior to entry to aggregate

# ── 1. Score every headline ─────────────────────────────────────────────────
print(f"Loading {NEWS_PATH}...")
news = pd.read_csv(NEWS_PATH)
news["datetime"] = pd.to_datetime(news["datetime"])
news["date"] = news["datetime"].dt.normalize()
print(f"  {len(news)} articles")

print(f"Scoring with VADER...")
sia = SentimentIntensityAnalyzer()
news["text"] = (news["headline"].fillna("") + ". " + news["summary"].fillna("")).str[:500]
news["sentiment"] = news["text"].apply(lambda t: sia.polarity_scores(t)["compound"])

# ── 2. Aggregate to daily sentiment ──────────────────────────────────────────
daily = news.groupby("date").agg(
    sentiment=("sentiment", "mean"),
    n_articles=("sentiment", "count"),
).reset_index()
print(f"  Daily coverage: {len(daily)} days, avg {daily['n_articles'].mean():.1f} articles/day")
print(f"  Overall sentiment distribution: mean={news['sentiment'].mean():+.3f}")

# ── 3. Load checkpoint and join ──────────────────────────────────────────────
print(f"\nLoading {CHECKPOINT}...")
ckpt = pd.read_csv(CHECKPOINT, parse_dates=["date"])
ckpt["date"] = ckpt["date"].dt.normalize()
ckpt["pred_up"]   = ckpt["pred_close"]   > ckpt["entry_close"]
ckpt["actual_up"] = ckpt["actual_close"]  > ckpt["entry_close"]
ckpt["correct"]   = ckpt["pred_up"] == ckpt["actual_up"]
print(f"  {len(ckpt)} prediction windows ({ckpt['date'].iloc[0].date()} to {ckpt['date'].iloc[-1].date()})")

# Rolling N-day sentiment prior to each prediction date
def get_sentiment_window(d, days_back):
    mask = (daily["date"] >= d - timedelta(days=days_back)) & (daily["date"] < d)
    sub  = daily[mask]
    if len(sub) == 0:
        return float("nan"), 0
    # weighted by article count
    wsum = (sub["sentiment"] * sub["n_articles"]).sum()
    n    = sub["n_articles"].sum()
    return (wsum / n if n > 0 else float("nan"), int(n))

print(f"Joining {WINDOW_DAYS}-day prior sentiment to each prediction...")
sents, ns = [], []
for d in ckpt["date"]:
    s, n = get_sentiment_window(d, WINDOW_DAYS)
    sents.append(s)
    ns.append(n)
ckpt["news_sentiment"] = sents
ckpt["news_n"] = ns

# Filter to rows where we have news data (not all of 868 windows have news coverage)
covered = ckpt[ckpt["news_n"] >= 3].copy()
print(f"  {len(covered)}/{len(ckpt)} windows with >=3 articles in prior {WINDOW_DAYS} days")

# ── 4. Bucket and analyze ────────────────────────────────────────────────────
def bucket(s):
    if s < -0.15: return "BEARISH"
    if s >  0.15: return "BULLISH"
    return "NEUTRAL"
covered["sent_bucket"] = covered["news_sentiment"].apply(bucket)

print(f"\n{'='*70}")
print(f"BACKTEST: Does news sentiment improve Kronos's RIVN signal?")
print(f"{'='*70}")
print(f"\nBaseline (all signals, no filter):")
overall_acc  = covered["correct"].mean() * 100
longs        = covered[covered["pred_up"]]
overall_long = longs["actual_up"].mean() * 100 if len(longs) else float("nan")
print(f"  Overall acc:    {overall_acc:.1f}% (n={len(covered)})")
print(f"  Long signal WR: {overall_long:.1f}% (n={len(longs)})")

print(f"\nBy sentiment bucket (news in prior {WINDOW_DAYS} days):")
print(f"  {'Bucket':<10} {'N':>5} {'Overall acc':>12} {'Long WR':>10} {'Long N':>8}")
print(f"  {'-'*55}")
for bk in ["BULLISH", "NEUTRAL", "BEARISH"]:
    sub = covered[covered["sent_bucket"] == bk]
    if len(sub) < 10:
        continue
    acc = sub["correct"].mean() * 100
    sl = sub[sub["pred_up"]]
    lwr = sl["actual_up"].mean() * 100 if len(sl) > 0 else float("nan")
    lwr_s = f"{lwr:>9.1f}%" if not np.isnan(lwr) else "      n/a"
    print(f"  {bk:<10} {len(sub):>5} {acc:>11.1f}% {lwr_s} {len(sl):>8}")

print(f"\n{'='*70}")
print("INTERPRETATION:")
bullish = covered[covered["sent_bucket"] == "BULLISH"]
bullish_long = bullish[bullish["pred_up"]]
bullish_long_wr = bullish_long["actual_up"].mean() * 100 if len(bullish_long) > 0 else 0

bearish = covered[covered["sent_bucket"] == "BEARISH"]
bearish_long = bearish[bearish["pred_up"]]
bearish_long_wr = bearish_long["actual_up"].mean() * 100 if len(bearish_long) > 0 else 0

improvement = bullish_long_wr - overall_long
print(f"  Long WR baseline:        {overall_long:.1f}%")
print(f"  Long WR (bullish news):  {bullish_long_wr:.1f}%   delta = {improvement:+.1f}pp")
print(f"  Long WR (bearish news):  {bearish_long_wr:.1f}%")

if abs(improvement) >= 10:
    print(f"\n  VERDICT: News sentiment moves the needle ({improvement:+.1f}pp). KEEP.")
elif abs(improvement) >= 5:
    print(f"\n  VERDICT: Marginal effect ({improvement:+.1f}pp). Worth more testing.")
else:
    print(f"\n  VERDICT: No meaningful effect ({improvement:+.1f}pp). KILL the idea.")

# Save scored data
out = f"C:/Users/Dream/Projects/StockAI/news_data/{TICKER}_scored.csv"
news.to_csv(out, index=False)
print(f"\nScored news saved to {out}")
