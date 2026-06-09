"""
News sentiment overlay for active watchlist.
Uses Alpha Vantage News & Sentiment API (free tier).

Returns per-ticker:
  - sentiment_score: -1.0 (very bearish) to +1.0 (very bullish)
  - bucket: BEARISH / NEUTRAL / BULLISH
  - n_articles: number of articles in last 48hr
  - top_headline: highest-relevance headline

Usage:
  from news_sentiment import get_sentiment
  s = get_sentiment("RIVN")
  print(s)  # {"score": 0.34, "bucket": "BULLISH", "n": 12, "headline": "..."}

Set env var ALPHA_VANTAGE_KEY=xxxxxxx
Get free key at https://www.alphavantage.co/support/#api-key
"""
import os
import urllib.request
import urllib.parse
import json
from datetime import datetime, timedelta


API_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "")

# Sentiment buckets per Alpha Vantage docs
def bucket(score):
    if score <= -0.35: return "VERY BEARISH"
    if score <= -0.15: return "BEARISH"
    if score <   0.15: return "NEUTRAL"
    if score <   0.35: return "BULLISH"
    return "VERY BULLISH"


def get_sentiment(ticker, hours_back=48):
    """Fetch news sentiment for a ticker over the last N hours."""
    if not API_KEY:
        return {"score": 0.0, "bucket": "NO_KEY", "n": 0,
                "headline": "Set ALPHA_VANTAGE_KEY env var"}

    time_from = (datetime.utcnow() - timedelta(hours=hours_back)).strftime("%Y%m%dT%H%M")
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers":  ticker,
        "time_from": time_from,
        "limit":    50,
        "apikey":   API_KEY,
    }
    url = "https://www.alphavantage.co/query?" + urllib.parse.urlencode(params)

    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return {"score": 0.0, "bucket": "ERROR", "n": 0, "headline": f"API error: {e}"}

    if "feed" not in data or not data["feed"]:
        return {"score": 0.0, "bucket": "NO_NEWS", "n": 0, "headline": "(no recent articles)"}

    feed = data["feed"]
    # Aggregate ticker-specific sentiment scores
    scores = []
    top_article = None
    top_relevance = -1
    for art in feed:
        for ts in art.get("ticker_sentiment", []):
            if ts["ticker"] != ticker:
                continue
            try:
                sc = float(ts["ticker_sentiment_score"])
                rel = float(ts["relevance_score"])
                scores.append(sc * rel)  # weight by relevance
                if rel > top_relevance:
                    top_relevance = rel
                    top_article = art
            except (ValueError, KeyError):
                continue

    if not scores:
        return {"score": 0.0, "bucket": "NO_NEWS", "n": 0, "headline": "(no relevant articles)"}

    avg_score = sum(scores) / len(scores)
    headline = top_article.get("title", "(unknown)") if top_article else "(unknown)"
    return {
        "score":    round(avg_score, 3),
        "bucket":   bucket(avg_score),
        "n":        len(scores),
        "headline": headline[:120],
    }


if __name__ == "__main__":
    for tk in ["RIVN", "ENVX", "TSLA"]:
        s = get_sentiment(tk)
        print(f"{tk}: {s['bucket']:>13}  ({s['score']:+.2f})  n={s['n']}")
        print(f"       Top: {s['headline']}")
