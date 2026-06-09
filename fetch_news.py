"""
Fetch historical news for backtest from Finnhub.
Free tier: 60 calls/min, ~1 year of company news.
Saves to news_data/{TICKER}.csv
"""
import os, sys, json, urllib.request, urllib.parse, csv, time
from datetime import datetime, timedelta

API_KEY = os.environ.get("FINNHUB_KEY", "d7enp4pr01qi33g71030d7enp4pr01qi33g7103g")
TICKER = sys.argv[1] if len(sys.argv) > 1 else "RIVN"
DAYS_BACK = int(sys.argv[2]) if len(sys.argv) > 2 else 365

OUT_DIR = "C:/Users/Dream/Projects/StockAI/news_data"
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"{TICKER}.csv")

# Finnhub limits to 1000 results per call. Chunk by month to be safe.
end_date = datetime.utcnow().date()
start_date = end_date - timedelta(days=DAYS_BACK)

print(f"Fetching {TICKER} news from {start_date} to {end_date}...")

all_news = []
cur = start_date
while cur < end_date:
    chunk_end = min(cur + timedelta(days=30), end_date)
    url = (f"https://finnhub.io/api/v1/company-news?"
           f"symbol={TICKER}&from={cur}&to={chunk_end}&token={API_KEY}")
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"  {cur} - {chunk_end}: ERROR {e}")
        cur = chunk_end
        continue

    n = len(data) if isinstance(data, list) else 0
    print(f"  {cur} - {chunk_end}: {n} articles")
    if isinstance(data, list):
        for art in data:
            all_news.append({
                "datetime": datetime.utcfromtimestamp(art.get("datetime", 0)).isoformat(),
                "headline": art.get("headline", ""),
                "summary":  art.get("summary", "")[:500],
                "source":   art.get("source", ""),
                "url":      art.get("url", ""),
            })
    cur = chunk_end
    time.sleep(1.1)  # respect 60/min limit

# Dedupe by URL
seen = set()
deduped = []
for art in all_news:
    if art["url"] in seen:
        continue
    seen.add(art["url"])
    deduped.append(art)

with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["datetime","headline","summary","source","url"])
    w.writeheader()
    w.writerows(deduped)

print(f"\nSaved {len(deduped)} unique articles -> {out_path}")
