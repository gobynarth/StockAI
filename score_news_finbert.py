"""
Score news headlines with FinBERT (financial sentiment).
Runs on RunPod GPU. Reads news_data/{TICKER}.csv, writes news_scored/{TICKER}.csv.

Usage: python score_news_finbert.py RIVN
"""
import os, sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

TICKER = sys.argv[1] if len(sys.argv) > 1 else "RIVN"
IN_PATH  = f"/workspace/news_data/{TICKER}.csv"
OUT_PATH = f"/workspace/news_scored/{TICKER}.csv"
os.makedirs("/workspace/news_scored", exist_ok=True)

print(f"Loading FinBERT (ProsusAI/finbert)...")
tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", use_safetensors=True).to("cuda").eval()

# Labels: 0=positive, 1=negative, 2=neutral (per ProsusAI/finbert)
LABELS = ["positive", "negative", "neutral"]

print(f"Loading {IN_PATH}...")
df = pd.read_csv(IN_PATH)
print(f"  {len(df)} articles to score")

# Score in batches
BATCH = 32
scores  = []
buckets = []
with torch.no_grad():
    for i in range(0, len(df), BATCH):
        batch = df.iloc[i:i+BATCH]
        texts = (batch["headline"].fillna("") + ". " + batch["summary"].fillna("")).str[:512].tolist()
        enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
        logits = mdl(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        # Sentiment score = P(positive) - P(negative), range [-1, 1]
        # Label indices for ProsusAI/finbert: 0=positive, 1=negative, 2=neutral
        for p in probs:
            score = float(p[0] - p[1])
            scores.append(round(score, 4))
            label_idx = int(p.argmax())
            buckets.append(LABELS[label_idx])
        if (i + BATCH) % 200 == 0:
            print(f"  scored {min(i+BATCH, len(df))}/{len(df)}")

df["sentiment"] = scores
df["bucket"]    = buckets
df.to_csv(OUT_PATH, index=False)

# Quick stats
n = len(df)
pos = (df["bucket"] == "positive").sum()
neg = (df["bucket"] == "negative").sum()
neu = (df["bucket"] == "neutral").sum()
print(f"\nResults: {n} articles")
print(f"  positive: {pos} ({pos/n*100:.0f}%)")
print(f"  negative: {neg} ({neg/n*100:.0f}%)")
print(f"  neutral:  {neu} ({neu/n*100:.0f}%)")
print(f"  avg sentiment: {df['sentiment'].mean():+.3f}")
print(f"\nSaved to {OUT_PATH}")
