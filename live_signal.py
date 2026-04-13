"""
Live signal generator: loads all 3 Kronos models, runs today's data,
reports consensus signals (all 3 agree = high confidence trade).
Optionally sends Discord webhook notification.

Usage: python live_signal.py
Discord: set env var DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
"""
import os, sys, json, urllib.request
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta, datetime

sys.path.append("C:/Users/Dream/Kronos")
from model import Kronos, KronosTokenizer, KronosPredictor

EDGE_TICKERS = ["ENVX", "RIVN", "TSLA", "COIN"]
CRYPTO       = {"BTC", "SOL", "TAO", "ETH", "DOGE", "XRP"}

# Updated best configs from extended horizons analysis
BEST_PARAMS = {
    "ENVX": {"horizon": 90, "temp": 1.0, "lookback": 200},
    "RIVN": {"horizon": 40, "temp": 1.0, "lookback": 200},
    "TSLA": {"horizon": 90, "temp": 0.5, "lookback": 400},
    "COIN": {"horizon": 60, "temp": 1.0, "lookback": 200},
}

# Backtest accuracy per ticker with ensemble filter
BACKTEST_ACC = {
    "ENVX": "69.5% (h=90 OOS)",
    "RIVN": "75.0% (h=40 OOS)",
    "TSLA": "57.2% (h=90 OOS)",
    "COIN": "71.8% (h=60 OOS)",
}

MODELS = [
    {"name": "mini",  "model_id": "NeoQuasar/Kronos-mini",  "tok_id": "NeoQuasar/Kronos-Tokenizer-2k",   "max_ctx": 2048},
    {"name": "small", "model_id": "NeoQuasar/Kronos-small", "tok_id": "NeoQuasar/Kronos-Tokenizer-base", "max_ctx": 512},
    {"name": "base",  "model_id": "NeoQuasar/Kronos-base",  "tok_id": "NeoQuasar/Kronos-Tokenizer-base", "max_ctx": 512},
]

def load_data(ticker, lookback):
    CRYPTO_local = {"BTC", "SOL", "TAO", "ETH", "DOGE", "XRP"}
    yf_ticker = f"{ticker}-USD" if ticker in CRYPTO_local else ticker
    csv_path = f"C:/Users/Dream/StockAI/data/{ticker}.csv"
    if os.path.exists(csv_path):
        raw = pd.read_csv(csv_path, parse_dates=["timestamps"])
        raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)
    else:
        raw = yf.download(yf_ticker, period="3y", interval="1d",
                          auto_adjust=True, progress=False, timeout=30)
        raw = raw.reset_index()
        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
        date_col = "date" if "date" in raw.columns else "datetime"
        raw = raw.rename(columns={date_col: "timestamps"})
        raw["amount"] = raw["close"] * raw["volume"]
        raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)
    raw = raw[["timestamps","open","high","low","close","volume","amount"]].dropna()
    return raw

def get_prediction(predictor, raw, params):
    lb      = min(params["lookback"], 512)
    horizon = params["horizon"]
    is_crypto = False  # all EDGE_TICKERS are stocks

    x_df = raw.iloc[-lb:][["open","high","low","close","volume","amount"]].reset_index(drop=True)
    x_ts = raw.iloc[-lb:]["timestamps"].reset_index(drop=True)
    entry_close = raw.iloc[-1]["close"]
    entry_date  = raw.iloc[-1]["timestamps"]

    y_dates = pd.bdate_range(start=entry_date + timedelta(days=1), periods=horizon)
    y_ts = pd.Series(y_dates)

    pred = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                             pred_len=horizon, T=params["temp"], top_p=0.9,
                             sample_count=1, verbose=False)
    pred_close = pred["close"].iloc[-1]
    direction  = "UP" if pred_close > entry_close else "DOWN"
    pct        = (pred_close - entry_close) / entry_close * 100
    return direction, pct, entry_close, pred_close, y_dates[-1]

def send_discord(webhook_url, message):
    try:
        data = json.dumps({"content": message}).encode("utf-8")
        req  = urllib.request.Request(webhook_url, data=data,
                                      headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        print("Discord notification sent.")
    except Exception as e:
        print(f"Discord send failed: {e}")

# ─── Main ──────────────────────────────────────────────────────────────
today = datetime.now().strftime("%Y-%m-%d")
print(f"\nLOADING MODELS...")

predictors = {}
for m in MODELS:
    print(f"  Loading kronos-{m['name']}...")
    tok  = KronosTokenizer.from_pretrained(m["tok_id"])
    mdl  = Kronos.from_pretrained(m["model_id"])
    predictors[m["name"]] = KronosPredictor(mdl, tok, max_context=m["max_ctx"])

print(f"\n{'='*70}")
print(f"DAILY SIGNALS -- {today}")
print(f"{'='*70}")
print(f"{'Ticker':<7} {'mini':>6} {'small':>6} {'base':>6} {'Consensus':>11} {'Action':>8}")
print(f"{'-'*55}")

results     = []
action_lines = []  # for Discord

for ticker in EDGE_TICKERS:
    p   = BEST_PARAMS[ticker]
    raw = load_data(ticker, p["lookback"])

    dirs = {}
    pcts = {}
    for m in MODELS:
        direction, pct, entry_close, pred_close, target_date = get_prediction(
            predictors[m["name"]], raw, p)
        dirs[m["name"]] = direction
        pcts[m["name"]] = pct

    all_up   = all(d == "UP"   for d in dirs.values())
    all_down = all(d == "DOWN" for d in dirs.values())
    if all_up:
        consensus = "AGREE UP"
        action    = "BUY"
    elif all_down:
        consensus = "AGREE DOWN"
        action    = "SELL/SHORT"
    else:
        consensus = "MIXED"
        action    = "SKIP"

    avg_pct = np.mean(list(pcts.values()))
    print(f"{ticker:<7} {dirs['mini']:>6} {dirs['small']:>6} {dirs['base']:>6} "
          f"{consensus:>11} {action:>8}  ({avg_pct:+.1f}%  target: {target_date.strftime('%Y-%m-%d')})")

    if action != "SKIP":
        acc = BACKTEST_ACC.get(ticker, "?")
        action_lines.append(
            f"**{action} {ticker}** — {consensus} ({avg_pct:+.1f}% predicted over {p['horizon']}d)"
            f"  |  backtest acc: {acc}"
        )

    results.append({
        "date": today, "ticker": ticker,
        "mini": dirs["mini"], "small": dirs["small"], "base": dirs["base"],
        "consensus": consensus, "action": action,
        "avg_pct": round(avg_pct, 2), "horizon": p["horizon"],
        "target_date": target_date.strftime("%Y-%m-%d"),
        "entry": round(entry_close, 2),
    })

print(f"\n{'='*70}")
print("Ensemble filter: only BUY/SELL when all 3 models agree (highest confidence)")
print("SKIP = models disagree, no trade today")
print(f"{'='*70}")

# Save CSV
out = f"C:/Users/Dream/StockAI/live_signals_{today.replace('-','')}.csv"
pd.DataFrame(results).to_csv(out, index=False)
print(f"\nSaved: {out}")

# Discord notification
webhook = os.environ.get("DISCORD_WEBHOOK", "")
if webhook:
    if action_lines:
        msg = f"**StockAI Daily Signals — {today}**\n" + "\n".join(action_lines)
    else:
        msg = f"**StockAI Daily Signals — {today}**\nNo consensus signals today. All models disagree — SKIP."
    send_discord(webhook, msg)
else:
    print("\nNo DISCORD_WEBHOOK env var set — skipping notification.")
    print("To enable: set DISCORD_WEBHOOK=https://discord.com/api/webhooks/...")
