import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from datetime import timedelta
sys.path.append("C:/Users/Dream/Kronos")
from model import Kronos, KronosTokenizer, KronosPredictor

TICKER  = sys.argv[1] if len(sys.argv) > 1 else "UBER"
N_WINDOWS = 50   # rolling windows to test
LOOKBACK  = 400
HORIZONS  = [1, 5, 10]

CRYPTO = {"BTC", "SOL", "TAO", "ETH"}
yf_ticker = f"{TICKER}-USD" if TICKER.upper() in CRYPTO else TICKER
is_crypto = TICKER.upper() in CRYPTO

def next_dates(from_date, n):
    if is_crypto:
        return pd.date_range(start=from_date + timedelta(days=1), periods=n)
    return pd.bdate_range(start=from_date + timedelta(days=1), periods=n)

print(f"Fetching {TICKER}...")
raw = yf.download(yf_ticker, period="3y", interval="1d", auto_adjust=True, progress=False)
raw = raw.reset_index()
raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
date_col = "date" if "date" in raw.columns else "datetime"
raw = raw.rename(columns={date_col: "timestamps"})
raw["amount"] = raw["close"] * raw["volume"]
raw = raw[["timestamps", "open", "high", "low", "close", "volume", "amount"]].dropna()
raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)

print("Loading Kronos-mini...")
tokenizer  = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
model      = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
predictor  = KronosPredictor(model, tokenizer, max_context=2048)

# Need enough rows: LOOKBACK + max_horizon + N_WINDOWS
max_h = max(HORIZONS)
required = LOOKBACK + max_h + N_WINDOWS
if len(raw) < required:
    print(f"Not enough data: {len(raw)} rows, need {required}")
    sys.exit(1)

# test windows start far enough back that we have actuals for all horizons
test_start = len(raw) - N_WINDOWS - max_h

results = {h: [] for h in HORIZONS}

for i in range(N_WINDOWS):
    idx = test_start + i
    entry_close = raw.iloc[idx]["close"]
    entry_date  = raw.iloc[idx]["timestamps"]

    x_df = raw.iloc[idx - LOOKBACK:idx][["open","high","low","close","volume","amount"]].reset_index(drop=True)
    x_ts = raw.iloc[idx - LOOKBACK:idx]["timestamps"].reset_index(drop=True)

    # Run one prediction for max horizon, slice for shorter ones
    nd = next_dates(entry_date, max_h)
    y_ts = pd.Series(nd)
    pred = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                             pred_len=max_h, T=1.0, top_p=0.9, sample_count=1, verbose=False)

    for h in HORIZONS:
        pred_close = pred["close"].iloc[h - 1]
        # Find actual close h trading days later
        actual_idx = idx + h
        if actual_idx >= len(raw):
            continue
        actual_close = raw.iloc[actual_idx]["close"]

        pred_dir   = pred_close > entry_close
        actual_dir = actual_close > entry_close
        correct    = pred_dir == actual_dir
        error_pct  = (pred_close - actual_close) / actual_close * 100

        results[h].append({
            "date":         entry_date,
            "entry_close":  entry_close,
            "pred_close":   pred_close,
            "actual_close": actual_close,
            "pred_dir":     pred_dir,
            "actual_dir":   actual_dir,
            "correct":      correct,
            "error_pct":    error_pct,
        })

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{N_WINDOWS} windows done")

# ---- Summary ----
print(f"\n{'Horizon':<10} {'Dir Acc':>8} {'MAE':>8} {'Correct':>8} {'Total':>8}")
print("-" * 46)
for h in HORIZONS:
    df = pd.DataFrame(results[h])
    dir_acc = df["correct"].mean() * 100
    mae     = df["error_pct"].abs().mean()
    print(f"{h}-day{'':<6} {dir_acc:>7.1f}% {mae:>7.2f}% {df['correct'].sum():>8} {len(df):>8}")

# ---- Plot ----
fig, axes = plt.subplots(2, 1, figsize=(13, 9))
fig.patch.set_facecolor("#12121f")
fig.suptitle(f"{TICKER} — Horizon Accuracy Backtest (Kronos-mini, {N_WINDOWS} windows)",
             fontsize=14, fontweight="bold", color="white")

colors_h = {1: "#ff6b6b", 5: "#a78bfa", 10: "#34d399"}

# Panel 1: Direction accuracy over time (rolling 20-window)
ax1 = axes[0]
ax1.set_facecolor("#1a1a2e")
ax1.axhline(50, color="#666", lw=1, linestyle=":", label="50% (coin flip)")
for h in HORIZONS:
    df = pd.DataFrame(results[h])
    rolling_dir = df["correct"].rolling(20).mean() * 100
    dir_acc = df["correct"].mean() * 100
    ax1.plot(df["date"], rolling_dir, color=colors_h[h], lw=2,
             label=f"{h}-day  overall={dir_acc:.1f}%  MAE={df['error_pct'].abs().mean():.2f}%")
ax1.set_title("Direction Accuracy (rolling 20-window avg)", color="white", fontsize=11)
ax1.set_ylabel("Direction Accuracy %", color="white")
ax1.set_ylim(0, 100)
ax1.tick_params(colors="white")
ax1.spines[:].set_color("#444")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.legend(fontsize=10, facecolor="#2a2a3e", labelcolor="white")
ax1.grid(True, alpha=0.2)

# Panel 2: MAE over time (rolling 20-window)
ax2 = axes[1]
ax2.set_facecolor("#1a1a2e")
for h in HORIZONS:
    df = pd.DataFrame(results[h])
    rolling_mae = df["error_pct"].abs().rolling(20).mean()
    ax2.plot(df["date"], rolling_mae, color=colors_h[h], lw=2, label=f"{h}-day MAE")
ax2.set_title("Price Error % (rolling 20-window avg) — lower is better", color="white", fontsize=11)
ax2.set_ylabel("MAE %", color="white")
ax2.tick_params(colors="white")
ax2.spines[:].set_color("#444")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax2.legend(fontsize=10, facecolor="#2a2a3e", labelcolor="white")
ax2.grid(True, alpha=0.2)

plt.tight_layout()
out = f"C:/Users/Dream/StockAI/horizon_backtest_{TICKER}.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved: {out}")
