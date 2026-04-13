import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from datetime import timedelta
sys.path.append("C:/Users/Dream/Kronos")
from model import Kronos, KronosTokenizer, KronosPredictor

TICKER = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
LOOKBACK = 400
N_DAYS = 9  # rolling 1-day windows with known actuals

# Crypto tickers need -USD suffix for yfinance
CRYPTO = {"BTC", "SOL", "TAO", "ETH"}
yf_ticker = f"{TICKER}-USD" if TICKER.upper() in CRYPTO else TICKER

MODELS = [
    {"name": "mini",  "model_id": "NeoQuasar/Kronos-mini",  "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",  "max_context": 2048, "params": "4.1M"},
    {"name": "small", "model_id": "NeoQuasar/Kronos-small", "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base","max_context": 512,  "params": "24.7M"},
    {"name": "base",  "model_id": "NeoQuasar/Kronos-base",  "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base","max_context": 512,  "params": "102.3M"},
]

print(f"Fetching {TICKER} ({yf_ticker})...")
raw = yf.download(yf_ticker, period="3y", interval="1d", auto_adjust=True, progress=False)
raw = raw.reset_index()
raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
date_col = "date" if "date" in raw.columns else "datetime"
raw = raw.rename(columns={date_col: "timestamps"})
raw["amount"] = raw["close"] * raw["volume"]
raw = raw[["timestamps", "open", "high", "low", "close", "volume", "amount"]].dropna()
raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)

# For crypto use calendar days, for stocks use bdate
is_crypto = TICKER.upper() in CRYPTO
def next_dates(from_date, n):
    if is_crypto:
        return pd.date_range(start=from_date + timedelta(days=1), periods=n)
    return pd.bdate_range(start=from_date + timedelta(days=1), periods=n)

test_start_idx = len(raw) - N_DAYS - 1
actuals = []
for i in range(N_DAYS):
    idx = test_start_idx + i + 1
    actuals.append({"date": raw.iloc[idx]["timestamps"], "actual": raw.iloc[idx]["close"]})
actuals_df = pd.DataFrame(actuals).set_index("date")

# ---- Rolling 1-day model comparison ----
all_results = {}
loaded_models = {}

for cfg in MODELS:
    print(f"\nLoading Kronos-{cfg['name']} ({cfg['params']})...")
    tokenizer = KronosTokenizer.from_pretrained(cfg["tokenizer_id"])
    model = Kronos.from_pretrained(cfg["model_id"])
    predictor = KronosPredictor(model, tokenizer, max_context=cfg["max_context"])
    loaded_models[cfg["name"]] = (predictor, cfg)

    preds = []
    for i in range(N_DAYS):
        idx = test_start_idx + i
        lb = min(LOOKBACK, cfg["max_context"])
        x_df = raw.iloc[idx - lb:idx][["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
        x_ts = raw.iloc[idx - lb:idx]["timestamps"].reset_index(drop=True)
        nd = next_dates(raw.iloc[idx]["timestamps"], 1)
        y_ts = pd.Series(nd)
        pred = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                                 pred_len=1, T=1.0, top_p=0.9, sample_count=1, verbose=False)
        preds.append({"date": nd[0], "predicted": pred["close"].iloc[0]})
        actual_val = actuals[i]['actual']
        print(f"  [{cfg['name']}] Day {i+1}/{N_DAYS}: pred={pred['close'].iloc[0]:.2f}, actual={actual_val:.2f}")

    pred_df = pd.DataFrame(preds).set_index("date")
    pred_df["actual"] = actuals_df["actual"]
    pred_df["error_pct"] = (pred_df["predicted"] - pred_df["actual"]) / pred_df["actual"] * 100
    pred_df["mae"] = pred_df["error_pct"].abs()
    # Direction: did it predict up/down correctly vs previous day?
    prev_closes = [raw.iloc[test_start_idx + i]["close"] for i in range(N_DAYS)]
    pred_df["prev_close"] = prev_closes
    pred_df["actual_dir"] = pred_df["actual"] > pred_df["prev_close"]
    pred_df["pred_dir"]   = pred_df["predicted"] > pred_df["prev_close"]
    pred_df["dir_correct"] = pred_df["actual_dir"] == pred_df["pred_dir"]
    all_results[cfg["name"]] = pred_df
    dir_acc = pred_df["dir_correct"].mean() * 100
    print(f"  [{cfg['name']}] MAE: {pred_df['mae'].mean():.2f}%  Direction: {dir_acc:.0f}%")

# ---- Forward predictions: 1, 5, 10 days using best model (mini) ----
print("\nRunning forward predictions (1, 5, 10 days) with Kronos-mini...")
best_predictor, best_cfg = loaded_models["mini"]
lb = min(LOOKBACK, best_cfg["max_context"])
x_df_fwd = raw.iloc[-lb:][["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
x_ts_fwd = raw.iloc[-lb:]["timestamps"].reset_index(drop=True)
last_date = raw.iloc[-1]["timestamps"]

fwd_preds = {}
for horizon in [1, 5, 10]:
    nd = next_dates(last_date, horizon)
    y_ts = pd.Series(nd)
    pred = best_predictor.predict(df=x_df_fwd, x_timestamp=x_ts_fwd, y_timestamp=y_ts,
                                  pred_len=horizon, T=1.0, top_p=0.9, sample_count=1, verbose=False)
    pred.index = nd
    fwd_preds[horizon] = pred
    print(f"  {horizon}-day: {pred['close'].iloc[-1]:.2f} (last close: {raw['close'].iloc[-1]:.2f})")

# ---- Plot ----
colors_model = {"mini": "#f59e0b", "small": "#3b82f6", "base": "#10b981"}
colors_fwd   = {1: "#ff6b6b", 5: "#a78bfa", 10: "#34d399"}
params_map   = {"mini": "4.1M", "small": "24.7M", "base": "102.3M"}

fig, axes = plt.subplots(3, 1, figsize=(14, 13))
fig.patch.set_facecolor("#12121f")
fig.suptitle(f"{TICKER} — Kronos Predictions", fontsize=15, fontweight="bold", color="white")

# Panel 1: Forward predictions (1/5/10 day) from today
ax1 = axes[0]
ax1.set_facecolor("#1a1a2e")
context_window = raw.tail(30)
ax1.plot(context_window["timestamps"], context_window["close"],
         color="white", lw=2, label="Recent actual", zorder=5)
last_close = raw["close"].iloc[-1]
for horizon, pred in fwd_preds.items():
    # Bridge from last actual to first prediction
    bridge_dates = [last_date] + list(pred.index)
    bridge_vals  = [last_close] + list(pred["close"].values)
    pct = (pred["close"].iloc[-1] - last_close) / last_close * 100
    ax1.plot(bridge_dates, bridge_vals, color=colors_fwd[horizon], lw=2,
             linestyle="--", marker="o", ms=4,
             label=f"{horizon}-day forecast ({pct:+.1f}%)")
ax1.axvline(last_date, color="#666", lw=1, linestyle=":")
ax1.set_title("Forward Forecast — Kronos-mini (best model)", color="white", fontsize=11)
ax1.set_ylabel("Price", color="white")
ax1.tick_params(colors="white")
ax1.spines[:].set_color("#444")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.legend(fontsize=9, facecolor="#2a2a3e", labelcolor="white")
ax1.grid(True, alpha=0.2)

# Panel 2: Rolling 1-day model comparison vs actual
ax2 = axes[1]
ax2.set_facecolor("#1a1a2e")
ax2.plot(actuals_df.index, actuals_df["actual"], color="white", lw=2.5,
         marker="s", ms=6, label="Actual", zorder=5)
for name, df in all_results.items():
    mae = df["mae"].mean()
    dir_acc = df["dir_correct"].mean() * 100
    ax2.plot(df.index, df["predicted"], color=colors_model[name], lw=1.8,
             linestyle="--", marker="o", ms=5,
             label=f"Kronos-{name} ({params_map[name]})  MAE={mae:.2f}%  Dir={dir_acc:.0f}%")
    # Mark wrong-direction predictions with X
    wrong = df[~df["dir_correct"]]
    ax2.scatter(wrong.index, wrong["predicted"], marker="x", s=80,
                color=colors_model[name], zorder=6, linewidths=2)
ax2.set_title("Model Comparison — Rolling 1-Day Accuracy  (✗ = wrong direction)", color="white", fontsize=11)
ax2.set_ylabel("Price", color="white")
ax2.tick_params(colors="white")
ax2.spines[:].set_color("#444")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax2.legend(fontsize=9, facecolor="#2a2a3e", labelcolor="white")
ax2.grid(True, alpha=0.2)

# Panel 3: Error % bars
ax3 = axes[2]
ax3.set_facecolor("#1a1a2e")
x = range(N_DAYS)
width = 0.28
labels = [str(d.date()) for d in actuals_df.index]
for j, (name, df) in enumerate(all_results.items()):
    offset = (j - 1) * width
    ax3.bar([i + offset for i in x], df["error_pct"].values, width,
            label=f"Kronos-{name}", color=colors_model[name], alpha=0.85)
ax3.axhline(0, color="white", lw=0.8)
ax3.set_xticks(list(x))
ax3.set_xticklabels(labels, rotation=30, ha="right", fontsize=8, color="white")
ax3.set_ylabel("Error %", color="white")
ax3.tick_params(colors="white")
ax3.spines[:].set_color("#444")
ax3.legend(fontsize=9, facecolor="#2a2a3e", labelcolor="white")
ax3.grid(True, alpha=0.2, axis="y")
ax3.set_title("Daily Error % per Model  (e.g. +3% = predicted 3% too high, -3% = 3% too low)", color="white", fontsize=10)

summary = "  |  ".join([
    f"Kronos-{n}: MAE {all_results[n]['mae'].mean():.2f}%  Dir {all_results[n]['dir_correct'].mean()*100:.0f}%"
    for n in all_results
])
fig.text(0.5, 0.01, summary, ha="center", fontsize=10, fontweight="bold", color="white")

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
out = f"C:/Users/Dream/StockAI/model_compare_{TICKER}.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved: {out}")
print(f"\n{summary}")
