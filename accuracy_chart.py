import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from datetime import timedelta
sys.path.append("C:/Users/Dream/Kronos")
from model import Kronos, KronosTokenizer, KronosPredictor

TICKER = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
LOOKBACK = 400

print(f"Fetching {TICKER}...")
raw = yf.download(TICKER, period="3y", interval="1d", auto_adjust=True, progress=False)
raw = raw.reset_index()
raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
raw = raw.rename(columns={"date": "timestamps"})
raw["amount"] = raw["close"] * raw["volume"]
raw = raw[["timestamps", "open", "high", "low", "close", "volume", "amount"]].dropna()
raw["timestamps"] = pd.to_datetime(raw["timestamps"])

print("Loading Kronos model...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)

# --- 10-day prediction (same as before, use last LOOKBACK candles) ---
print("Running 10-day prediction...")
df10 = raw.tail(LOOKBACK + 10).reset_index(drop=True)
x_df10 = df10.loc[:LOOKBACK-1, ["open", "high", "low", "close", "volume", "amount"]]
x_ts10 = df10.loc[:LOOKBACK-1, "timestamps"]
last_date10 = df10.loc[LOOKBACK-1, "timestamps"]
y_ts10 = pd.Series(pd.bdate_range(start=last_date10 + timedelta(days=1), periods=10))
pred10 = predictor.predict(df=x_df10, x_timestamp=x_ts10, y_timestamp=y_ts10,
                           pred_len=10, T=1.0, top_p=0.9, sample_count=1, verbose=False)
pred10.index = y_ts10.values

# --- Rolling 1-day predictions over last 10 trading days ---
print("Running rolling 1-day predictions (10 windows)...")
one_day_results = []
# We need actual data for those 10 days too
test_start_idx = len(raw) - 10
for i in range(10):
    idx = test_start_idx + i
    x_df1 = raw.iloc[idx - LOOKBACK:idx][["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    x_ts1 = raw.iloc[idx - LOOKBACK:idx]["timestamps"].reset_index(drop=True)
    next_date = pd.bdate_range(start=raw.iloc[idx]["timestamps"] + timedelta(days=1), periods=1)
    y_ts1 = pd.Series(next_date)
    pred1 = predictor.predict(df=x_df1, x_timestamp=x_ts1, y_timestamp=y_ts1,
                              pred_len=1, T=1.0, top_p=0.9, sample_count=1, verbose=False)
    actual_idx = idx + 1
    actual_close = raw.iloc[actual_idx]["close"] if actual_idx < len(raw) else None
    one_day_results.append({
        "date": next_date[0],
        "predicted": pred1["close"].iloc[0],
        "actual": actual_close,
    })
    actual_str = f"{actual_close:.2f}" if actual_close is not None else "N/A"
    print(f"  Day {i+1}/10: predicted {pred1['close'].iloc[0]:.2f}, actual {actual_str}")

df1 = pd.DataFrame(one_day_results).set_index("date")
df1 = df1.dropna(subset=["actual"])
df1["error_pct"] = (df1["predicted"] - df1["actual"]) / df1["actual"] * 100

# --- actual prices for 10-day window ---
act_dates = raw.tail(10)["timestamps"].values
act_closes = raw.tail(10)["close"].values

# --- Plot ---
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle(f"{TICKER} — Kronos Prediction Accuracy", fontsize=15, fontweight="bold")

# Panel 1: 10-day prediction vs actual
ax1 = axes[0]
context_tail = raw.tail(30)
ax1.plot(context_tail["timestamps"], context_tail["close"], color="#4a90d9", lw=1.5, label="Actual (context)")
ax1.plot(pred10.index, pred10["close"], color="red", lw=2, linestyle="--", marker="o", ms=4, label="10-day prediction")
ax1.plot(act_dates[-10:], act_closes[-10:], color="green", lw=2, marker="s", ms=4, label="Actual (pred window)")
ax1.set_title("10-Day Prediction vs Actual", fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylabel("Price ($)")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

# Panel 2: 1-day rolling predictions vs actual
ax2 = axes[1]
ax2.plot(df1.index, df1["actual"], color="green", lw=2, marker="s", ms=5, label="Actual")
ax2.plot(df1.index, df1["predicted"], color="orange", lw=2, linestyle="--", marker="o", ms=5, label="1-day prediction")
ax2.set_title("1-Day Rolling Prediction vs Actual (last 10 trading days)", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylabel("Price ($)")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

# Panel 3: Error % comparison
ax3 = axes[2]
x = range(len(df1))
width = 0.35
actual_vals = df1["actual"].values
pred10_vals = pred10["close"].values[:len(df1)]
err10 = (pred10_vals - actual_vals) / actual_vals * 100
err1 = df1["error_pct"].values
labels = [str(d.date()) for d in df1.index]
bars1 = ax3.bar([i - width/2 for i in x], err1, width, label="1-day error", color="orange", alpha=0.8)
bars10 = ax3.bar([i + width/2 for i in x], err10, width, label="10-day error", color="red", alpha=0.8)
ax3.axhline(0, color="black", lw=0.8)
ax3.set_xticks(list(x))
ax3.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
ax3.set_title("Prediction Error % (1-day vs 10-day)", fontsize=12)
ax3.set_ylabel("Error %")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis="y")

mae1 = df1["error_pct"].abs().mean()
mae10 = pd.Series(err10).abs().mean()
fig.text(0.5, 0.01, f"Mean Absolute Error — 1-day: {mae1:.2f}%  |  10-day: {mae10:.2f}%",
         ha="center", fontsize=11, fontweight="bold")

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("C:/Users/Dream/StockAI/accuracy_chart.png", dpi=150, bbox_inches="tight")
print(f"\nChart saved to accuracy_chart.png")
print(f"\nMean Absolute Error — 1-day: {mae1:.2f}%  |  10-day: {mae10:.2f}%")
