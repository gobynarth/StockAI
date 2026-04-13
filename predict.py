import sys
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
sys.path.append("C:/Users/Dream/Kronos")
from model import Kronos, KronosTokenizer, KronosPredictor

TICKER = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
LOOKBACK = 400  # candles of history to feed
PRED_LEN = 10   # candles to predict forward

print(f"\nFetching {TICKER} daily data...")
raw = yf.download(TICKER, period="3y", interval="1d", auto_adjust=True, progress=False)
raw = raw.reset_index()
raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
raw = raw.rename(columns={"date": "timestamps"})
raw["amount"] = raw["close"] * raw["volume"]
raw = raw[["timestamps", "open", "high", "low", "close", "volume", "amount"]].dropna()
raw["timestamps"] = pd.to_datetime(raw["timestamps"])

if len(raw) < LOOKBACK + PRED_LEN:
    print(f"Not enough data: got {len(raw)} rows, need {LOOKBACK + PRED_LEN}")
    sys.exit(1)

print("Loading Kronos model (downloads on first run)...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)

# Use most recent LOOKBACK candles as context
df = raw.tail(LOOKBACK + PRED_LEN).reset_index(drop=True)
x_df = df.loc[:LOOKBACK-1, ["open", "high", "low", "close", "volume", "amount"]]
x_timestamp = df.loc[:LOOKBACK-1, "timestamps"]
# Future timestamps: extend by PRED_LEN business days
last_date = df.loc[LOOKBACK-1, "timestamps"]
future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=PRED_LEN)
y_timestamp = pd.Series(future_dates)

print(f"Running prediction for next {PRED_LEN} trading days...\n")
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=PRED_LEN,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=False,
)

last_close = x_df["close"].iloc[-1]
pred_df.index = future_dates

print(f"{'Date':<14} {'Pred Close':>12} {'Change %':>10}")
print("-" * 38)
prev = last_close
for date, row in pred_df.iterrows():
    pct = (row["close"] - prev) / prev * 100
    print(f"{str(date.date()):<14} {row['close']:>12.2f} {pct:>+9.2f}%")
    prev = row["close"]

total_pct = (pred_df["close"].iloc[-1] - last_close) / last_close * 100
print("-" * 38)
print(f"{'Last close:':<14} {last_close:>12.2f}")
print(f"{'Total change:':<14} {total_pct:>+11.2f}%")
