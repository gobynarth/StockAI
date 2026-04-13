import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
from datetime import timedelta
sys.path.append("C:/Users/Dream/Kronos")
from model import Kronos, KronosTokenizer, KronosPredictor

TICKER    = sys.argv[1] if len(sys.argv) > 1 else "TSLA"
N_WINDOWS = 1000
LOOKBACK  = 400
HORIZONS  = [1, 5, 10]
CHECKPOINT_DIR = "C:/Users/Dream/StockAI/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODELS = [
    {"name": "mini",  "model_id": "NeoQuasar/Kronos-mini",  "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",  "max_context": 2048, "params": "4.1M"},
    {"name": "small", "model_id": "NeoQuasar/Kronos-small", "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base","max_context": 512,  "params": "24.7M"},
    {"name": "base",  "model_id": "NeoQuasar/Kronos-base",  "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base","max_context": 512,  "params": "102.3M"},
]

CRYPTO = {"BTC", "SOL", "TAO", "ETH", "DOGE", "XRP"}
yf_ticker = f"{TICKER}-USD" if TICKER.upper() in CRYPTO else TICKER
is_crypto = TICKER.upper() in CRYPTO

def next_dates(from_date, n):
    if is_crypto:
        return pd.date_range(start=from_date + timedelta(days=1), periods=n)
    return pd.bdate_range(start=from_date + timedelta(days=1), periods=n)

csv_path = f"C:/Users/Dream/StockAI/data/{TICKER}.csv"
if os.path.exists(csv_path):
    print(f"Loading {TICKER} from local CSV ({csv_path})...")
    raw = pd.read_csv(csv_path, parse_dates=["timestamps"])
    raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)
else:
    print(f"Fetching {TICKER}...")
    raw = yf.download(yf_ticker, period="5y", interval="1d", auto_adjust=True, progress=False, timeout=30)
    raw = raw.reset_index()
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    date_col = "date" if "date" in raw.columns else "datetime"
    raw = raw.rename(columns={date_col: "timestamps"})
    raw["amount"] = raw["close"] * raw["volume"]
    raw["timestamps"] = pd.to_datetime(raw["timestamps"]).dt.tz_localize(None)

raw = raw[["timestamps", "open", "high", "low", "close", "volume", "amount"]].dropna()

max_h = max(HORIZONS)
required = LOOKBACK + max_h + N_WINDOWS
if len(raw) < required:
    actual_windows = len(raw) - LOOKBACK - max_h
    print(f"Only {len(raw)} rows, reducing to {actual_windows} windows")
    N_WINDOWS = actual_windows

test_start = len(raw) - N_WINDOWS - max_h
all_results = {}

for cfg in MODELS:
    checkpoint_path = f"{CHECKPOINT_DIR}/{TICKER}_{cfg['name']}.csv"

    # Resume from checkpoint if exists
    if os.path.exists(checkpoint_path):
        existing = pd.read_csv(checkpoint_path, parse_dates=["date"])
        completed = len(existing) // len(HORIZONS)
        print(f"\n[{cfg['name']}] Resuming from checkpoint ({completed}/{N_WINDOWS} done)")
        for h in HORIZONS:
            all_results[(cfg["name"], h)] = existing[existing["horizon"] == h].drop(columns="horizon").reset_index(drop=True)
        if completed >= N_WINDOWS:
            print(f"  [{cfg['name']}] Already complete, skipping.")
            continue
        start_i = completed
    else:
        start_i = 0

    print(f"\nLoading Kronos-{cfg['name']} ({cfg['params']})...")
    tokenizer = KronosTokenizer.from_pretrained(cfg["tokenizer_id"])
    model_obj = Kronos.from_pretrained(cfg["model_id"])
    predictor = KronosPredictor(model_obj, tokenizer, max_context=cfg["max_context"])

    rows = {h: (all_results[(cfg["name"], h)].to_dict("records") if (cfg["name"], h) in all_results else []) for h in HORIZONS}
    lb = min(LOOKBACK, cfg["max_context"])
    BATCH_SIZE = 16

    batch_x_dfs, batch_x_ts_list, batch_y_ts_list, batch_meta = [], [], [], []

    def flush_batch():
        if not batch_x_dfs:
            return
        preds = predictor.predict_batch(
            df_list=batch_x_dfs, x_timestamp_list=batch_x_ts_list,
            y_timestamp_list=batch_y_ts_list, pred_len=max_h,
            T=1.0, top_p=0.9, sample_count=1, verbose=False)
        for j, (bi, bidx, bentry_close, bentry_date) in enumerate(batch_meta):
            for h in HORIZONS:
                actual_idx = bidx + h
                if actual_idx >= len(raw): continue
                pred_close   = preds[j]["close"].iloc[h - 1]
                actual_close = raw.iloc[actual_idx]["close"]
                rows[h].append({
                    "date": bentry_date, "entry_close": bentry_close,
                    "pred_close": pred_close, "actual_close": actual_close,
                    "correct": (pred_close > bentry_close) == (actual_close > bentry_close),
                    "error_pct": (pred_close - actual_close) / actual_close * 100,
                })
        batch_x_dfs.clear(); batch_x_ts_list.clear()
        batch_y_ts_list.clear(); batch_meta.clear()

    for i in range(start_i, N_WINDOWS):
        idx = test_start + i
        entry_close = raw.iloc[idx]["close"]
        entry_date  = raw.iloc[idx]["timestamps"]

        x_df = raw.iloc[idx - lb:idx][["open","high","low","close","volume","amount"]].reset_index(drop=True)
        x_ts = raw.iloc[idx - lb:idx]["timestamps"].reset_index(drop=True)
        y_ts = pd.Series(next_dates(entry_date, max_h))

        batch_x_dfs.append(x_df); batch_x_ts_list.append(x_ts)
        batch_y_ts_list.append(y_ts); batch_meta.append((i, idx, entry_close, entry_date))

        if len(batch_x_dfs) >= BATCH_SIZE or i == N_WINDOWS - 1:
            flush_batch()

        if (i + 1) % 50 == 0:
            print(f"  [{cfg['name']}] {i+1}/{N_WINDOWS}")
            frames = []
            for h in HORIZONS:
                df_h = pd.DataFrame(rows[h])
                df_h["horizon"] = h
                frames.append(df_h)
            pd.concat(frames).to_csv(checkpoint_path, index=False)

    for h in HORIZONS:
        all_results[(cfg["name"], h)] = pd.DataFrame(rows[h])

    # Final checkpoint save
    frames = []
    for h in HORIZONS:
        df_h = pd.DataFrame(all_results[(cfg["name"], h)])
        df_h["horizon"] = h
        frames.append(df_h)
    pd.concat(frames).to_csv(checkpoint_path, index=False)
    print(f"  [{cfg['name']}] Checkpoint saved.")

    del model_obj, predictor

# ---- Train/test split ----
SELECTION_N  = 600
VALIDATION_N = 400

print(f"\n{'='*60}")
print(f"SELECTION PHASE (first {SELECTION_N} windows)")
print(f"{'='*60}")
print(f"\n{'Model':<8} {'Horizon':<10} {'Dir Acc':>8} {'MAE':>8}")
print("-" * 40)
best_dir = ("", 0, 0.0)
for cfg in MODELS:
    for h in HORIZONS:
        df = all_results[(cfg["name"], h)].iloc[:SELECTION_N]
        dir_acc = df["correct"].mean() * 100
        mae     = df["error_pct"].abs().mean()
        print(f"{cfg['name']:<8} {h}-day{'':<6} {dir_acc:>7.1f}% {mae:>7.2f}%")
        if dir_acc > best_dir[2]:
            best_dir = (cfg["name"], h, dir_acc)

print(f"\n>> Best combo: Kronos-{best_dir[0]} {best_dir[1]}-day  ({best_dir[2]:.1f}% dir acc in selection)")

print(f"\n{'='*60}")
print(f"OUT-OF-SAMPLE VALIDATION (last {VALIDATION_N} windows)")
print(f"{'='*60}")
val_df  = all_results[(best_dir[0], best_dir[1])].iloc[-VALIDATION_N:]
val_dir = val_df["correct"].mean() * 100
val_mae = val_df["error_pct"].abs().mean()
print(f"Kronos-{best_dir[0]} {best_dir[1]}-day  Dir Acc: {val_dir:.1f}%  MAE: {val_mae:.2f}%")
print(f"Correct: {int(val_df['correct'].sum())} / {len(val_df)}")
if val_dir > 55:
    print("EDGE HOLDS out-of-sample")
elif val_dir > 50:
    print("WEAK EDGE out-of-sample")
else:
    print("NO EDGE out-of-sample -- likely overfit to selection period")

# ---- Plot ----
colors_model = {"mini": "#f59e0b", "small": "#3b82f6", "base": "#10b981"}
style_h      = {1: "-", 5: "--", 10: ":"}

fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor("#12121f")
fig.suptitle(f"{TICKER} -- Full Backtest ({N_WINDOWS} windows, all models x horizons)",
             fontsize=14, fontweight="bold", color="white")

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.3)

for col, cfg in enumerate(MODELS):
    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor("#1a1a2e")
    ax.axhline(50, color="#666", lw=1, linestyle=":")
    for h in HORIZONS:
        df  = all_results[(cfg["name"], h)]
        rol = df["correct"].rolling(50).mean() * 100
        dir_acc = df["correct"].mean() * 100
        ax.plot(df["date"], rol, lw=1.5, linestyle=style_h[h], label=f"{h}-day {dir_acc:.0f}%")
    ax.set_title(f"Kronos-{cfg['name']} ({cfg['params']})\nDir Accuracy (roll-50)", color="white", fontsize=9)
    ax.set_ylim(0, 100)
    ax.tick_params(colors="white", labelsize=7)
    ax.spines[:].set_color("#444")
    ax.legend(fontsize=7, facecolor="#2a2a3e", labelcolor="white")
    ax.grid(True, alpha=0.15)

for col, cfg in enumerate(MODELS):
    ax = fig.add_subplot(gs[1, col])
    ax.set_facecolor("#1a1a2e")
    for h in HORIZONS:
        df  = all_results[(cfg["name"], h)]
        rol = df["error_pct"].abs().rolling(50).mean()
        mae = df["error_pct"].abs().mean()
        ax.plot(df["date"], rol, lw=1.5, linestyle=style_h[h], label=f"{h}-day MAE={mae:.1f}%")
    ax.set_title(f"Kronos-{cfg['name']}\nPrice Error % (roll-50)", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=7)
    ax.spines[:].set_color("#444")
    ax.legend(fontsize=7, facecolor="#2a2a3e", labelcolor="white")
    ax.grid(True, alpha=0.15)

model_names = [c["name"] for c in MODELS]
ax_sel = fig.add_subplot(gs[2, :2])
ax_oos = fig.add_subplot(gs[2, 2])

for ax, label, n, offset in [
    (ax_sel, f"Selection (first {SELECTION_N})", SELECTION_N, 0),
    (ax_oos, f"Out-of-Sample (last {VALIDATION_N})", VALIDATION_N, -VALIDATION_N),
]:
    ax.set_facecolor("#1a1a2e")
    if offset == 0:
        heat_data = np.array([[all_results[(m, h)].iloc[:n]["correct"].mean() * 100
                               for h in HORIZONS] for m in model_names])
    else:
        heat_data = np.array([[all_results[(m, h)].iloc[offset:]["correct"].mean() * 100
                               for h in HORIZONS] for m in model_names])
    im = ax.imshow(heat_data, cmap="RdYlGn", vmin=30, vmax=80, aspect="auto")
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([f"{h}-day" for h in HORIZONS], color="white", fontsize=10)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels([f"Kronos-{m}" for m in model_names], color="white", fontsize=10)
    ax.set_title(f"Dir Accuracy -- {label}", color="white", fontsize=9)
    for i, m in enumerate(model_names):
        for j, h in enumerate(HORIZONS):
            val = heat_data[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    color="black" if 40 < val < 70 else "white", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.04)

out = f"C:/Users/Dream/StockAI/full_backtest_{TICKER}.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out}")
