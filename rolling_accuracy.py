"""
Rolling accuracy analysis: shows whether edge is stable over time or concentrated
in a specific period. Uses existing sweep checkpoints (best config per ticker).
No GPU needed.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SWEEP_DIR    = "C:/Users/Dream/StockAI/sweep_checkpoints"
CHECKPOINTS  = "C:/Users/Dream/StockAI/checkpoints"
EDGE_TICKERS = ["ENVX", "RIVN", "TSLA", "COIN"]
VALIDATION_N = 400
SELECTION_N  = 600
ROLL_WIN     = 50

# Best configs per ticker (from sweep results)
BEST_CONFIGS = {
    "ENVX": ("sweep", "ENVX_h30_t10_lb200"),
    "RIVN": ("sweep", "RIVN_h30_t10_lb200"),
    "TSLA": ("sweep", "TSLA_h20_t05_lb400"),
    "COIN": ("base",  "COIN_base"),          # fallback to original checkpoint
}

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("#12121f")
fig.suptitle("Rolling Direction Accuracy (50-window) -- Edge Tickers",
             fontsize=13, fontweight="bold", color="white")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

print(f"{'='*65}")
print(f"ROLLING ACCURACY ANALYSIS")
print(f"{'='*65}")

for idx, ticker in enumerate(EDGE_TICKERS):
    src, name = BEST_CONFIGS[ticker]

    if src == "sweep":
        # try new top_p/top_k format first, fall back to old
        import glob
        base_pattern = f"{SWEEP_DIR}/{name}_p*.csv"
        matches = glob.glob(base_pattern)
        if matches:
            # pick best OOS from available
            best_path, best_oos = None, 0
            for p in matches:
                df_tmp = pd.read_csv(p, parse_dates=["date"])
                if len(df_tmp) < VALIDATION_N: continue
                oos = df_tmp.iloc[-VALIDATION_N:]["correct"].mean() * 100
                if oos > best_oos:
                    best_oos = oos; best_path = p
            path = best_path
        else:
            path = f"{SWEEP_DIR}/{name}.csv"
    else:
        # original checkpoint format
        path = f"{CHECKPOINTS}/{name}.csv"
        if os.path.exists(path):
            df_raw = pd.read_csv(path, parse_dates=["date"])
            df_raw = df_raw[df_raw["horizon"] == 10].reset_index(drop=True)
            df_raw.to_csv("/tmp/_tmp.csv", index=False)
            path = "/tmp/_tmp.csv"

    if not path or not os.path.exists(path):
        print(f"  [{ticker}] no data found, skipping")
        continue

    df = pd.read_csv(path, parse_dates=["date"])
    if "horizon" in df.columns:
        df = df[df["horizon"] == df["horizon"].max()].reset_index(drop=True)

    roll = df["correct"].rolling(ROLL_WIN).mean() * 100
    overall = df["correct"].mean() * 100
    oos_acc = df.iloc[-VALIDATION_N:]["correct"].mean() * 100
    sel_acc = df.iloc[:SELECTION_N]["correct"].mean() * 100 if len(df) >= SELECTION_N else overall

    # Split into early/mid/late thirds
    n = len(df)
    third = n // 3
    early = df.iloc[:third]["correct"].mean() * 100
    mid   = df.iloc[third:2*third]["correct"].mean() * 100
    late  = df.iloc[2*third:]["correct"].mean() * 100

    print(f"\n  {ticker} ({os.path.basename(path).replace('.csv','')})")
    print(f"  Overall: {overall:.1f}%  |  Selection: {sel_acc:.1f}%  |  OOS: {oos_acc:.1f}%")
    print(f"  Early: {early:.1f}%  Mid: {mid:.1f}%  Late: {late:.1f}%")
    if late > early:
        print(f"  >> Edge IMPROVING over time (+{late-early:.1f}%)")
    elif early > late + 5:
        print(f"  >> Edge FADING over time ({early-late:.1f}% drop) -- be cautious")
    else:
        print(f"  >> Edge STABLE")

    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    ax.set_facecolor("#1a1a2e")
    ax.plot(df["date"], roll, lw=1.5, color="#3b82f6", label=f"Roll-{ROLL_WIN}")
    ax.axhline(50, color="#666", lw=1, linestyle=":")
    ax.axhline(oos_acc, color="#10b981", lw=1, linestyle="--", label=f"OOS avg {oos_acc:.1f}%")
    ax.fill_between(df["date"], roll, 50, where=(roll > 50), alpha=0.15, color="#10b981")
    ax.fill_between(df["date"], roll, 50, where=(roll < 50), alpha=0.15, color="#ef4444")
    # Mark selection/OOS boundary
    if len(df) > SELECTION_N:
        ax.axvline(df["date"].iloc[SELECTION_N], color="#f59e0b", lw=1, linestyle=":", alpha=0.7, label="OOS start")
    ax.set_title(f"{ticker}  (OOS={oos_acc:.1f}%)", color="white", fontsize=10)
    ax.set_ylim(20, 90)
    ax.tick_params(colors="white", labelsize=7)
    ax.spines[:].set_color("#444")
    ax.legend(fontsize=7, facecolor="#2a2a3e", labelcolor="white")
    ax.grid(True, alpha=0.15)

out = "C:/Users/Dream/StockAI/rolling_accuracy.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved: {out}")
