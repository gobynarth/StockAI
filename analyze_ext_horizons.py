"""
Analyze extended horizons results: h=30/40/60/90 for RIVN, ENVX, TSLA, COIN
Compare OOS accuracy across horizons.
"""
import pandas as pd
import os

TICKERS = ["RIVN", "ENVX", "TSLA", "COIN"]
HORIZONS = [30, 40, 60, 90]
DIR = "C:/Users/Dream/StockAI/ext_checkpoints"
VALIDATION_N = 400

def load_oos_accuracy(path, n_oos=VALIDATION_N):
    df = pd.read_csv(path)
    if len(df) == 0:
        return None, 0
    oos = df.tail(n_oos)
    acc = oos["correct"].mean() * 100
    return acc, len(oos)

print("Extended Horizons Analysis (h=30/40/60/90)")
print("=" * 60)
print(f"{'Ticker':<8} {'h=30':>8} {'h=40':>8} {'h=60':>8} {'h=90':>8} {'Best':>8}")
print("-" * 60)

best_configs = {}
for ticker in TICKERS:
    accs = {}
    for h in HORIZONS:
        fname = f"{ticker}_h{h}_t10_lb200.csv"
        path = os.path.join(DIR, fname)
        if not os.path.exists(path):
            accs[h] = None
            continue
        acc, n = load_oos_accuracy(path)
        accs[h] = acc

    vals = [(h, a) for h, a in accs.items() if a is not None]
    best_h, best_acc = max(vals, key=lambda x: x[1]) if vals else (None, None)
    best_configs[ticker] = (best_h, best_acc)

    row = f"{ticker:<8}"
    for h in HORIZONS:
        a = accs.get(h)
        s = f"{a:.1f}%" if a is not None else "  N/A"
        row += f" {s:>8}"
    row += f" h={best_h}({best_acc:.1f}%)" if best_h else ""
    print(row)

print()
print("Key findings:")
for ticker, (bh, ba) in best_configs.items():
    # compare to h=30 baseline
    h30_path = os.path.join(DIR, f"{ticker}_h30_t10_lb200.csv")
    h30_acc, _ = load_oos_accuracy(h30_path) if os.path.exists(h30_path) else (None, 0)
    if h30_acc and ba:
        diff = ba - h30_acc
        sign = "+" if diff >= 0 else ""
        print(f"  {ticker}: best h={bh} ({ba:.1f}%) vs h=30 ({h30_acc:.1f}%) = {sign}{diff:.1f}%")
