"""
Analyze sample count results: sc=1 (baseline from sweep) vs sc=5 vs sc=10
"""
import pandas as pd
import os

TICKERS = ["RIVN", "ENVX", "TSLA", "COIN"]
SAMPLE_COUNTS = [5, 10]
SWEEP_DIR = "C:/Users/Dream/StockAI/sweep_checkpoints"
SAMPLE_DIR = "C:/Users/Dream/StockAI/sample_checkpoints"
VALIDATION_N = 400

def load_oos_accuracy(path, n_oos=VALIDATION_N):
    if not os.path.exists(path):
        return None, 0
    df = pd.read_csv(path)
    if len(df) == 0:
        return None, 0
    oos = df.tail(n_oos)
    return oos["correct"].mean() * 100, len(oos)

print("Sample Count Analysis (sc=1 vs sc=5 vs sc=10)")
print("=" * 60)
print(f"{'Ticker':<8} {'sc=1 (base)':>12} {'sc=5':>8} {'sc=10':>8} {'Best':>8}")
print("-" * 60)

for ticker in TICKERS:
    # sc=1 baseline: use best h=30 config from sweep
    # The sweep file for h=30 T=1.0 lb=200 (best config for most)
    # Filenames in sweep_checkpoints: {ticker}_h30_t10_lb200_p90_k0.csv or similar
    # Find the best sweep file
    sweep_files = [f for f in os.listdir(SWEEP_DIR) if f.startswith(f"{ticker}_h30_t10_lb200")]
    best_sc1_acc = None
    for sf in sweep_files:
        acc, n = load_oos_accuracy(os.path.join(SWEEP_DIR, sf))
        if acc and (best_sc1_acc is None or acc > best_sc1_acc):
            best_sc1_acc = acc

    sc_accs = {}
    for sc in SAMPLE_COUNTS:
        path = os.path.join(SAMPLE_DIR, f"{ticker}_h30_t10_lb200_sc{sc}.csv")
        acc, n = load_oos_accuracy(path)
        sc_accs[sc] = (acc, n)

    # find best overall
    all_vals = []
    if best_sc1_acc:
        all_vals.append((1, best_sc1_acc))
    for sc, (a, n) in sc_accs.items():
        if a:
            all_vals.append((sc, a))
    best_sc, best_acc = max(all_vals, key=lambda x: x[1]) if all_vals else (None, None)

    sc1_str = f"{best_sc1_acc:.1f}%" if best_sc1_acc else "N/A"
    sc5_str = f"{sc_accs[5][0]:.1f}% ({sc_accs[5][1]})" if sc_accs[5][0] else "N/A"
    sc10_str = f"{sc_accs[10][0]:.1f}% ({sc_accs[10][1]})" if sc_accs[10][0] else "N/A"

    print(f"{ticker:<8} {sc1_str:>12} {sc5_str:>8} {sc10_str:>8}  best=sc{best_sc}({best_acc:.1f}%)" if best_sc else f"{ticker:<8} {sc1_str:>12} {sc5_str:>8} {sc10_str:>8}")

print()
print("sc=N means averaging N separate Kronos predictions per window.")
print("Higher sc reduces variance but adds compute; worth it if acc improves >1%.")
