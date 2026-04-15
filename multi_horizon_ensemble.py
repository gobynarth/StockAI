"""
Multi-horizon ensemble: check if requiring agreement across horizons improves accuracy.
e.g. only trade when h=20 AND h=30 both predict same direction.
Uses sweep_checkpoints (best T=1.0, lb=200 configs).
"""
import os
import pandas as pd
from env_paths import base_path

SWEEP_DIR    = base_path("sweep_checkpoints")
EDGE_TICKERS = ["ENVX", "RIVN", "TSLA", "COIN"]
VALIDATION_N = 400
SELECTION_N  = 600
TEMP         = "10"   # T=1.0 -> "10"
LB           = 200

HORIZON_COMBOS = [
    ([10, 20],      "h10+h20"),
    ([10, 30],      "h10+h30"),
    ([20, 30],      "h20+h30"),
    ([10, 20, 30],  "h10+h20+h30"),
]

print(f"{'='*72}")
print(f"MULTI-HORIZON ENSEMBLE (base, T=1.0, lb={LB})")
print(f"{'='*72}")

for ticker in EDGE_TICKERS:
    print(f"\n--- {ticker} ---")
    print(f"{'Combo':<16} {'All N':>6} {'Single Best':>12} {'Agree N':>8} {'Agree%':>7} {'Agree Acc':>10} {'Lift':>6}")
    print(f"{'-'*68}")

    # Load all horizon DFs
    horizon_dfs = {}
    for h in [10, 20, 30]:
        cfg = f"h{h}_t{TEMP}_lb{LB}"
        path = f"{SWEEP_DIR}/{ticker}_{cfg}.csv"
        if not os.path.exists(path):
            # try with top_p/top_k suffix pattern
            import glob
            matches = glob.glob(f"{SWEEP_DIR}/{ticker}_h{h}_t{TEMP}_lb{LB}_*.csv")
            if matches:
                path = matches[0]
            else:
                break
        df = pd.read_csv(path, parse_dates=["date"])
        horizon_dfs[h] = df
    else:
        # all loaded
        for horizons, label in HORIZON_COMBOS:
            if not all(h in horizon_dfs for h in horizons):
                continue

            # Merge on date
            base = horizon_dfs[horizons[0]][["date","entry_close","pred_close","actual_close"]].copy()
            base = base.rename(columns={"pred_close": f"pred_h{horizons[0]}"})
            for h in horizons[1:]:
                m = horizon_dfs[h][["date","pred_close"]].rename(columns={"pred_close": f"pred_h{h}"})
                base = base.merge(m, on="date", how="inner")

            if len(base) < SELECTION_N + 50:
                continue

            oos = base.iloc[-VALIDATION_N:].copy()
            for h in horizons:
                oos[f"dir_h{h}"] = oos[f"pred_h{h}"] > oos["entry_close"]

            # All horizons agree
            dirs = [oos[f"dir_h{h}"] for h in horizons]
            oos["all_agree"] = dirs[0]
            for d in dirs[1:]:
                oos["all_agree"] = oos["all_agree"] == d
            # fix: all_agree should be True when all same direction
            oos["all_agree"] = True
            for h in horizons[1:]:
                oos["all_agree"] = oos["all_agree"] & (oos[f"dir_h{h}"] == oos[f"dir_h{horizons[0]}"])

            oos["actual_dir"] = oos["actual_close"] > oos["entry_close"]
            oos["correct"]    = oos[f"dir_h{horizons[0]}"] == oos["actual_dir"]

            agree_oos = oos[oos["all_agree"]]
            if len(agree_oos) < 10:
                continue

            # Single best horizon accuracy (h=30)
            best_h_df = horizon_dfs[max(horizons)]
            if len(best_h_df) >= SELECTION_N + VALIDATION_N:
                single_acc = best_h_df.iloc[-VALIDATION_N:]["correct"].mean() * 100
            else:
                single_acc = best_h_df.iloc[-min(VALIDATION_N, len(best_h_df)):]["correct"].mean() * 100

            agree_acc = agree_oos["correct"].mean() * 100
            agree_pct = len(agree_oos) / len(oos) * 100
            lift      = agree_acc - single_acc

            print(f"{label:<16} {len(oos):>6} {single_acc:>11.1f}% {len(agree_oos):>8} {agree_pct:>6.1f}% {agree_acc:>9.1f}% {lift:>+5.1f}%")
        continue
    print(f"  missing data for some horizons, skipping")

print(f"\n{'='*72}")
print("Done.")
