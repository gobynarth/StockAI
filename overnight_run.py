"""
Overnight orchestrator — full parameter sweep pipeline (batched, ~16x faster).
Focuses on finding the best parameter combination, not more tickers.

Order:
  1.  Hyperparam sweep COIN + TSLA (resume from checkpoints)
      — now includes top_p [0.8, 0.9, 0.95] and top_k [0, 10, 50]
  2.  Re-run RIVN + ENVX with new top_p/top_k configs
  3.  P&L backtest long-only (no GPU, instant)
  4.  P&L backtest long/short
  5.  Extended horizons h=40,60,90 on all edge tickers
  6.  Sample count sc=5,10,20 on all edge tickers (best config h=30,T=1.0)
  7.  Multi-horizon ensemble: load sweep results and find best h combo

Run: python overnight_run.py
"""
import subprocess, sys, os

PYTHON = sys.executable
BASE   = "C:/Users/Dream/StockAI"

def run(script, *args):
    cmd = [PYTHON, "-u", os.path.join(BASE, script)] + list(args)
    print(f"\n{'='*65}")
    print(f"RUNNING: {script} {' '.join(args)}")
    print(f"{'='*65}")
    result = subprocess.run(cmd, cwd=BASE)
    if result.returncode != 0:
        print(f"WARNING: exited {result.returncode}, continuing...")

EDGE_TICKERS = ["ENVX", "RIVN", "TSLA", "COIN"]

# 1-2. Full parameter sweep on all edge tickers
#      (h=10/20/30, T=0.5/0.7/1.0, lb=200/400, top_p=0.8/0.9/0.95, top_k=0/10/50)
for ticker in EDGE_TICKERS:
    run("hyperparam_sweep.py", ticker)

# 3-4. P&L backtest both modes
run("pnl_backtest.py")
run("pnl_backtest.py", "--long-short")

# 5. Extended horizons h=40,60,90 — does longer keep improving?
run("extended_horizons.py", "RIVN")   # most promising
run("extended_horizons.py", "ENVX")
run("extended_horizons.py", "TSLA")
run("extended_horizons.py", "COIN")

# 6. Sample count sc=5,10,20 on best config per ticker
for ticker in EDGE_TICKERS:
    run("sample_count_sweep.py", ticker)

# 7. Multi-horizon ensemble from sweep results
run("multi_horizon_ensemble.py")

print(f"\n{'='*65}")
print("OVERNIGHT RUN COMPLETE")
print(f"{'='*65}")
