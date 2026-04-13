"""
Resume run: pick up what the overnight_run missed.
- COIN hyperparam sweep (was crashing due to CRYPTO set bug, now fixed)
- COIN extended horizons
- Sample count sc=5,10 on all tickers (ENVX sc5 done, sc10 partial)
- Multi-horizon ensemble
"""
import subprocess, sys, os
PYTHON = sys.executable
BASE   = "C:/Users/Dream/StockAI"

def run(script, *args):
    cmd = [PYTHON, "-u", os.path.join(BASE, script)] + list(args)
    print(f"\n{'='*60}")
    print(f"RUNNING: {script} {' '.join(args)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=BASE)
    if result.returncode != 0:
        print(f"WARNING: exited {result.returncode}, continuing...")

# COIN missed pieces
run("hyperparam_sweep.py", "COIN")
run("extended_horizons.py", "COIN")

# P&L backtests (fast, no GPU)
run("pnl_backtest.py")
run("pnl_backtest.py", "--long-short")

# Sample counts (ENVX sc5 done, sc10 partial → will resume from checkpoint)
for ticker in ["ENVX", "RIVN", "TSLA", "COIN"]:
    run("sample_count_sweep.py", ticker)

# Multi-horizon ensemble
run("multi_horizon_ensemble.py")

# Rolling accuracy analysis (no GPU)
run("rolling_accuracy.py")

# Live signals for today
run("live_signal.py")

print("\n" + "="*60)
print("FINISH RUN COMPLETE")
print("="*60)
