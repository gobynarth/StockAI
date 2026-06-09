#!/bin/bash
cd /workspace/StockAI || exit 1
while ps -eo cmd | grep hyperparam_sweep.py | grep -v grep >/dev/null; do
  sleep 60
done
export PYTHONUNBUFFERED=1
python3 hyperparam_sweep.py COIN > rerun_hyperparam_COIN.txt 2>&1
python3 mass_screener_batched.py > rerun_mass_screener.txt 2>&1
python3 phase1_full_validation.py > rerun_phase1_full_validation.txt 2>&1
python3 validate_tier1.py > rerun_validate_tier1.txt 2>&1
