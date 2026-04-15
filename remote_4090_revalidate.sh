#!/bin/bash
set -euo pipefail

export STOCKAI_BASE=/workspace/StockAI
export KRONOS_PATH=/workspace/Kronos
export PYTHONUNBUFFERED=1

cd /workspace/StockAI

python3 hyperparam_sweep.py ENVX > rerun_hyperparam_ENVX.txt 2>&1
python3 hyperparam_sweep.py RIVN > rerun_hyperparam_RIVN.txt 2>&1
python3 hyperparam_sweep.py TSLA > rerun_hyperparam_TSLA.txt 2>&1
python3 hyperparam_sweep.py COIN > rerun_hyperparam_COIN.txt 2>&1

python3 pnl_backtest.py > rerun_pnl_backtest.txt 2>&1
python3 pnl_backtest.py --long-short > rerun_pnl_backtest_long_short.txt 2>&1
python3 extended_horizons.py ENVX > rerun_extended_horizons_ENVX.txt 2>&1
python3 extended_horizons.py RIVN > rerun_extended_horizons_RIVN.txt 2>&1
python3 extended_horizons.py TSLA > rerun_extended_horizons_TSLA.txt 2>&1
python3 extended_horizons.py COIN > rerun_extended_horizons_COIN.txt 2>&1
python3 sample_count_sweep.py ENVX > rerun_sample_count_ENVX.txt 2>&1
python3 sample_count_sweep.py RIVN > rerun_sample_count_RIVN.txt 2>&1
python3 sample_count_sweep.py TSLA > rerun_sample_count_TSLA.txt 2>&1
python3 sample_count_sweep.py COIN > rerun_sample_count_COIN.txt 2>&1
python3 multi_horizon_ensemble.py > rerun_multi_horizon_ensemble.txt 2>&1

python3 mass_screener_batched.py > rerun_mass_screener.txt 2>&1
python3 phase1_full_validation.py > rerun_phase1_full_validation.txt 2>&1
python3 validate_tier1.py > rerun_validate_tier1.txt 2>&1
