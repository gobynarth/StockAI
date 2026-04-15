# RunPod Revalidation

Use this to rerun the main StockAI research pipeline on a Linux GPU pod and compare the outputs with the current local results.

## Target

- Preferred GPU: `RTX 4090`
- Better if available: `H100`
- Disk: at least `50 GB`
- Container: standard PyTorch CUDA image is fine

## Repo layout expected on pod

- `/workspace/StockAI`
- `/workspace/Kronos`

The Python scripts now support these paths via env vars:

- `STOCKAI_BASE=/workspace/StockAI`
- `KRONOS_PATH=/workspace/Kronos`

## Clone

```bash
cd /workspace
git clone https://github.com/gobynarth/StockAI.git
git clone https://github.com/gobynarth/Kronos.git
cd /workspace/StockAI
```

If the repos are private, use your authenticated clone URL instead.

## Environment

```bash
export STOCKAI_BASE=/workspace/StockAI
export KRONOS_PATH=/workspace/Kronos
export PYTHONUNBUFFERED=1
```

Create a venv if you want:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the likely dependencies:

```bash
pip install --upgrade pip
pip install pandas numpy yfinance matplotlib torch transformers accelerate sentencepiece ib_insync
```

If `Kronos` has its own dependency file, install that too.

## Main rerun order

### 1. Core edge sweeps

```bash
python hyperparam_sweep.py ENVX
python hyperparam_sweep.py RIVN
python hyperparam_sweep.py TSLA
python hyperparam_sweep.py COIN
```

### 2. Derived analysis from those sweeps

```bash
python pnl_backtest.py
python pnl_backtest.py --long-short
python extended_horizons.py ENVX
python extended_horizons.py RIVN
python extended_horizons.py TSLA
python extended_horizons.py COIN
python sample_count_sweep.py ENVX
python sample_count_sweep.py RIVN
python sample_count_sweep.py TSLA
python sample_count_sweep.py COIN
python multi_horizon_ensemble.py
```

### 3. Broad screener and shortlist validation

```bash
python mass_screener_batched.py
python phase1_full_validation.py
python validate_tier1.py
```

### 4. Live daily signal regeneration

This is optional for pure research confirmation:

```bash
python live_signal.py
```

If you do not want email or IBKR interaction on the pod:

- do not set `GMAIL_APP_PASSWORD`
- do not run IB Gateway there

## What should match

The most important checks:

- `validate_tier1.py`
  - `PATH` stable `YES`
  - `HON` stable `YES`
  - `ITW` stable `NO`
  - `NCLH` stable `NO`
- `pnl_backtest.py`
  - older core names should remain directionally similar to current local outputs
- `ensemble_filter.py`
  - agree-only accuracy should be modestly better than all-signals accuracy for the core names

## Known caveats

- Results can shift if Yahoo data has changed since the cached local checkpoints were built.
- The huge compounded return numbers in `validate_tier1.py` are not realistic portfolio returns; compare direction accuracy and stability first.
- `live_signal.py` has been patched locally to avoid unvalidated shorts by default and to avoid same-day paper-trade reentry.

## Files to compare after pod run

- `sweep_checkpoints/`
- `ext_checkpoints/`
- `sample_checkpoints/`
- `screener_results.csv`
- `screener_survivors.csv`
- `ensemble_checkpoints/`
- `live_signals_YYYYMMDD.csv`

## Fast compare workflow

On the pod:

```bash
python validate_tier1.py | tee validate_tier1_runpod.txt
python pnl_backtest.py | tee pnl_backtest_runpod.txt
python ensemble_filter.py | tee ensemble_filter_runpod.txt
```

Then compare those text outputs against the local runs first before going deeper into every CSV.
