## H100 Revalidation Status

Date: 2026-04-14
Repo: `C:\Users\Dream\projects\StockAI`
Remote pod checked: `root@216.243.220.224:10002`

### What was confirmed on the H100

- The pod checkout is a live revalidation workspace, not a clean clone.
- It contains local script edits plus added files:
  - `env_paths.py`
  - `mass_screener_batched.py`
  - `phase1_full_validation.py`
  - `validate_tier1.py`
  - `ensemble_sweep_pod.py`
  - `RUNPOD_REVALIDATION.md`
- Existing artifacts were present:
  - `checkpoints/` for `ENVX`, `RIVN`, `TSLA`, `COIN` across `mini/small/base`
  - `ensemble_checkpoints/` for `PATH`, `ITW`, `HON`, `NCLH`, `HD`, `WM`
  - `screener_checkpoints/`
  - `screener_results.csv`

### What was rerun successfully

#### `validate_tier1.py`

This reproduced the key claimed conclusions:

- `PATH`: walk-forward `YES`
- `HON`: walk-forward `YES`
- `ITW`: walk-forward `NO`
- `NCLH`: walk-forward `NO`

Final summary observed from the pod:

- `PATH`: agree acc `82.8%`, strategy `FIXED`, `TP 20% / SL 10%`, Sharpe `1.13`
- `ITW`: agree acc `76.0%`, unstable walk-forward, Sharpe `4.83`
- `HON`: agree acc `70.5%`, stable, Sharpe `2.04`
- `NCLH`: agree acc `71.0%`, unstable walk-forward, Sharpe `3.22`

#### `phase1_full_validation.py`

This completed and wrote `screener_survivors.csv` on the pod.

The reproduced top-20 survivors did **not** match the earlier shortlist artifact exactly.

Observed survivors included:

- `ANF`
- `BITF`
- `AAL`
- `ABNB`
- `BILL`
- `AR`
- `AMZN`
- `AAPL`
- `BDX`
- `BA`
- `ANET`
- `BIDU`
- `AFRM`

Important result:

- `BITF` remained `KEEP`
- best strategy was `TRAIL_3`
- Sharpe was about `0.72`
- `needs_vix_filter=True`

### Interpretation

- Tier 1 ensemble validation mostly reproduced Claude's claims.
- Broad screener / shortlist reproduction did **not** cleanly reproduce the earlier shortlist.
- Most likely causes:
  - stale checkpoints vs fresh rerun differences
  - changed Yahoo data
  - later filtering logic diverged from the saved shortlist

### Attempted next step

Started upstream rerun attempts for:

- `python3 hyperparam_sweep.py ENVX`
- `python3 hyperparam_sweep.py RIVN`

But the session was interrupted and the pod later became unreachable (`ssh ... port 10002: Connection timed out`), so those reruns were not confirmed complete.

### What still needs rerun for true end-to-end revalidation

1. Core sweeps:
   - `python3 hyperparam_sweep.py ENVX`
   - `python3 hyperparam_sweep.py RIVN`
   - `python3 hyperparam_sweep.py TSLA`
   - `python3 hyperparam_sweep.py COIN`
2. Broad screener:
   - `python3 mass_screener_batched.py`
3. Then rerun:
   - `python3 phase1_full_validation.py`
   - `python3 validate_tier1.py`

### Recommendation

- If the H100 is billable and currently unreachable or idle, it can be stopped.
- When ready to resume true revalidation, bring up a fresh GPU pod and continue from the rerun list above.

## 4090 Handoff State

Date: 2026-04-15
New host: `root@149.36.1.193 -p 49486`

### What was done on the 4090

- Connected successfully.
- Confirmed GPU:
  - `NVIDIA GeForce RTX 4090`
  - ~`24 GB` VRAM
- Confirmed Python:
  - `Python 3.11.10`
- Cloned:
  - `/workspace/StockAI`
  - `/workspace/Kronos`

### Important repo mismatch found

- The GitHub `StockAI` clone is older than the current local working tree.
- Example:
  - remote `hyperparam_sweep.py` still had hardcoded Windows paths
  - local workspace version had Linux-safe `env_paths.py` support

### Overlay prepared locally

Created local file:

- `C:\Users\Dream\projects\StockAI\remote_4090_revalidate.sh`

Created local archive:

- `C:\Users\Dream\AppData\Local\Temp\stockai_4090_overlay.zip`

This overlay contains the updated Linux-safe files:

- `env_paths.py`
- `hyperparam_sweep.py`
- `pnl_backtest.py`
- `extended_horizons.py`
- `sample_count_sweep.py`
- `multi_horizon_ensemble.py`
- `ensemble_filter.py`
- `mass_screener_batched.py`
- `phase1_full_validation.py`
- `validate_tier1.py`
- `full_backtest.py`
- `ensemble_sweep_pod.py`
- `RUNPOD_REVALIDATION.md`
- `remote_4090_revalidate.sh`

### Remote 4090 current state

- `/workspace/stockai_4090_overlay.zip` was uploaded successfully.
- Attempted unattended launch command returned:
  - `started_4090_pipeline`
  - then failed immediately with:
    - `bash: line 1: unzip: command not found`

### Meaning

- The 4090 pipeline is **not** actually running yet.
- The main blocker is just missing `unzip` on the remote container.
- Best next step from a Linux/Ubuntu session:
  1. install `unzip` or unpack with Python
  2. extract `/workspace/stockai_4090_overlay.zip` into `/workspace/StockAI`
  3. install dependencies
  4. run `remote_4090_revalidate.sh` under `nohup`

### Remote paths to reuse

- `/workspace/StockAI`
- `/workspace/Kronos`
- `/workspace/stockai_4090_overlay.zip`
- expected pipeline log:
  - `/workspace/StockAI/revalidate_4090.out`
