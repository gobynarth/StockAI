# Handoff

## Goals
- Replace the biased research flow with honest walk-forward validation.
- Keep the live set conservative and only promote names that survive honest retesting and exit-design checks.
- After stabilizing live, run a full honest screener rerun from scratch.

## Current State
- Honest research pipeline was added locally:
  - `research_pipeline.py`
  - `honest_walk_forward.py`
  - tests for splits, selection, execution simulation, CLI, checkpoint loading, and live candidate loading.
- Local verification passed: `10` tests passing.
- Live candidate loading is now file-backed via:
  - `approved_live_candidates.csv`
  - `approved_pending_candidates.csv`
  - `live_candidates.py`
- `live_signal.py` now uses the approved live-candidate file instead of the original hardcoded set.
- Approved live set is now:
  - `RIVN`
  - `BITF`
  - `PATH`
  - `HON`
  - `ITW`
  - `NCLH`
- Promotion results already obtained:
  - `ITW` promoted with fixed exits, conservative config `tp=0.20`, `sl=0.10`, `alloc=0.02`
  - `NCLH` promoted with fixed exits, conservative config `tp=0.25`, `sl=0.10`, `alloc=0.02`
- 4090 pod is active and reachable:
  - host: `root@213.181.111.2 -p 48484`
  - workspace: `/root/StockAI_sync/StockAI`
- 4090 outputs present under `/root/StockAI_sync/StockAI/honest_results`:
  - `survivors_mixed_honest_walk_forward.csv/json`
  - `screener_survivors_honest_walk_forward.csv/json`
  - `screener_deeper_recheck.csv/json`
  - `itw_nclh_exit_promotion_test.json`
  - `itw_nclh_exit_promotion_test_small_windows.json`
- 4090 smoke run of `live_signal.py` now works with minimal Kronos runtime and installed deps.
- Smoke run outcome:
  - models loaded
  - signals generated
  - output written to `/root/StockAI_sync/StockAI/live_signals_20260417.csv`
  - IBKR failed as expected because no gateway was running on `127.0.0.1:4002`
  - email skipped because `GMAIL_APP_PASSWORD` was not set

## Known Issues / Cleanup
- `live_signal.py` still prints a stale hardcoded summary line mentioning removed names like `ENVX` and `TSLA`.
- The paper trade log still contains an old `TSLA` open trade, which is historical state and should be cleaned or reconciled before trusting the portfolio summary.
- The 4090 was only prepared as a smoke environment, not a true production broker host.

## Next Steps
1. Clean the stale hardcoded summary text in `live_signal.py` so it reflects the current approved live set.
2. Clean or reconcile `paper_trades.csv` so removed names do not appear as active holdings.
3. Optionally run one more local/remote smoke test after the cleanup.
4. Then start the full honest screener rerun from scratch on the 4090.
5. After the honest screener rerun, compare newly discovered names against the current approved/pending lists and decide on the next promotion wave.
