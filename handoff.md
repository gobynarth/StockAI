# Handoff

## Goals
- Keep the research flow honest with walk-forward validation.
- Keep live conservative and only promote names that survive honest retesting and exit-design checks.
- Let the native Windows automation prove itself before the second promotion wave.

## Current State
- Honest research pipeline was added locally:
  - `research_pipeline.py`
  - `honest_walk_forward.py`
  - tests for splits, selection, execution simulation, CLI, checkpoint loading, and live candidate loading
- Verified locally:
  - `tests/test_live_candidates.py` passes
- Live candidate loading is file-backed via:
  - `approved_live_candidates.csv`
  - `approved_pending_candidates.csv`
  - `live_candidates.py`
- `live_signal.py` now:
  - loads the active live set from `approved_live_candidates.csv`
  - uses `approved_pending_candidates.csv` as the rerun/promotion watchlist source
  - prints the active summary dynamically instead of the old stale hardcoded line
- Stale local paper trades were cleared; `paper_trades.csv` now only contains the old closed `ENVX` row until new runs add open trades again.

## Live Set
- Approved live set:
  - `RIVN`
  - `BITF`
  - `PATH`
  - `HON`
  - `ITW`
  - `NCLH`
  - `NVAX`
  - `WK`
  - `AAL`
- Newly promoted into live from the honest screener wave:
  - `NVAX` with conservative fixed exits `tp=0.30`, `sl=0.10`, `alloc=0.02`
  - `WK` with conservative fixed exits `tp=0.25`, `sl=0.10`, `alloc=0.02`
  - `AAL` with conservative fixed exits `tp=0.30`, `sl=0.10`, `alloc=0.02`

## Pending Queue
- Next promotion wave, ranked:
  - `HYLN`
  - `RRC`
  - `TXN`
  - `NIO`
  - `MGM`
  - `SBUX`

## Research Status
- Full honest screener rerun was completed on the temporary 4090 pod.
- Results were copied back locally before the pod was shut down.
- The pod is no longer needed for the current phase.
- Honest promotion passes were run locally for the strongest new names.
- Best names from the latest promotion wave were:
  - `NVAX`
  - `WK`
  - `AAL`
  - `HYLN`
  - `RRC`
  - `TXN`
  - `NIO`

## Windows Automation Status
- Windows scheduled task `\StockAI_Daily_Signal` exists and points at `run_daily_signal.bat`.
- Windows-side work outside this WSL session added IBKR startup logic to the native launcher path.
- Important nuance:
  - hot-start runs from this WSL-driven path worked when IBKR was already up
  - cold-start verification from WSL was not authoritative because WSL-to-Windows launch behavior is different from the native Windows trigger path
- Native Windows runs were verified by the user outside this session and should be treated as the source of truth for Monday.

## Commit
- Main refactor/live-shortlist commit:
  - `44c92f6` `Refactor live shortlist and honest validation flow`

## Cleanup Done
- Removed obvious stale local artifacts:
  - old logs
  - stale screener result exports
  - old `screener_survivors.csv`
  - local Codex metadata/plans
  - duplicated nested `new_ticker_checkpoints/new_ticker_checkpoints`
- Left older research scripts and main checkpoint/data folders intact.

## Next Steps
1. Do not change the live set again before Monday.
2. Let the native Windows scheduled run be the proof run.
3. After Monday, if automation/broker behavior is clean, promote the second wave:
   - `HYLN`
   - `RRC`
   - `TXN`
   - `NIO`
