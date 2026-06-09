# Screener Shortlist — 2026-04-14

Top 20 tickers from mass screener (331 universe), after full Phase 1 validation.
All passed walk-forward stability (4 quarters) and correlation dedupe.

## Auto-added to live watchlist
- **BITF** — crypto miner slot — val 83.0%, long_wr 96.7%, TRAIL_3, TP=25%/SL=2%, skip_low_vix=True

## Additions to consider (user review)

Ranked by Sharpe. All have been validated for walk-forward, exit optimizer, trailing stop, VIX regime, earnings.

### Tier 1 (Sharpe >= 1.0) — strong candidates

| Ticker | Sector | val_acc | long_wr | Strategy | TP | SL | Sharpe | Notes |
|--------|--------|---------|---------|----------|-----|-----|--------|-------|
| ITW | Industrials | 77.5% | 94.1% | FIXED | 10% | 10% | 3.22 | Illinois Tool Works — stable industrial |
| HD | Retail | 75.5% | 78.0% | FIXED | 10% | 10% | 1.47 | Home Depot, needs VIX filter |
| NCLH | Cruise | 83.0% | 87.5% | FIXED | 25% | 10% | 1.28 | Norwegian Cruise |
| HON | Industrials | 74.5% | 76.9% | FIXED | 25% | 10% | 1.22 | Honeywell, needs VIX filter |
| WM | Waste Mgmt | 74.5% | 74.2% | FIXED | 15% | 5% | 1.12 | Waste Management, needs VIX filter |
| PATH | Software | 75.0% | 78.2% | FIXED | 15% | 10% | 1.04 | UiPath |

### Tier 2 (Sharpe 0.5-1.0) — worth watching

| Ticker | Sector | val_acc | long_wr | Strategy | TP | SL | Sharpe | Notes |
|--------|--------|---------|---------|----------|-----|-----|--------|-------|
| ISRG | Med Devices | 87.0% | 77.0% | FIXED | 25% | 10% | 0.81 | Intuitive Surgical |
| NIO | EV | 74.5% | 87.6% | FIXED | 15% | 10% | 0.81 | Nio (EV) |
| GREE | Crypto miner | 75.0% | 80.5% | TRAIL_3 | 20% | 10% | 0.80 | Greenidge, needs VIX |
| OSCR | Healthcare | 74.0% | 67.5% | FIXED | 10% | 10% | 0.65 | Oscar Health |
| TXN | Semis | 74.5% | 81.9% | TRAIL_7 | 25% | 5% | 0.62 | Texas Instruments, needs VIX |
| ANF | Retail | 88.0% | 88.0% | FIXED | 25% | 10% | 0.58 | Abercrombie, needs VIX |
| MGM | Casino | 77.5% | 95.9% | FIXED | 15% | 10% | 0.57 | MGM Resorts, needs VIX |
| NVAX | Biotech | 81.0% | 91.4% | FIXED | 25% | 10% | 0.55 | Novavax, needs VIX |
| INDI | Semis | 75.0% | 93.0% | TRAIL_3 | 25% | 2% | 0.45 | Indie Semi, needs VIX |

### Tier 3 (Sharpe < 0.5) — marginal, skip

| Ticker | Sharpe | Why skip |
|--------|--------|----------|
| MOS | 0.37 | Marginal edge |
| NTLA | 0.31 | Biotech, weak Sharpe |
| OXY | 0.04 | Barely positive |
| TWLO | overflow | Sample issue (long_n=55, no losses) |

## Notes
- All 20 survivors passed correlation check (none correlate >=0.70 with RIVN/ENVX/TSLA)
- No tickers needed earnings filter (rare pattern)
- Most need VIX filter (low VIX kills edge broadly)
- Phase 2 (ensemble mini+small+base) was skipped to save GPU cost — single-model base h=60 results used

## Recommended next moves
1. **Paper trade BITF for 30 days** alongside RIVN/ENVX/TSLA
2. **Consider adding**: ITW (Sharpe 3.22 is unusually strong for industrial name, verify in paper)
3. **Deep dive Tier 1**: before adding any Tier 1 beyond BITF, run ensemble test to verify
