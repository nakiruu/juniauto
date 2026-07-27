# Signal separation experiment (audit Fix 1 + Fix 3)

## Two independent bugs from the 2026-07-27 audit

### Finding 1: Composite edge dominates μ in Kelly numerator

The formula in `main.py:302-361` and `backtest/engine.py::_compute_predictions`:

```python
composite = μ + membership_bps * friction     # = μ + 460 * 0.30 = μ + 138 bps
```

Then in sizing:
```python
Candidate(conservative_edge_bps=composite)    # ranking uses composite, not μ
raw_scores = max(0, edge) / σ²                # = (μ + 138) / σ²
```

With μ range ±15 bps, the +138 constant produces essentially-tied Kelly
scores across all symbols. Weights collapse to `~138/σ²` — the model's
actual predictions contribute almost nothing. Hit rate is decent (60.6%
in universe-narrow) but the model can't scale bets by conviction.

### Finding 3: 11 of 26 features are dead

| Family | # | Data source | Reality |
|---|---:|---|---|
| Technical | 7 | Alpaca bars | ✓ |
| Liquidity | 4 | Alpaca bars | ✓ |
| Risk | 4 | Alpaca bars | ✓ |
| Fundamental | 6 | yfinance | 10-15% symbols empty (MUB, XBI, BRK.B...) |
| Event | 3 | yfinance next-earnings | Mostly zero outside earnings |
| Semantic | 2 | **no data source** | Hardcoded 0.5 |

15 useful features, 11 polluting.

## Fixes

**Fix 1**: use `mu_edge_bps` (raw Bayesian μ), NOT `composite_edge_bps`, as
the Kelly ranking numerator and sizing input. The composite is still
computed and still gates the gateway (§2.22a) — but sizing sees the
pure model prediction.

**Fix 3**: `ReducedFeatureBayesian` zeros out the 11 dead feature columns
before both `ridge.update()` and `ridge.predict_mean/variance()`. The
C++ GroupedRidge feature dimension is fixed at 26, so we retain the
shape but effectively drop the columns.

## Baseline

v2_trained (2022-01-01 → 2026-07-24): Sharpe 0.44, CAGR 10.64%,
MaxDD ~15%. Universe-narrow experiment (Rank 2): Sharpe 0.47, CAGR
11.79%, MaxDD -28%.

## Expected effect (from audit)

- Fix 1 alone: Sharpe +0.2-0.4
- Fix 3 alone: Sharpe +0.1-0.2
- Combined (this experiment): Sharpe +0.3-0.6

Target: clear the 0.85 Sharpe promotion bar.

## How to run

```bash
docker exec juniauto-engine python -m experiments.signal_separation.run \\
  --start 2022-01-01 --end 2026-07-24 \\
  --run-id exp_sigsep_v1 --fill next_open --benchmarks spy,ew
```

Wall-time: ~60 min.

## Promotion criteria (unchanged from prior rank criteria)

1. Sharpe ≥ 0.85
2. CAGR ≥ 13%
3. MaxDD ≤ 30%
4. Holds on walk-forward split
5. 60 sessions of paper shadow-monitor > 60% agreement

## Isolation from main

No changes to `main.py`, `production.yaml`, or the live paper-trader.
Sandbox writes to `backtest_*` with `run_id=exp_sigsep_v1`.

## If this passes

The audit findings apply to the LIVE pipeline too. Merging to main will
require:
1. `main.py::_daily_decision_cycle` change to use μ in the sizing
   Candidate construction
2. `training.py::_CANONICAL_FEATURE_COLS` shrunk to 15 (requires
   coordinated feature-schema migration and clearing training data or
   defensive-writing dead columns as 0 in the features table)

Do NOT ship to main until this passes + walk-forward split confirms.
