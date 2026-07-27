# Universe narrowing experiment (Rank 2)

## Hypothesis

Replace the 299-name mega-cap-heavy universe with a quarterly-refreshed
40-60 name selection ranked by three bars-only signals:
  - 60-day average dollar volume (liquidity)
  - 12-month minus 1-month price momentum
  - Inverse of 60-day realized volatility (quality proxy)

Ridge posterior variance shrinks materially when N/features improves.
Top-K selector concentrates on names where the model has discriminative
power instead of diluting across 300 mega-caps where signal is nil.

## Data policy: $0/month

Uses ONLY the `bars` table already populated by backfill. No Yahoo, no
Sharadar, no new subscriptions. Rationale in coordinator research
2026-07-26.

## Baseline to beat

v2_trained (2022-01-01 → 2026-07-24): Sharpe 0.44, CAGR 10.64%, MaxDD ~15%.

## Prior experiment

exp_mh_v1 (multi-horizon, Rank 1) FAILED: Sharpe 0.24, CAGR 5.3%,
MaxDD -47%. Root cause: daily-cadence + multi-day signal mismatch caused
turnover explosion. This experiment is INDEPENDENT of that failure —
does not use multi-horizon signal at all.

## Selector logic

For each quarter-start date T in [start, end]:
  1. For each symbol in the full universe (299 names):
     - `liq_score`  = mean(close[T-60..T] × volume[T-60..T])
     - `mom_score`  = (close[T-21] / close[T-252]) - 1
     - `vol_score`  = 1 / std(returns[T-60..T])
     - `combined` = z(liq_score) + z(mom_score) + z(vol_score)
  2. Take top N symbols by combined score.

Quarter is a fixed calendar quarter (Q1=Jan-Mar, Q2=Apr-Jun, ...).
Rebalance happens on the first trading day of each new quarter.

## Engine wiring

`UniverseNarrowBacktestEngine` subclasses `BacktestEngine` and overrides
only `_compute_predictions(features)`. Before delegating to parent:
  1. Look up active symbol set for current cycle date
  2. Filter features.index to that set
  3. Delegate

Everything downstream (top-K, sizing, routing, persistence) unchanged.

## How to run

```bash
docker exec juniauto-engine python -m experiments.universe_narrow.run \\
  --start 2022-01-01 --end 2026-07-24 \\
  --run-id exp_un_v1_topN50 --top-n 50 \\
  --fill next_open --benchmarks spy,ew
```

Wall-time: ~60 min (same cycle count as baseline; universe selection is
one-time at engine init).

## Promotion criteria (all five must clear)

1. Sharpe ≥ 0.85 on the full 4.6y window
2. CAGR ≥ 13% net of the cost model
3. MaxDD ≤ 30%
4. Improvement holds on walk-forward split
5. 60 sessions of live paper shadow-monitor > 60% agreement

## Circuit-breaker note

Per-name concentration with K=8 in a 50-name universe = 12.5% per position.
Coordinator flagged this: "single-name blowup risk (NKLA/WISH/SIVB event)
costs 8-12% of NAV instantly." If we promote to main, must couple with
per-name hard cap 15% of NAV + news-gap kill switch.
