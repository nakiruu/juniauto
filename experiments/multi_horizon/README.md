# Multi-horizon Bayesian ridge experiment

## Hypothesis

Blending predictions from three horizons — 1-day, 5-day, 20-day — with a
weighting that favors the multi-day horizons will materially lift Sharpe
above the single-horizon 1d baseline. Rationale: signal-to-noise ratio
scales roughly as sqrt(H); the 5d target has ~2x SNR of 1d and the 20d
target has ~4x SNR of 1d. The 1d component is retained as a small
overlay so short-horizon signals aren't discarded entirely.

## Baseline to beat

Live production run `v2_trained` (2022-01-01 -> 2026-07-24):

| Metric | v2_trained | This experiment must clear |
|---|---:|---:|
| Sharpe | 0.44 | >= 0.85 |
| CAGR | 10.64% | >= 13% |
| Ann vol | 15.30% | (no explicit target) |
| MaxDD | (measured) | <= 30% |

## Blend weights

  5d = 0.50    (primary signal — best SNR for daily-cadence execution)
  20d = 0.35   (regime signal — smoother, longer-lived edges)
  1d = 0.15    (short-term overlay — noise-diversification)

Total = 1.00. Sum-to-one is asserted at engine boot.

## Signal computation

For each (symbol, decision_date T):
  y_1d  = 10_000 * (close[T+1] - close[T]) / close[T]
  y_5d  = 10_000 * (close[T+5] - close[T]) / close[T]
  y_20d = 10_000 * (close[T+20] - close[T]) / close[T]

Where T+N is the N-th next trading day (not calendar day). Training
requires enough future bars to compute the target, so rows near the
end of history contribute only to shorter horizons.

## Model

Three independent `qe.GroupedRidge` instances, each trained on the
same 26 features and 6 spec-native groups (§1.4) but on a different
horizon's y. All spec constants (kappa=20, lambda=5, zq=1.0, rho=1.0)
preserved unchanged so this remains §2.6-§2.8 compliant per horizon.

Predict returns three (mu, sigma) tuples; blend combines them:
  mu_blend = 0.15*mu_1d + 0.50*mu_5d + 0.35*mu_20d
  sigma_blend = max(sigma_1d, sigma_5d, sigma_20d)     # conservative

## Isolation from main

- New code lives entirely in `experiments/multi_horizon/`.
- Backtest writes to the existing `backtest_*` tables with
  `run_id = 'exp_mh_v1_...'` so Grafana overlays work without any
  dashboard changes.
- No modifications to `config/production.yaml`, `orchestrator/juniauto/main.py`,
  or any live-side code. The live paper-trader continues on the
  single-horizon ridge.

## How to run

```bash
docker exec juniauto-engine python -m experiments.multi_horizon.run \
  --start 2022-01-01 --end 2026-07-24 \
  --run-id exp_mh_v1 --fill next_open --benchmarks spy,ew
```

Wall-time: ~60-90 min (3x model training + same backtest cycle count as
the baseline).

## Promotion criteria (from coordinator research 2026-07-26)

All five must clear before considering merge to main:
1. Sharpe >= 0.85 on the full 4.6y window
2. CAGR >= 13% net of §2.9-§2.24 cost model
3. MaxDD <= 30%
4. Improvement holds on walk-forward split (train 2022-01 -> 2024-12,
   test 2025-01 -> 2026-07)
5. 60 sessions of live paper-trading shadow-monitor agreement > 60%
   with backtest predictions

## Rollback trigger

If promoted to main and then live Sharpe drops > 0.3 below backtest
Sharpe for any 90-day rolling window, revert to `v2_trained` config.
