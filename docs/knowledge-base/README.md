# JuniAuto Knowledge Base

Persistent, section-anchored digest of `PRINCIPLESLONG.md` (~1980 lines) for future coding sessions. All formulas cite spec section numbers (e.g., `(§2.24)`) so implementation and review can back-reference exact text.

## Core operational context (read this first)

The spec is `PRINCIPLES.md` adapted for two operational constraints. Every parametric change from the theoretical baseline is downstream of these two facts:

1. **Alpaca IEX free stream = 15-minute delayed data.** No real-time quotes. Sub-day signals are unobservable in time to act on.
2. **Account < $20,000 → US Pattern Day Trader rule.** Max 3 day trades per rolling 5-trading-day window. Enforced as a **1-trading-day minimum holding period** (no same-day open-and-close for any security).

Whenever a formula reads "recalibrated to trading sessions" or "IEX delay note:", it is one of the load-bearing operational deltas. Files below flag each of these with an `> **IEX/PDT adaptation:**` callout.

## File index

| File | Spec range | One-line summary |
|---|---|---|
| `part1-data-preparation.md` | §1.1–§1.8 | Universe, causal standardization, winsorization, MAD, freshness halflives (recalibrated to trading days), six signal families, spike-and-slab group prior |
| `part2-signals-bayesian.md` | §2.1–§2.8 | Log-wealth objective, mu_i_H decomposition, no sub-day horizons, zq=1.0 posterior edge, grouped ridge posterior, rho/rho_h contribution accounting |
| `part3-costs.md` | §2.9–§2.18 | TotalCost decomposition, spread/slippage/impact (SQRT_IMPACT_COEFF=25), Roll bounce correction, partial/missed-fill EV, RoundTripEV, dynamic hurdle |
| `part4-gateway-execution.md` | §2.19–§2.30 | Applied Model Parameters, source package selection (95 bps, decay=1.0), gateway (primary=460, secondary=348, retained=200), dynamic friction 0.30, cost components, gross action edge by type, position sizing caps |
| `part5-shadow-and-replay.md` | §2.31–§2.42 | Drift, action decision equations, shadow promotion monitor (kappa_0=7, min_clean=30, positive_share=0.55, k=2), n_eff for overlapping labels, dynamic action-surface challenger (ridge lambda=5, prior_strength=20) |
| `part6-operational.md` | §3.1–§3.14 | PDT constraints, 7-step decision cycle, order type routing, session multipliers (1.0/1.5/2.0/2.5), self-test certificate, purged CV, PBO, DSR, IC / SRPC / alpha decay metrics |
| `glossary.md` | all | Every symbol and constant with one-line meaning and section pointer |

## Constants at a glance

Numbers are exact — do not round.

| Constant | Value | Source | Notes |
|---|---|---|---|
| `target_cadence` | end-of-day (daily) | §2.19 | primary execution window = regular session close |
| `minimum_holding_period` | 1 trading day | §2.19 | PDT constraint; no same-day round trip |
| `max_day_trades_rolling_5_day_window` | 3 | §2.19, §3.1 | PDT hard constraint |
| `effective_execution_horizon` | 1 trading day = 390 minutes | §2.19, §2.27 | derived from PDT + IEX; replaces 15-minute derivation |
| `minimum_hurdle_bps` | 0 | §2.19, §2.26 | Singularity.pdf p.78 default |
| `retained_baseline_floor` | 200 bps | §2.19, §2.22 | full-window peak |
| `primary_role_signal` | 460 bps | §2.19, §2.22 | chosen from sweep, lower of tie with 461 |
| `secondary_role_signal` | 348 bps | §2.19, §2.22 | center of 345–350 plateau |
| `dynamic_friction_multiplier` seed | 0.30 (all roles) | §2.19, §2.23 | prior mean of adaptive surface |
| `exit_reserve` | 1.00 | §2.19, §2.24 | full modeled future exit cost reserved |
| `rotation_funded_sells` | true | §2.19 | rotations treat sell proceeds as buy funding |
| `adaptive_action_memory_enforcement` | shadow_only | §2.19 | reversal charge tracked, not enforced |
| `automatic_surface_switching` | disabled | §2.19, §2.41 | challengers only marked promotion-ready |
| `source_selection_evidence_threshold` | 95 bps | §2.20 | switch surface only if G_p >= 95 bps |
| `source_selection_decay` | 1.0 | §2.20 | no exponential decay in active config |
| `zq` (posterior caution) | 1.0 | §2.6 | ~84% credible bound above zero |
| `rho` (group utility penalty) | 1.0 | §2.7 | m_k - rho * sqrt(V_k,k) |
| `rho_h` (contribution penalty) | 1.0 flat | §2.8 | horizon-differentiated ladder deferred |
| `SQRT_IMPACT_COEFF` | 25 | §2.12 | Bouchaud; 9 rejected |
| `alpha_fill` (slippage decay) | 0.75 per fill, K_max=10 | §2.24 | clamp [0, 50]; cold-start alpha=0.90 over 20 fills |
| `fallback_round_trip_cost_bps` | 60 | §2.24 | until first ticker fill |
| `api_budget_cost_bps` | 2 | §2.24 | Alpaca REST cancel-failure proxy |
| `lost_queue_priority_bps` | 1 | §2.24 | Alpaca PFOF marginal reprice |
| `adverse_selection_share` (regular/ext/closed) | 0.35 / 0.55 / 0.90 | §2.24 | Barclay-Hendershott |
| BUY exit haircut | 0.65 * 0.85 = 0.5525 | §2.24 | on top of future-discount factor |
| `action_memory_horizon_seconds` | 23,400 (= 390 * 60) | §2.24 | one trading day of market seconds |
| `session_multiplier` (reg/pre/after/closed) | 1.0 / 1.5 / 2.0 / 2.5 | §3.4 | Chordia-Roll-Subrahmanyam calibration |
| `operational_risk_bps` cold-start | 10 | §2.26 | tiered production formula, cap 40 |
| `cash_waiting_value_bps` | (r_cash / 252) * 10000 | §2.24 | r_cash = broker sweep or Fed Funds - 50 bps |
| `maxNameWeight` | 0.10 | §2.29 | 10% NAV per name hard cap |
| `cash_floor` | 0.05 | §2.29 | raise to 0.10 when posterior delta < 0 |
| `aggregateComfortableWeight` | 0.20 | §2.30 | cross-horizon soft cap per name |
| Shadow `kappa_0` (prior strength) | 7 | §2.41 | normal-normal conjugate |
| Shadow `delta_0` (prior mean) | 0 | §2.41 | |
| Shadow `min_clean_resolved_rows` | 30 (1d only) | §2.41 | raise per horizon for overlap |
| Shadow `min_positive_share` | 0.55 | §2.41 | |
| Shadow peeking k | 2 consecutive cycles | §2.41 | anti-multiplicity correction |
| Shadow n_eff mins (1d/2-3d/1wk/2wk) | 30 / 45 / 60 / 80 | §2.41 | after label-autocorr shrink |
| rho_label (1d/2-3d/1wk/2wk) | 0.00 / 0.30 / 0.50 / 0.70 | §2.41 | overlapping-window autocorrelation |
| Ridge `RIDGE_LAMBDA` | 5 | §2.42 | challenger L2 |
| Ridge `PRIOR_STRENGTH_KAPPA` | 20 | §2.42 | equivalent observations for bucket prior |
| Ridge decay (challenger) | 0.75 → n_eff_decay ≈ 4 | §2.41 | 0.90 → 10; 0.95 → 20 |
| Purged-CV embargo | 11 trading days | §3.7 | h_max + 1 (h_max = 10) |
| PBO threshold | 0.50 | §3.7 | do not promote if PBO > 0.50 |
| DSR threshold | 1.0 | §2.41, §3.7 | below → improvement is plausibly noise |
| IC alert thresholds | healthy > 0.04, degraded < 0.02 for 3 windows, flipped < -0.03 | §3.6 | per family, per horizon |
| SRPC alert | < 0.80 | §3.6 | execution reordering positions |
| MAD scale factor | 1.4826 | §1.2 | consistent SD estimator for Gaussian |
| Winsorization | 1st/99th (ranks), ±5 MAD (fill/label) | §1.2 | |

## Cross-cutting invariants

- Coherent one-sigma penalty across three layers: `zq = rho = rho_h = 1.0` (§2.6, §2.7, §2.8). Any bump/cut here must be reasoned about jointly.
- `model_edge_bps` (§2.22a) is the composite that enters the gate: `after_cost_edge + source_member_edge * dynamic_friction_multiplier`. The provenance path is always non-negative; the Bayesian path can be negative.
- `sigma^2` is treated as a plug-in constant in §2.6–§2.8 and §2.41; if the calibration set is small (< 500 rows/group) the Gaussian approximation for theta becomes Student-t and zq bounds are only approximate.
