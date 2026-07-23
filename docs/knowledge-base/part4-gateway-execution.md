# Part 4 — Gateway and Execution (§2.19–§2.30)

## 2.19 Applied model parameters (§2.19)

Replay-calibrated values, not theoretical constants. Each governs a decision boundary.

```
target_cadence                          = end-of-day (daily)
source_selection_mode                   = prediction_sign_regime_evidence
retained_baseline_floor                 = 200 bps
primary_role_signal                     = 460 bps
secondary_role_signal                   = 348 bps
dynamic_friction_multiplier_primary     = 0.30
dynamic_friction_multiplier_secondary   = 0.30
exit_reserve                            = 1.00
effective_execution_horizon             = 1 trading day (390 minutes)
rotation_funded_sells                   = true
adaptive_action_memory_enforcement      = shadow_only
automatic_surface_switching             = disabled
minimum_holding_period                  = 1 trading day  (PDT constraint)
max_day_trades_rolling_5_day_window     = 3               (PDT hard constraint)
minimum_hurdle_bps                      = 0               (Singularity.pdf p.78)
```

> **IEX/PDT adaptation:** `effective_execution_horizon = 1 trading day = 1.0 * 390 min = 390 minutes`. Replaces prior derivation `0.25 hours * 60 min/hr = 15 minutes`. Both PDT (account < $20K, ≤ 3 day trades / 5-day window) and IEX (15-min delayed feed) independently force this floor.

Rationale:
- `exit_reserve = 1.00`: full modeled future exit cost reserved on every buy entry.
- `rotation_funded_sells = true`: prevents rejecting rotations because cash is low before the sell leg fills.
- `automatic_surface_switching = disabled`: challengers only marked promotion-ready, never silently deployed.

## 2.20 Source package selection (§2.20)

The selector picks which learned source package holds authority.

Evidence score update:

```
G_p,t                = decay * G_p,t-1 + realized_evidence_p,t
realized_evidence_p,t = sign(prediction_p,t) * realized_after_cost_return_p,t,h
```

Active package selection:

```
p*_t = argmax_p G_p,t
Use p*_t if G_p*_t,t >= 95 bps
```

Applied: `evidence_threshold = 95 bps`, `decay = 1.0` (no exponential fading). The selector asks which learned decision surface deserves authority given recent sign+magnitude evidence — not merely which ticker looks best.

## 2.21 Target portfolio gateway (§2.21)

Converts the selected source package's candidates into a target portfolio. No fixed position cap or holding count.

**Active tradable set** (fresh price and positive signal required):

```
A_t = { i in C_t : price_i,t is fresh and positive }
```

**Source-conviction boost:**

```
b_i,t = max(p_i,t - p_min, 1)^gamma   if p_i,t > p_min
      = 1                             otherwise
```

Applied: `gamma = 1`, `p_min = 0` (linear weighting; `gamma = 3` was rejected for overconcentration).

**Gross target exposure** (inherited from selected source):

```
G_t = min(1, sum_j(s_j,t) for j in A_t)
```

**Source-conviction target weight:**

```
w_i,t = G_t * (s_i,t * b_i,t) / sum_j(s_j,t * b_j,t)
```

A single ticker can receive nearly all of `G_t` when it dominates positive source-conviction score. Cash is the residual when source doesn't allocate the full book or execution state blocks movement to target. Cash is not a failure state.

## 2.22 Provenance membership action signal (§2.22)

Execution layer assigns a prior action edge by source role:

```
source_member_edge_bps(i, t) =
    460   if M_i(t) = primary
    348   if M_i(t) = secondary
    200   if M_i(t) = retained_baseline
    0     otherwise
```

These are Bayesian priors, not standalone forecasts. The gate discounts them through dynamic friction and compares against real costs.

**Calibration sweeps (annualized after-cost return from replay):**

Primary sweep (secondary at floor):
```
455 -> 1037.064021%
458 -> 1038.794167%
459 -> 1038.825237%
460 -> 1038.995427%   <- chosen (lower of tie with 461)
461 -> 1038.995427%
462 -> 1038.808247%
465 -> 1038.808247%
470 -> 1012.943303%
```

Secondary sweep (primary at 460):
```
320 -> 1039.706152%
340 -> 1040.615335%
345 -> 1043.294209%
348 -> 1043.294209%   <- chosen (center of plateau)
350 -> 1043.294209%
352 -> 1042.806548%
355 -> 1042.774181%
360 -> 1042.219407%
460 -> 1038.995427%
```

Retained baseline: strongest full-window point = 200 bps.

## 2.22a Composite model edge (§2.22a)

The two edge paths (Bayesian §2.7–§2.8 and provenance §2.22) combine into a single scalar entering the gate:

```
model_edge_bps(i, t, h) = after_cost_edge_i,t,h
                        + source_member_edge_bps(i, t) * dynamic_friction_multiplier(t)
```

- `after_cost_edge_i,t,h` — posterior expected edge net of position risk and feature uncertainty. Primary path, can be positive or negative.
- `source_member_edge_bps(i,t) * dynamic_friction_multiplier(t)` — provenance prior, always non-negative. Structural quality floor.

Interpretation: a name with strong posterior evidence and validated membership gets an additive boost; a name with weak posterior evidence can still pass if provenance edge is large enough. This is the intended behavior — membership provides a structural floor, the posterior modulates around it.

The gate then computes:

```
gross_action_value_bps = model_edge_bps(i, t, h) - total_cost_bps(i, t, h)
Action triggers when: gross_action_value_bps > minimum_hurdle_bps  (= 0)
```

`model_edge_bps` also feeds `raw_weight_i` (§2.29) via `positive_net_edge_i = max(0, model_edge_bps)`.

## 2.23 Dynamic friction gate (§2.23)

Discounts the source membership edge to convert target recommendation into trade / no-trade.

```
effective_model_edge_bps(i, t) = source_member_edge_bps(i, t) * dynamic_friction_multiplier(i, t)
```

Multiplier depends on: `role_i`, `regime_t`, `ticker_i`, `liquidity_t`, `source_package_t`, `execution_quality_t`.

**Baseline seeds** (prior mean of adaptive surface):

```
primary_baseline_multiplier   = 0.30
secondary_baseline_multiplier = 0.30
retained_baseline_multiplier  = 0.30
```

Effective edges at seed:

```
primary:   460 * 0.30 = 138.0 bps
secondary: 348 * 0.30 = 104.4 bps
retained:  200 * 0.30 =  60.0 bps
```

The 0.30 seed is an empirical prior for the fraction of nominal source edge that counts against friction; context evidence moves it up or down.

**Adaptive update (logit-Bayesian form):**

```
logit(m_r,k,p,t) = eta_0 + eta_role[r] + eta_ticker[k] + eta_package[p]
                 + eta_liquidity[k,t] + eta_regime[r,t] + eta_execution_quality[k,t]
eta_0 = logit(0.30)
eta_role[primary] = 0
eta_role[secondary] = 0

effective_multiplier_i,t = clamp(baseline_multiplier(role_i) + shrink(realized - predicted),
                                 multiplier_floor, multiplier_ceiling)
```

`multiplier_floor` and `multiplier_ceiling` are (unspecified in spec — TODO). Single fills do not move the multiplier; updates require repeated resolved actions showing the baseline is systematically wrong.

## 2.24 Gateway cost components (§2.24)

All costs in bps of action notional.

### Normalization

```
notional     = abs(target_delta_dollars)
spread_bps   = min(1000, max(0, spread_pct) * 100)
volatility_bps = max(0, volatility_pct) * 100
size_ratio   = notional / bar_dollar_volume   if bar_dollar_volume > 0
             = 1                              if bar_dollar_volume missing and notional > 0
             = 0                              otherwise
```

### Liquidity capacity risk

```
liquidity_capacity_risk_bps = min(120, 0.25 + 25 * sqrt(min(9, size_ratio)))

If volume missing and notional > 0:
    liquidity_capacity_risk_bps = max(liquidity_capacity_risk_bps, 35)
```

Coefficient `25` matches `SQRT_IMPACT_COEFF` from §2.12 (`N/ADV` fraction). Both express the same Bouchaud square-root law; different denominators (bar volume vs full ADV) yield the same coefficient because `bar_dollar_volume ≈ ADV / 6.5` for a 1-hour bar, and `sqrt(6.5) ≈ 2.5` absorbs into the coefficient. Agree within ±15% for `size_ratio ∈ [0.001, 0.01]`.

### Stale quote risk

> **IEX/PDT adaptation:** Original second-level thresholds (8s, 60s) no longer meaningfully distinguish fresh from stale under a 15-minute mandatory delay. Recalibrated to sessions:

```
if quote_age_sessions > 1.0:   stale_quote_risk_bps = min(80, 4 + (quote_age_sessions - 1.0) * 40)
elif quote_age_sessions > 0.5:  stale_quote_risk_bps = min(20, (quote_age_sessions - 0.5) * 40)
else:                          stale_quote_risk_bps = 0

quote_age_sessions = age_in_minutes / 390
```

IEX 15-min delay → `15 / 390 = 0.038 sessions`, inside the zero-cost band (< 0.5). **stale_quote_risk_bps = 0 for all quotes from the current session by design.** Staleness charges arise only when a quote is > 195 minutes stale (data outage / connectivity gap), not from the normal IEX delay. The IEX delay cost is captured implicitly in (a) `recent_fill_slippage_bps` feedback and (b) `adverse_selection_bps` in exit cost.

### Gap risk

```
gap_risk_bps = clamp((gap_days_to_next_trading_session - 1) * 1.75, 0, 25)
```

### Base one-side cost

```
base_side_cost_bps = (0.5 * spread_bps
                   + 0.10 * min(200, volatility_bps)
                   + liquidity_capacity_risk_bps
                   + stale_quote_risk_bps
                   + recent_fill_slippage_bps)
                  * session_multiplier
                  + gap_risk_bps
```

Terms:
- `0.5 * spread_bps`: expected half-spread
- `0.10 * min(200, volatility_bps)`: near-term volatility markout allowance. `volatility_bps` is annualized realized vol in bps (= `volatility_pct * 10000 * sqrt(252)`). The `min(200, ...)` cap binds for nearly all liquid equities → effective vol-markout contribution ≈ `0.10 * 200 = 20 bps/side`.
- `liquidity_capacity_risk_bps`: capacity + partial-fill risk from size
- `stale_quote_risk_bps`: price-observation reliability (separate from volatility)
- `recent_fill_slippage_bps`: realized-fill feedback

**`recent_fill_slippage_bps` aggregation** (exponential decay over fills, ticker-specific):

```
recent_fill_slippage_bps(i, t) =
    sum_{k=1..K} (alpha_fill^(k-1) * slippage_bps_k(i))
    / sum_{k=1..K} (alpha_fill^(k-1))

alpha_fill = 0.75    (per-fill decay)
K          = min(n_fills_available(i), 10)
Indexing:  k=1 is most recent fill

slippage_bps_k(i) = 10000 * (fill_price_k - mid_at_decision_k) / mid_at_decision_k    (BUY)
                  = 10000 * (mid_at_decision_k - fill_price_k) / mid_at_decision_k    (SELL)

Cold start (< 3 ticker-specific fills):
    universe-level fallback over last 20 fills with alpha_fill = 0.90

Floor/cap: clamp(computed_value, 0, 50)
    floor 0: lucky fills do not reduce structural cost floor
    cap 50:  one anomalous fill does not permanently block future trades
```

### Entry cost by action type

```
entry_cost_bps = base_side_cost_bps    for BUY, ROTATE, REPLACE
               = 0                      otherwise
```

### Exit cost by action type

```
exit_cost_raw_bps = base_side_cost_bps         if action in {SELL, ROTATE, REPLACE}
                  = 0                           if action = CANCEL
                  = 0.85 * base_side_cost_bps   if action = BUY   (reserved future exit)
```

**Combined BUY factor:** `exit_cost_modeled_bps = 0.65 * exit_cost_raw_bps = 0.65 * 0.85 * base_side_cost_bps = 0.5525 * base_side_cost_bps`. The 0.65 haircut applies **on top of** the 0.85 future-discount factor.

### Adverse selection component (Glosten-Milgrom 1985)

The `0.65` haircut is a practitioner scalar bundling half-spread, half-impact, and adverse selection. Explicit decomposition (target model; use once realized-fill adverse-selection share estimates exist):

```
exit_cost_explicit_bps = (0.5 * spread_bps
                        + adverse_selection_bps
                        + 0.5 * liquidity_capacity_risk_bps) * session_multiplier_exit

adverse_selection_bps = 0.5 * spread_bps * adverse_selection_share(session)

adverse_selection_share:
    regular session:    0.35   (Barclay & Hendershott 2003)
    extended hours:     0.55   (informed traders dominate)
    closed / overnight: 0.90   (Barclay & Hendershott 2003)
```

Reserved future exit cost:

```
reserved_future_exit_cost_bps = exit_cost_modeled_bps * clamp(exit_reserve_fraction, 0, 1)
With exit_reserve = 1.00, the gate reserves the full modeled future exit cost.
```

### Queue delay risk (BUY, SELL, ROTATE, REPLACE only)

```
queue_delay_risk_bps = min(60, 0.8 + 12 * size_ratio + 4.5 * max(0, session_multiplier - 1))
```

### Cancel/replace mutation risk (REPLACE, CANCEL, or when open order exists)

```
cancel_replace_risk_bps = min(60, api_budget_cost_bps + lost_queue_priority_bps + max(0, session_multiplier - 1))

api_budget_cost_bps     = 2   (Alpaca REST; no per-request fee; captures ~0.5% cancel-failure prob + latency)
lost_queue_priority_bps = 1   (Alpaca PFOF: no exchange queue; marginal MM reprice on replace)

Evaluated:
    Regular   (mult 1.0):  3 bps
    Premarket (mult 1.5):  3.5 bps
    After-hrs (mult 2.0):  4 bps
    Closed    (mult 2.5):  4.5 bps
```

### Same-ticker reversal (action memory) charge

> **IEX/PDT adaptation:** Action memory horizon is one **trading day** of market seconds, not a 15-minute window.

```
action_memory_horizon_seconds = predicted_holding_seconds    if predicted_holding_seconds > 0
                              = edge_horizon_minutes * 60    otherwise
                                (market seconds; 390 * 60 = 23,400 s per trading day)

action_memory_cost_bps = max(recent_opposite_fill_cost_bps, fallback_round_trip_cost_bps)
                       * exp(-recent_opposite_fill_age_seconds / action_memory_horizon_seconds)

fallback_round_trip_cost_bps = 60
```

`fallback_round_trip_cost_bps = 60` derivation for the target universe (price > $5, ADV > 500K, mktcap > $500M): half-spread round-trip ~8 + vol markout ~30 + capacity ~8 + stale-quote floor ~10 + impact ~2 ≈ 58, rounded to 60. Replaced by `recent_opposite_fill_cost_bps` once first fill occurs.

Applied config: `adaptive_action_memory_enforcement = shadow_only` (§2.19) — cost tracked but not enforced against orders yet.

### Cash waiting value (opportunity value of waiting; BUY, ROTATE)

```
C_wait(a, t) = cash_waiting_value_bps(t)   for BUY, ROTATE

cash_waiting_value_bps(t) = (r_cash(t) / 252) * 10000

r_cash(t) = broker_sweep_apy(t)         if available from broker API
          = fed_funds_rate(t) - 0.005   otherwise (50 bps below Fed Funds as retail sweep haircut)
```

Example: at `r_cash = 0.045` (4.5% APY) → `cash_waiting_value_bps = 1.79 bps per trading day`. Update `r_cash(t)` daily. Never negative. Applies symmetrically: BUY / ROTATE incur `C_wait` as cost; SELL uses `cash_waiting_value_bps` as the threshold value of returning to cash.

## 2.25 Gross action edge by type (§2.25)

```
BUY:
    gross_action_edge_bps = model_edge_bps - cash_waiting_value_bps
    action_cost_bps       = entry_cost_bps + reserved_future_exit_cost_bps
                          + queue_delay_risk_bps + action_memory_cost_bps

ROTATE:
    gross_action_edge_bps = model_edge(new_target) - model_edge(old_position) - cash_waiting_value_bps
    action_cost_bps       = entry_cost_bps + exit_cost_modeled_bps
                          + queue_delay_risk_bps + action_memory_cost_bps

SELL:
    gross_action_edge_bps = cash_waiting_value_bps - model_edge_bps
    action_cost_bps       = exit_cost_modeled_bps + queue_delay_risk_bps + action_memory_cost_bps

REPLACE:
    gross_action_edge_bps = replacement_improvement_bps
    action_cost_bps       = cancel_replace_risk_bps + queue_delay_risk_bps + action_memory_cost_bps

CANCEL:
    gross_action_edge_bps = replacement_improvement_bps
    action_cost_bps       = cancel_replace_risk_bps
```

**`replacement_improvement_bps` computation:**

```
replacement_improvement_bps(t) = EV(proposed_order_state, t) - EV(current_order_state, t)

EV(limit at price L, t) = P_fill(L) * (alpha - C_fill(L))
                        + (1 - P_fill(L)) * cash_waiting_value_bps(t)

P_fill(L) = clamp((L - bid_bps) / spread_bps, 0, 1)      (linear: 0 at/below bid, 1 at/above ask)
C_fill(L) = 0.5 * spread_bps - (L - mid_bps) / mid_bps * 10000
alpha     = effective_model_edge_bps(i, t)

REPLACE (L_old -> L_new):
    replacement_improvement_bps =
        [P_fill(L_new) * (alpha - C_fill(L_new)) + (1 - P_fill(L_new)) * cash_waiting_value_bps]
      - [P_fill(L_old) * (alpha - C_fill(L_old)) + (1 - P_fill(L_old)) * cash_waiting_value_bps]

CANCEL (proposed = cash):
    replacement_improvement_bps =
        cash_waiting_value_bps(t)
      - [P_fill(L_old) * (alpha - C_fill(L_old)) + (1 - P_fill(L_old)) * cash_waiting_value_bps]
```

If proposed limit equals existing limit, `replacement_improvement_bps = 0` and `cancel_replace_risk_bps` correctly blocks the redundant action.

## 2.26 Final gateway decision (§2.26)

```
minimum_hurdle_bps        = 0    (default; override only via governance)
minimum_required_edge_bps = max(minimum_hurdle_bps, action_cost_bps + operational_risk_bps)
effective_net_edge_bps    = gross_action_edge_bps - minimum_required_edge_bps

Execute iff:
    effective_net_edge_bps > 0
    AND account_constraints pass
    AND market_state allows
    AND no duplicate conflicting order
```

**`operational_risk_bps`:**

```
Cold-start default: operational_risk_bps = 10

Production tiered:
    operational_risk_bps = clamp(
        5
        + 10 * I(api_error_rate_24h > 0.01)
        + 15 * I(no_current_session_quote_confirmed)
        + 20 * I(broker_position != system_position),
        0, 40
    )
```

`5 bps` base (latency, order-state lag, rounding); `+10` if API error rate > 1% in last 24h; `+15` if no current-session quote confirmed for this ticker; `+20` if broker position ≠ system-tracked position. Cap 40 — if risk exceeds cap, halt trading rather than buffer.

## 2.27 Why the effective horizon is 1 trading day (§2.27)

> **IEX/PDT adaptation (primary derivation):**
> 1. PDT: account < $20K → ≤ 3 day trades per rolling 5-trading-day window.
> 2. A day trade = open+close same security same calendar day.
> 3. Minimum holding period that produces **0 day trades** = **1 full trading day**. Zero day trades satisfies ≤ 3 with maximum safety margin.
> 4. 1 trading day = 390 minutes regular session.
> 5. IEX 15-minute delay independently makes sub-day execution unreliable.
> 6. `effective_execution_horizon = 1.0 trading day * 390 min/trading day = 390 minutes`.
> 7. Governs action memory: `edge_horizon_minutes = 390`, `action_memory_horizon_seconds = 390 * 60 = 23,400`.

This is an after-cost evaluation window, not a forced delay. The model can act at the end of each trading day if the order clears the gate. The horizon sets the time scale over which the edge must remain economically meaningful.

## 2.28 Portfolio construction objective (§2.28)

```
F(w) = (w' * mu)
     - (0.5 * gamma_risk(t) * w' * Sigma_risk(t) * w)
     + (lambda_confirm(t) * w' * Sigma_confirm(t) * w)
     - c(w_current, w) - u(w) - l(w) - k(w) - g(w)

Subject to:
    sum_i(w_i) + cash_weight = 1
    0 <= w_i <= max_weight_i
    cash_weight >= cash_min
    liquidity_capacity_i >= planned_notional_i
    security_i is tradable
```

Covariance split into two objects:
- `Sigma_risk` (penalized via `gamma_risk >= 0`): joint loss, drawdown, vol drag, liquidity stress, crowding
- `Sigma_confirm` (rewarded via `lambda_confirm >= 0`): correlated positive continuation, sector/theme leadership, breadth-confirmed momentum

In stress regime, `lambda_confirm → 0`, `gamma_risk` rises.

## 2.29 Position sizing (§2.29)

**Kelly reference** (single security, unconstrained):

```
w_kelly = mu / sigma^2
```

Holds when `gamma_risk = 1` (pure log-utility / Kelly case), the operative value here (consistent with §2.1 log objective). The heuristic `raw_weight_i` is a practical approximation of `w_kelly` with `mu ← positive_net_edge`, `sigma^2 ← regime-conditional volatility scaled by confidence`.

**Practical modified sizing:**

```
raw_weight_i = positive_net_edge_i^p * confidence_multiplier_i
             * liquidity_multiplier_i * diversification_multiplier_i
```

`p > 1` creates top-heavy allocation toward strongest opportunities. Exact `p` is (unspecified in spec — TODO; typical 2–3).

**Final weight with soft cap:**

```
w_i = min(max_position_i, raw_weight_i / sum_j(raw_weight_j) * investable_weight)
```

**Hard per-name cap (MacLean, Thorp & Ziemba 2011):**

Soft concentration penalty is insufficient to bound tail risk. Without a hard cap, a single-candidate slate can produce `w_i = 1 - cash_floor`; full-Kelly drawdowns exceed 50% even with known parameters. Enforce as deterministic projection **after** the soft concentration penalty:

```
w_i = min(w_i, maxNameWeight)
maxNameWeight = 0.10    (10% of portfolio NAV per name)

residual_mass = investable_weight - sum_j min(w_j, maxNameWeight)
Redistribute residual_mass proportionally to uncapped names.
```

Residual cash: `cash_weight = 1 - sum_i(w_i)`. Cash is not a failure when no candidate clears the hurdle.

**Cash floor:**

```
cash_floor = 0.05    (5% minimum; raise to 0.10 when posterior delta < 0)
```

The 2% legacy default is insufficient under estimation-error amplification (Michaud 1989). 5% reduces effective leverage and provides dry powder without materially reducing expected return at daily cadence.

## 2.30 Concentration (§2.30)

Concentration is correct when best opportunities materially exceed alternatives and extra return compensates for extra risk.

```
concentration_cost = concentration_scale * sum_i max(w_i - comfortable_weight_i, 0)^2
crowding_cost_i    = crowding_scale * (1 - crowding_control_i)
```

Numeric `concentration_scale`, `comfortable_weight_i`, `crowding_scale` are (unspecified in spec — TODO).

Deconcentrate only when it raises expected geometric after-cost return — not for optics.

**Cross-horizon concentration aggregation:**

If the system evaluates multiple horizons simultaneously (1d, 2-3d, 1w) the same name may appear in multiple lanes. Realized returns across 1d/2-3d/1w are 0.5–0.8 correlated (López de Prado 2018), so aggregate risk ≈ additive.

```
aggregate_weight_i = sum_h(w_i,h)   (sum across all horizon lanes)

aggregate_weight_i <= aggregateComfortableWeight = 0.20   (soft cap)
aggregate_weight_i <= maxNameWeight * n_active_horizons   (hard cap)
```

The `aggregate_weight` check runs after per-lane sizing and before any order is placed. Any name exceeding the aggregate cap has its per-lane weights proportionally reduced before submission.
