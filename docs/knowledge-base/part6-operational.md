# Part 6 — Operational, Replay, Metrics (§3.1–§3.14)

## 3.1 Account constraints (§3.1)

All optimization is bounded by real account state.

```
Feasible action set F_t = { A_t : A_t satisfies account, broker, and market constraints }
```

Constraint types: cash, cash available to trade, existing positions, pending orders, minimum order size, whole-share rounding, order session rules, cash floor, order rate limits, brokerage/API payload rules.

> **IEX/PDT adaptation — additional constraints active:**
> - `day_trades_rolling_5_day_window <= 3` (PDT hard constraint; account < $20,000; day trade = open+close same security same calendar day)
> - `minimum_holding_period = 1 trading day` (derived; position opened day D cannot generate a sell order on day D for the same security)

```
Actual decision: A*_t = argmax_{A in F_t} EV_after_cost(A | S_t)
```

A mathematically attractive trade that cannot be validly placed is not a feasible action.

## 3.2 Complete trading cycle — 7 steps (§3.2)

Per decision tick:

1. **Refresh observations.** Market data, account balances, positions, open orders, authorization state, feature freshness. If critical state stale/invalid, correct action may be repair, wait, or avoid real orders.
2. **Source package selector.** Decides which source package has current authority using prediction-sign regime evidence, not raw ticker rank. Applies `G_p,t >= 95 bps` threshold (§2.20).
3. **Target gateway.** Filters compact source slate to currently tradable, evidence-qualified members; applies source-conviction weighting. A single ticker can receive most of the book when its opportunity dominates; no fixed cap.
4. **Provenance membership.** Assigns role edge: primary = 460 bps, secondary = 348 bps, retained = 200 bps. Discounted by adaptive dynamic-friction multiplier starting at 0.30 baseline seed.
5. **Execution gate.** Evaluates each required action. Rotations evaluated as sell-funded packages, not cash-only buys. Insufficient edge vs spread/slippage/queue/wait-value → hold or wait.
6. **Route orders.** Only if account and order state allow. Open orders and partial fills affect the next cycle.
7. **Record outcomes.** Prospective shadow monitor rows mature when action window resolves. **PDT day-trade counter updates on each resolved fill pair. If `day_trades_rolling_5_day_window = 3`, no new same-day round-trip orders may be generated until the oldest day trade ages out.**

User-facing summary:
1. Which strategy source is working best right now?
2. Which stocks would that source want to own?
3. Is moving toward those stocks worth the real cost of trading?
4. Are there account/brokerage issues that make waiting safer?
5. Did the decision later prove better than the alternative?

> **IEX/PDT adaptation:** Decision cycle runs daily, evaluation at or near regular session close, consistent with daily bar frequency and 1-day minimum hold.

## 3.3 Order type selection (§3.3)

```
EV_limit(L) = P_fill(L) * (alpha - C_fill(L)) + (1 - P_fill(L)) * (-C_miss(L))
EV_market   = alpha - spread_cost - slippage_cost - impact_cost
```

Use market order only if: `EV_market > max(EV_limit, EV_wait, EV_hold, EV_cash) + hurdle`.

Increase limit aggression when urgency rises, session deadline approaches, alpha decay risk rises, existing order has not filled, opportunity justifies worse price.

Stay passive or defer when spread too wide, fill probability poor, alpha weak, waiting has higher option value.

## 3.4 Session-specific execution costs (§3.4)

```
execution_cost_s = spread_cost_s + slippage_s + market_impact_s
                 + partial_fill_cost_s + missed_fill_cost_s
```

Session behavior:
- **Regular market:** tighter spreads, better liquidity
- **Premarket:** early information advantage but wider spreads, lower liquidity
- **After-hours:** event-reaction opportunity but higher spread and partial-fill risk
- **Friday close:** lower ability to exit soon, higher weekend carry

**`session_multiplier` (Chordia-Roll-Subrahmanyam 2001; Barclay-Hendershott 2003):**

```
regular  (09:30–16:00 ET):  session_multiplier = 1.0    (baseline)
premarket (04:00–09:29 ET): session_multiplier = 1.5    (spreads 1.4–1.8x wider)
after-hours (16:01–20:00):  session_multiplier = 2.0    (spreads 2–3x wider)
overnight / holiday / non-trading: session_multiplier = 2.5   (no continuous market)
```

Multiplies the parenthesized terms in `base_side_cost_bps` (§2.24) and determines additive increment in `queue_delay_risk_bps` and `cancel_replace_risk_bps`.

> **IEX/PDT adaptation:** Under 1-day minimum hold + daily EOD cadence, regular session (1.0) is the operative value for the vast majority of decisions.

## 3.5 Self-testing replay proof harness (§3.5)

Verifies the implementation computed the intended causal decision rule — not that future markets will pay.

**Self-test certificate:**

```
self_test_certificate = indicator[
    for all t in replay_window:
        features_are_causal_t
        AND costs_are_recomputable_t
        AND account_state_transition_matches_t
        AND no_unresolved_required_field_t
        AND no_unexpected_missing_value_t
        AND OI(state_t) = OS(state_t)
]
```

`OI(state_t) = OS(state_t)` = intended output = observed output for that state (deterministic replay match).

**Promotable condition:**

```
promotable(theta) = self_test_certificate(theta) = 1
                  AND full_window_after_cost_return(theta) > incumbent_full_window_after_cost_return
                  AND missing_required_rows(theta) = 0
                  AND unresolved_action_rows(theta) = 0
                  AND deployability_constraints_pass(theta) = 1
                  AND causal_or_shadow_evidence_pass(theta) = 1
```

**Replay checklist:**
1. Did it only use information available at the time?
2. Did it pay spread, slippage, queue, gap, stale-quote, reversal, and wait costs?
3. Did it respect cash-only action feasibility and order-state repair?
4. Did it beat holding, waiting, and alternate rotations?
5. Did it survive the full window rather than only a favorable slice?
6. Did every required row resolve cleanly?

## 3.6 Performance metrics (§3.6)

Standard metrics: CAGR, total return, max drawdown, volatility, Sharpe-like ratio, turnover, cost drag, slippage, partial-fill frequency, missed-fill cost, liquidity-limited dollars, capacity, percentage of trades with positive realized after-cost edge, cash dwell reason quality, dynamic friction all-leg pass rate, partial package rate.

> **IEX/PDT adaptation — expected operational profile under daily cadence** (all marked as recalibration required):
> - target cadence: end-of-day (daily)
> - average trades/year: [recalibration required; expected < 500 given 1-day min hold and daily cadence]
> - average turnover/year: [recalibration required; expected < 20x given daily cadence vs 109x intraday]
> - average annual cost drag: [recalibration required; expected materially lower than 47.6% given fewer trades]

Lower turnover and cost drag under daily cadence improve strategy survival under fill-quality, spread, and order-behavior degradation vs prior intraday config.

### Per-family per-horizon rolling IC

```
IC_f,h(t) = corr(rank(signal_score_family_f, t), rank(realized_net_bps_h, t))
```

Trailing 63-trading-day window (≈ one quarter). `f ∈ {Technical, Fundamental, Event, Semantic, Liquidity, Risk/Crowding}`. `h ∈ {1d, 2d, 3d, 1wk, 2wk}`.

Cross-reference: 10-day purged-CV embargo (§3.7) reduces effective training window per fold from 63 to ~53 usable trading days. Per-horizon `n_eff` minimums in §2.41 are derived after this reduction.

**IC alerting:**
- Healthy: `IC_f,h(t) > 0.04` for primary families
- Degraded: `|IC_f,h(t)| < 0.02` for three consecutive trailing windows → flag for review
- Flipped: `IC_f,h(t) < -0.03` → **mandatory halt** of that family's contribution pending investigation

Report `IC_f,h` alongside CAGR / Sharpe at each evaluation checkpoint.

### Signal Rank Preservation Coefficient (SRPC)

```
SRPC(t) = corr(rank(paper_edge_bps_i, t), rank(executed_edge_bps_i, t))   over active positions
```

Distinct from Grinold-Kahn (2000) Transfer Coefficient (which is `corr(w_active_paper, w_active_executed)` — a weight correlation). SRPC measures how well execution preserves rank-ordering of signal strength; a system with TC ≈ 1.0 can have low SRPC if partial fills invert rank ordering. Report both if a full Fundamental Law implementation is used.

**Threshold: `SRPC < 0.80` indicates execution is materially reordering positions relative to intended signal priority.**

### Alpha decay τ per family

Fit log-linear IC decay per family across horizons:

```
log(IC_f(h)) = log(IC_f,0) - h / tau_f
```

OLS on trailing 63-day IC estimates across available horizons. Short `tau_f` (fast decay) contributes primarily at 1d, should receive lower weight at 2wk. Long `tau_f` contributes across horizons and supports multi-period holding.

Practical use: when `composite_IC(h) < 0.01`, do not extend holding to that horizon regardless of prior-strength priors.

## 3.7 Overfit risk and generalization (§3.7)

**Anti-overfit rules:**
1. A short high-CAGR gap-window row is diagnostic only.
2. Full-window zero-missing zero-unresolved evidence is required for promotion-grade replay proof.
3. Prospective shadow outcomes are stronger than curve-level hindsight.
4. Action-level rows must be resolved under the same information constraints the practical executor would have had.
5. Dynamic-action challengers need enough clean samples before replacing static provenance membership.

### Purged rolling-origin CV (required for all offline evaluations)

Random K-fold CV is invalid for overlapping forward-return labels. Leakage inflates in-sample IC/Sharpe by roughly `sqrt(h)`.

```
For each test fold starting at t_k:
    train on [0, t_k - embargo_gap]
    purge all training rows whose forward window overlaps test period
    embargo_gap > h_max   (strictly greater; discards one full max-horizon window)
    test on [t_k, t_k + fold_size]

embargo_gap = max(H) + 1 = 11 trading days
```

Embargo must be **strictly greater** than `h_max` to prevent any training-set label whose forward window touches the first test observation. With 2-week (10-day) `h_max`, `embargo_gap = h_max + 1 = 11` closes the boundary. `embargo_gap = h_max = 10` leaves the boundary observation unembargoed.

Any Sharpe/IC without purged CV is not valid promotion evidence.

### Probability of Backtest Overfitting (PBO)

When evaluating `N_trials` configurations offline (grid search, signal weights, feature selection), use Combinatorially Symmetric CV (Bailey & López de Prado 2014):

```
1. Partition time series into M non-overlapping sub-periods (M = 16 standard).
2. Form all C(M, M/2) combinations of M/2 in-sample; remaining M/2 out-of-sample.
3. For each combination m:
   a. Select c*_m with best in-sample perf across M/2 in-sample sub-periods.
   b. omega_m = fractional out-of-sample rank of c*_m across all N_trials (in (0,1]; 1 = best OOS).
4. PBO = (1/M_total) * sum_m 1{ omega_m <= 0.5 }
```

Indicator applies to out-of-sample fractional rank, **not** to in-sample rank (which is always max = `N_trials` by definition).

**Threshold: `PBO > 0.50` → do not promote.** More than half the time the best in-sample configuration performs no better than median OOS.

### Deflated Sharpe Ratio (DSR) — offline gate

Same definition as §2.41. Report DSR alongside any grid-search result. `DSR-adjusted Sharpe < 1.0` → improvement is plausibly noise. Nominal Sharpe without DSR is not valid offline promotion evidence.

### Generalized Bayesian promotion selector

```
Delta_c,b,t = value_challenger_t - value_baseline_t
Delta_c,b,t | theta, x_t ~ N(x_t' * theta, sigma_delta^2)
theta ~ N(theta_0, V_0)

Promote when: P(E[Delta_c,b | current practical distribution] > 0 | data) is high
```

not on a single historical threshold.

## 3.8 Comparable testing (§3.8)

Strategies must be compared under identical assumptions or one appears better only because it was graded under easier assumptions.

Required common assumptions: same date window, same universe rules, same signal freshness assumptions, same transaction cost model, same spread/slippage model, same partial-fill model, same liquidity capacity model, same order timing assumptions, same account constraints, same promotion criteria.

## 3.9 Evaluation metric priority (§3.9)

```
Maximize expected future after-cost CAGR subject to realistic execution and survivability constraints.
```

Improvements target one or more of: expected return, conditional upside/downside, `P(up)`, signal confidence, feature redundancy, regime dependence, entry/exit spread, slippage, market impact, fill probability, partial-fill cost, missed-fill cost, opportunity cost, carry risk, portfolio covariance, drawdown/geometric growth penalty.

If a change does not improve one of those estimates or the ability to act on them, it probably does not improve returns.

## 3.10 Indicator explanation reports (§3.10)

Per-ticker report sorted by absolute share descending.

```
indicator_report(i, t, h) = sort_descending_by_abs_share({
    indicator_name_j, source_family_j, z_j,i,t, E[beta_j,h | D_t-],
    indicator_contribution_j,i,t,h, discounted_contribution_j,i,t,h, abs_share_j,i,t,h
} for j = 1 to p)
```

Cost: `O(p log p)` per ticker; `O(m * p * log p)` for full candidate slate (bounded).

**Explanation levels:**
1. Raw alpha contribution: which indicators made the name attractive
2. Discount contribution: which uncertainty or regime terms reduced confidence
3. Gateway contribution: which execution costs converted a target into a trade, wait, or reject

## 3.11 Indicator reference by source family (§3.11)

Representative indicators (amount = `E[beta_j | D] * z_j`). Summary only — see spec for full list.

- **Technical:** `trend_score`, `relative_strength_score`, `breakout_score`, `upside_torque_score`, `next_session_green_probability`, `support_defense_score`, `extension_alignment_score`, `oversold_reversal_score`
- **Volatility / range / execution capacity:** `realized_volatility_bps`, `atr_opportunity_score`, `atr_beta_risk_posture`, `volume_shock_score`, `spread_bps` (cost side; amount = `-C_spread`), `quote_age_seconds` (amount = `-C_stale`), `dollar_volume_capacity` (amount = `-C_liquidity`), `imminent_tradeability_score`
- **Fundamental:** `quality_growth_score`, `relative_strength_quality_score`, `liquidity_quality_score`, `earnings_quality_proxy`
- **Source / event / regime:** `primary_source_member` (amount = `source_role_edge_primary * dynamic_multiplier`), `secondary_source_member`, `retained_member`, `prior_target_weight`, `company_specific_catalyst`, `semantic_relative_conviction`, `sector_stabilization_score`, `market_return_5_bps`, `market_volatility_21_bps`, `breadth_state`
- **Calendar / portfolio / execution telemetry:** `session_bucket`, `gap_days_to_next_trading_session` (amount = `-C_gap`), `current_weight`, `target_delta`, `open_order_state` (amount = `-C_replace_or_duplicate`), `recent_slippage_bps`, `partial_fill_state`

## 3.12 Replay validation costs (§3.12)

```
C_self_test(B, T, n) = O(B * T * (n * p_active + G * d_max^2 + n * k + n * log(m) + m * q + events_t))
                     = Theta(B * T * n)   under bounded design dimensions
```

## 3.13 Computational complexity summary (§3.13)

```
C_theory_dense(B, T, n)   = Theta(B * T * n^3)   storage Theta(n^2)
C_theory_pairwise(B, T, n) = Theta(B * T * n^2)   storage Theta(n^2)

C_applied(B, T, n) = O(B * T * (n * p_active + G * d_max^2 + n * k + n * log(m) + m * q))
                   = Theta(B * T * n)   under bounded design dimensions

S_applied(n) = O(n * p_active + n * k + n + m * q + events_t)
```

`C_applied = o(B * T * n^2)`; `lim_{n→∞} C_applied / C_theory_pairwise = 0`.

**Key assumptions for linear bound:**
1. `p_active, H, G, d_max, k, m, q` bounded by model design
2. Sparse residual risk links per security bounded
3. Candidate pruning happens **before** pairwise action expansion
4. Quotes, features, posterior summaries, costs, account state cached once per timestamp
5. Replay uses same causal state transition as practical evaluation

**Three approximation concessions:**
1. Cached group sufficient statistics instead of full joint posterior every tick
2. Low-rank + sparse risk instead of dense n×n covariance inversion
3. Evidence-qualified candidate slate instead of all-pairs rotation enumeration

## 3.14 Candidate pruning cost (§3.14)

Most important computational move: prune **before** action expansion.

```
proxy_edge_i,t,h = E[y_i,t,h | D_t-, X_i,t] - preliminary_cost_floor_i,t - uncertainty_discount_i,t,h

M_t = top_m({ i in U_t : proxy_edge_i,t,h clears eligibility floor })
|M_t| = m << n
```

Ranking cost: `O(n log m)` bounded heap, or `O(n)` linear-time selection.

Gateway cost applied only after pruning: `|A_t| = O(m * q)`, `C_gate(t) = Theta(|A_t|) = O(m * q)`. When `m` and `q` are bounded, gateway is constant-time relative to full universe.
