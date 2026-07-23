# Part 5 — Drift, Order Management, Shadow & Challenger (§2.31–§2.42)

## 2.31 Target drift (§2.31)

```
drift_i         = target_weight_i - current_weight_i
gross_drift     = sum_i |drift_i|
portfolio_drift = gross_drift / 2
```

Divide by 2 to avoid double-counting a rotation (a 20% shift from one stock to another is 20% away from target, not 40%).

Rebalance only if: `EV(correcting drift now) > EV(waiting) + hurdle`.

## 2.32 Decision algorithms (§2.32)

**Buy:**

```
BuyEV_i(N) = expected_return_i(N) - entry_cost_i(N) - expected_exit_cost_i(N)
           - carry_cost_i(N) - uncertainty_cost_i - opportunity_cost_of_cash_i

Buy iff: BuyEV_i(N) > max(EV_hold_cash, EV_wait, EV_buy_other, EV_repair) + hurdle
```

If full-size fails, test `N*_i = argmax_N BuyEV_i(N)` subject to `min_order <= N <= target_notional_i`; place only if `BuyEV_i(N*_i) > hurdle`.

**Sell:**

```
SellEV_i(N) = avoided_future_loss_i(N) + opportunity_value_of_freed_capital_i(N)
            - sell_execution_cost_i(N) - rebound_risk_i(N) - operational_cost_i(N)

Sell iff: SellEV_i(N) > max(EV_hold, EV_wait_for_better_exit, EV_partial_sell, EV_rotate_later) + hurdle
```

A position absent from the target basket is not automatically an immediate sell.

**Rotate:**

```
RotationEV(A -> B) = EV_hold_B - EV_hold_A - sell_cost_A - buy_cost_B
                   - transition_risk - partial_fill_pairing_risk

Rotate iff: RotationEV > hurdle
```

Partial-fill pairing risk is critical: selling A but failing to buy B leaves unintended cash; buying B but failing to sell A creates unintended concentration.

**Hold:**

```
HoldEV_i = expected_future_return_i - carry_cost_i - risk_cost_i - opportunity_cost_of_capital_i

Hold iff: HoldEV_i > max(SellEV_i, RotationEV_i, CashEV) + hurdle
```

Holding is an active choice, not the absence of a choice.

**Rebalance (package-level):**

```
PackageEV(T) = EV(resulting_portfolio) - EV(current_portfolio) - total_execution_cost(T)
             - package_partial_fill_cost(T) - package_operational_cost(T)

Execute iff: PackageEV(T) > hurdle_package
```

A package can be rejected even when one trade looks good in isolation if the combined package creates too much concentration, cash shortfall, fill mismatch, or turnover cost.

**Full action value equation:**

```
Q(a, t) = E[DeltaW_{t:t+h} | I_t, a]
        - C_exec(a,t) - C_spread(a,t) - C_slippage(a,t) - C_impact(a,t)
        - C_partial(a,t) - C_missed(a,t) - C_queue(a,t) - C_stale(a,t)
        - C_risk(a,t) - C_opportunity(a,t) + R_repair(a,t)

a*_t = argmax_{a in A_t} Q(a, t)
```

## 2.33 Repair decisions (§2.33)

Repair actions restore operational state rather than seek alpha.

```
RepairEV = value_of_restored_trading_capacity + avoided_operational_loss - repair_execution_cost

Repair iff: RepairEV > EV(waiting_with_broken_state) + hurdle
```

## 2.34 Trade timing (fluid; §2.34)

```
TradeNowValue(a, t) = expected_alpha_capture - execution_cost - future_exit_cost
                    - carry_cost - uncertainty_cost - missed_better_timing_cost

WaitValue(a, t) = expected_alpha_capture_if_later - expected_execution_cost_if_later
                + option_value_of_new_information

Trade now iff: TradeNowValue(a, t) > WaitValue(a, t) + hurdle
```

> **IEX/PDT adaptation:** Because `minimum_holding_period = 1 trading day`, `TradeNowValue(a, t)` is evaluated **once per trading day at or near session close**. The fluid trading principle still applies but "now" = "end of today's session," not "within the next intraday bar."

## 2.35 Session-aware trading (§2.35)

```
execution_cost_s = spread_cost_s + slippage_s + market_impact_s
                 + partial_fill_cost_s + missed_fill_cost_s
```

Sessions: premarket, regular, after-hours, overnight, weekend, holiday, early close.

Note: Same-day round trips in any session are blocked by `minimum_holding_period`. Under daily target cadence, the primary execution sessions are regular market open and close. Premarket / after-hours remain relevant for carry cost and gap risk assessment, not primary execution.

## 2.36 Regime dependence (§2.36)

```
mu_i   = f(x_i, regime_t)
cost_i = c(x_i, regime_t, session_t)
```

Regime examples: risk-on, risk-off, high/low vol, strong/weak breadth, earnings-heavy calendar, holiday/low-liquidity, rate-sensitive, sector rotation.

A feature can be valuable in one regime and noise in another; the same ticker can be buy, hold, sell, or wait depending on the regime.

## 2.37 Feature weighting is Bayesian, not static (§2.37)

Naive weighted sums of correlated signals double-count edge. Bayesian grouped shrinkage (§1.6) prevents this.

```
General form: mu_i = f(x_i)
Learn f(x) that best predicts realized after-cost utility.
```

## 2.38 Covariance structure — low-rank + sparse (§2.38)

```
Sigma_t = B_t * F_t * B_t' + D_t + S_t
```

`B_t` = n×k factor loadings, `F_t` = k×k factor covariance, `D_t` = diagonal idiosyncratic variance, `S_t` = sparse residual dependencies.

Portfolio risk:

```
w_t' * Sigma_t * w_t = (B_t' * w_t)' * F_t * (B_t' * w_t)
                     + sum_i(D_i,t * w_i,t^2)
                     + w_t' * S_t * w_t
```

Cost: `O(n*k + k^2 + nnz(S_t)) = Theta(n)` when `k` and sparse links per security are bounded.

Practical components:

```
Sigma = B_market * Omega_market * B_market'
      + B_group  * Omega_group  * B_group'
      + B_style  * Omega_style  * B_style'
      + B_event  * Omega_event  * B_event'
      + D_idiosyncratic
      + S_sparse_residual
```

## 2.39 Cash budget constraint (§2.39)

Every buy must be funded from available cash or same-package sell proceeds.

```
cash_trade_budget_t     = max(0, cash_t - cash_floor_t)
sell_funded_cash_t(a)   = sum over eligible sell legs j of
                          max(0, expected_sell_proceeds_j,t - reserved_sell_cost_j,t)
available_buy_budget_t(a) = cash_trade_budget_t + sell_funded_cash_t(a)

Buy leg feasible iff:
    buy_notional_t(a) + estimated_buy_cost_t(a) <= available_buy_budget_t(a)

After full package: cash_after_action_t(a) >= cash_floor_t
```

Feasibility constraint, not an alpha signal.

## 2.40 Open order management (§2.40)

```
desired_delta_t  vs  working_order_delta_t  vs  filled_delta_t  vs  broker_position_delta_t

order_state_penalty_bps = duplicate_order_penalty + stale_order_penalty
                        + partial_fill_uncertainty + cancel_replace_cost
                        + broker_state_uncertainty

OrderAction* = argmax over order actions [EV_after_cost(resulting_order_state)]
```

Do not place a new order if an existing order already expresses the same trade at an acceptable price. Do not leave a stale order working if the current optimal portfolio no longer wants it.

## 2.41 Shadow promotion monitor (§2.41)

Validated baseline surface remains active while the background monitor collects causal evidence for challengers.

**Normal-normal conjugate posterior on delta:**

```
delta_post = (kappa_0 * delta_0 + n_clean * mean(delta_shadow_clean)) / (kappa_0 + n_clean)
```

**Applied settings:**

```
prior_delta (delta_0)       = 0
prior_strength (kappa_0)    = 7
min_clean_resolved_rows     = 30       (1-day horizon; other horizons raise this — see below)
min_positive_share          = 0.55
min_posterior_delta         = 0
automatic_surface_switching = disabled
```

**Positive share:**

```
positive_share = count(delta_shadow_clean > 0) / n_clean
```

**Promotion requires all four simultaneously:**

```
n_clean >= 30
AND positive_share >= 0.55
AND delta_post > 0
AND blocking_data_quality_issues = false
```

Even when all four pass, with `automatic_surface_switching = disabled` the challenger is only marked promotion-ready — not silently deployed.

### Effective sample size — overlapping labels

Forward-return labels at horizons > 1 day overlap across consecutive rows. Treating 30 raw rows as 30 independent obs overstates confidence.

```
n_eff = n_clean / (1 + 2 * rho_label)
```

`rho_label` by horizon:

```
rho_label(1d)   = 0.00   (non-overlapping)
rho_label(2-3d) = 0.30
rho_label(1wk)  = 0.50
rho_label(2wk)  = 0.70
```

Implied `n_eff = n_clean / (1 + 2*rho_label)` → `min_clean_rows` requirements per horizon:

```
1d:   n_eff = n_clean       → min_clean_rows = 30   for min_n_eff(1d) = 30
2-3d: n_eff = n_clean/1.60  → min_clean_rows = 72   for min_n_eff(2-3d) = 45
1wk:  n_eff = n_clean/2.00  → min_clean_rows = 120  for min_n_eff(1wk) = 60
2wk:  n_eff = n_clean/3.40  → min_clean_rows = 272  for min_n_eff(2wk) = 80
```

Gate promotion on `n_eff`, not `n_clean`.

**Minimum `n_eff` per horizon:**

```
min_n_eff(1 day)   = 30
min_n_eff(2-3 day) = 45
min_n_eff(1 week)  = 60
min_n_eff(2 weeks) = 80
```

### Effective sample size — exponential decay (challenger ridge)

Under decay factor `gamma`:

```
n_eff_decay = (1 - gamma^n) / (1 - gamma)     [exact finite-sample]
```

Infinite-horizon approximation `n_eff_decay ≈ 1/(1-gamma)` valid when `gamma^n < 0.05`:

```
gamma = 0.75: valid for n > 10
gamma = 0.90: valid for n > 45
gamma = 0.95: valid for n > 90
```

Approximate values at convergence:

```
gamma = 0.75 → n_eff_decay ≈ 4
gamma = 0.90 → n_eff_decay ≈ 10
gamma = 0.95 → n_eff_decay ≈ 20
```

Use the exact formula for early buckets (small `n`).

### Posterior standard error

```
sigma_post^2  = sigma_noise^2 / (kappa_0 + n_clean)
sigma_noise^2 = sample_var(delta_shadow_clean)    [empirical Bayes plug-in]
delta_post_se = sqrt(sigma_post^2)
```

Normal-normal conjugate posterior variance of the mean (not predictive variance of individual obs). Treats `sigma_noise^2` as known — approximation becomes exact as `n_clean` grows. When `n_clean < 30`, use Student-t (`df = n_clean - 1`) for credible intervals.

**Report `delta_post` and `delta_post_se` together. Do not promote when `delta_post_se > delta_post`** (posterior mean within one SE of zero).

### Peeking correction (k = 2 consecutive cycles)

Continuous evaluation inflates type-I error (Johari et al. 2017): peeking every cycle at nominal 5% inflates false-promotion rate to 20–40%.

```
promotion_ready = (all_conditions_pass_cycle_t) AND (all_conditions_pass_cycle_{t-1})
min_spacing_between_evaluations = 1 horizon window
    (e.g., 1 trading day for 1-day horizon)
```

`k = 2` restores near-nominal significance under weak dependence. Do not raise `k` above 3 for daily-cadence systems — legitimate improvements would be rejected too long.

### Deflated Sharpe Ratio (DSR) — offline tuning gate

```
DSR = (Sharpe - E[max_null]) / SD[max_null]

E[max_null]  ≈ (1 - euler_gamma) * Phi^{-1}(1 - 1/N_trials)
             + euler_gamma * Phi^{-1}(1 - 1/(N_trials * e))
SD[max_null] ≈ Phi^{-1}(1 - 1/N_trials) * sqrt(euler_gamma * pi^2 / 6 + (1 - euler_gamma)^2)

euler_gamma ≈ 0.5772
```

Sharpe must be standardized before DSR: `Sharpe_standardized = Sharpe * sqrt(T - 1)`, where `T ≈ (backtest_trading_days - h) / h` for daily data with `h`-day horizon and purged CV (embargo = h).

**Threshold: `DSR-adjusted Sharpe < 1.0` → improvement is plausibly noise.** Nominal uplift without DSR adjustment is not valid promotion evidence.

### Shadow Promotion Monitor — SRM canary

In addition to posterior check, maintain a Sequential Ratio Monitor canary on rolling shadow stream:

```
SRM_statistic(t) = max over tau in [0,t] { log_likelihood_ratio(shadow[tau:t] | H_null) }
```

Alert if `SRM_statistic(t)` exceeds threshold (calibrated at 5% false-positive rate). SRM alert requires manual review before any promotion cycle; does not automatically block.

## 2.42 Dynamic action-surface challenger (§2.42)

More granular action-value estimator than static provenance signal.

**Research form:**

```
dynamic_source_action_value_bps(t, ticker) =
    shrink(
        bucket_prior_bps(provenance, source_package, session_bucket)
        + opportunity_arrival_credit_bps(t)
        + target_weight_confidence_bps(ticker, t)
        - cash_wait_value_bps(t)
        - current_holding_opportunity_cost_bps(ticker, t),
        toward   = static_fallback_bps,
        strength = sample_count / (sample_count + prior_strength)
    )
```

**Context features tracked per bucket:**

```
(target_weight, scaled_source_prediction, source_prediction_available,
 current_weight, delta_weight, cash_fraction)
```

**Within-bucket ridge adjustment:**

```
PRIOR_STRENGTH_KAPPA = 20    (ridge prior strength; equivalent observations)
RIDGE_LAMBDA         = 5     (L2 regularization weight)

Note: kappa_0 = 7 in §2.41 is the SHADOW promotion prior strength;
PRIOR_STRENGTH_KAPPA = 20 here is the ridge-bucket prior. Different parameters.

beta_bucket        = (X_bucket' * X_bucket + RIDGE_LAMBDA * I)^(-1) * X_bucket' * y_bucket
adjustment_bucket  = (x_current - mean_x_bucket)' * beta_bucket
```

Features must be MAD-standardized before ridge (§1.2).

**Shrinkage estimate:**

```
action_value_hat = (prior_strength * fallback_bps
                  + sample_count * (bucket_mean_bps + adjustment_bucket))
                 / (prior_strength + sample_count)
```

Additional constants (bucket state management):
- `MAX_SAMPLES_PER_BUCKET` — (unspecified in spec — TODO; used for bucket capacity)
- `MIN_SAMPLES_FOR_RIDGE` — (unspecified in spec — TODO; minimum before ridge activates over fallback)
- Decay factor `gamma = 0.75` implied by §2.41 approximate `n_eff_decay ≈ 4` example.

A short high-CAGR window for the challenger is diagnostic, not promotion evidence. Promotion evidence requires full-window deployability testing or matured prospective shadow outcomes.

## 2.43 Learning target (§2.43)

```
label = realized_return - realized_entry_cost - realized_exit_cost
      - realized_slippage - realized_opportunity_cost
```

Features are judged on OOS predictive lift against this label in walk-forward testing — not on whether they sounded reasonable.

**Store for every decision:** state at decision time, candidate action, chosen action, rejected alternatives, predicted alpha/cost/fill probability, actual fill, actual slippage, actual subsequent return, actual opportunity cost.

## 2.44 What must never happen (§2.44)

- Trade because target changed without checking EV
- Buy when gross alpha is positive but after-cost alpha is negative
- Sell because a holding is visually stale without checking exit EV
- Rotate without checking both sell cost and replacement buy surplus
- Ignore partial-fill or missed-fill risk
- Ignore weekend/overnight carry risk
- Ignore stale data
- Double-count correlated signals
- Compare strategies under different execution assumptions
- Treat paper CAGR as realistic after-cost CAGR
- Hide cash dwell without explaining the opportunity-cost reason
- Allow non-economic safety gates to block positive-EV trades
- Allow non-economic aggression to force negative-EV trades
