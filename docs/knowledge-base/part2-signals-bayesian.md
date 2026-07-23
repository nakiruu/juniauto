# Part 2 — Signals and Bayesian Machinery (§2.1–§2.8)

## 2.1 Core objective — log-wealth (§2.1)

Choose the action that maximizes expected after-cost compounded portfolio growth.

```
A*_t = argmax_A { E[log(W_{t+H}) | S_t, A] - TotalCost(A | S_t) }
U(W) = log(W)
```

Utility is geometric because trading compounds; raw expected return ignores volatility drag, drawdown, and ruin risk.

Expected log growth for candidate weight vector `w`:

```
G(w) = (w' * mu)
     - (0.5 * gamma_risk(t) * w' * Sigma_risk(t) * w)
     + (lambda_confirm(t) * w' * Sigma_confirm(t) * w)
     - C_transition(w_current -> w)
     - C_future_exit(w)
     - C_uncertainty(w)
     - C_liquidity(w)
     - C_concentration(w)
     - C_carry(w)
     - C_operational(w)
```

Optimal holding: `w* = argmax_w G(w)`.

Non-concavity note: `+lambda_confirm` makes `G` non-concave when `lambda_confirm > 0` and `Sigma_confirm` is PD. Default `lambda_confirm = 0.05` is small; solve the concave sub-problem (`lambda_confirm = 0`) then apply a rank-one update for the confirmatory term. Valid only when the confirmatory correction is small relative to the risk-adjusted-return term.

## 2.2 Target vs execution separation (§2.2)

Do not conflate holding optimization and trade timing.

```
optimal_target != immediate_order
```

Holding optimization: "What should the account ideally own?"
Execution optimization: "Given current holdings, what should it trade right now?"

A security can be in the optimal target and still be a wrong buy right now (wide spread); a security removed from the target can be a wrong sell right now (exit cost > waiting cost).

## 2.3 Expected return decomposition (§2.3)

Full form:

```
mu_i_H = P(up_i_H) * E[return | up]
       + P(flat_i_H) * E[return | flat]
       + P(down_i_H) * E[return | down]
```

Simplified (flat ≈ 0):

```
mu_i_H = P(up_i_H) * upside_i_H - P(down_i_H) * downside_i_H
```

**Long-only positive-edge form** (only credits probability above neutral, applied in this system since Alpaca long-only):

```
positive_edge_i_H = max(P(up_i_H) - 0.5, 0) * conditional_upside_i_H
```

Full gross alpha (if downside estimable):

```
gross_alpha_i_H = P(up_i_H) * conditional_upside_i_H - (1 - P(up_i_H)) * conditional_downside_i_H
```

Conservative proxy when downside magnitude unknown:

```
gross_alpha_i_H = max(P(up_i_H) - 0.5, 0) * conditional_upside_i_H
```

## 2.4 Multi-horizon forecasting (§2.4)

```
H in {1 day, 2-3 days, 1 week, 2 weeks, 1 month}
```

> **IEX/PDT adaptation:** No sub-day entries (minutes, hours). Reasons: (a) 15-minute delayed feed makes intraday signals unactionable; (b) PDT forces 1-trading-day minimum hold, so sub-day round trips are structurally unavailable.

For each horizon, estimate `mu_i_H`, `sigma_i_H`, `confidence_i_H`, `cost_i_H`.

Preferred horizon:

```
H*_i = argmax_H [ mu_i_H - cost_i_H - risk_penalty_i_H - uncertainty_penalty_i_H ]
```

Signal-to-horizon mapping (minimum actionable horizon = 1 trading day):
- event momentum: hours to days
- fundamental quality: days to months
- sector rotation: days to weeks

Excluded: "live spread and order book: seconds to minutes" and "intraday structure: minutes to hours" — neither is actionable with the IEX + PDT stack.

## 2.5 Signal confidence (§2.5)

Every estimate has a paired confidence `q_i ∈ [0, 1]`.

Confidence-adjusted expected return (shrinks uncertain estimates toward zero):

```
mu_adj_i = q_i * mu_i + (1 - q_i) * mu_prior_i     with mu_prior_i = 0
```

Uncertainty as explicit cost:

```
uncertainty_cost_i = uncertainty_scale * (1 - q_i)
```

Confidence rises with fresh data, independent confirming signals, high liquidity, low missingness, stable definitions, validated historical predictive power, current-session quote availability, event confirmation. Confidence falls with staleness, missing bid/ask, thin liquidity, conflicting signals, unproven features, binary event risk, semantic ambiguity, abnormal volatility, newly listed uncertainty.

## 2.6 Bayesian posterior edge (§2.6)

The system maintains a posterior distribution over action value.

```
E[mu_i | D_t]       — posterior mean
Var(mu_i | D_t)     — posterior variance
```

Uncertainty-discounted edge (conservative credible bound):

```
conservative_edge(i, t, h) = mu_edge(i, t, h) - zq * sigma_total(i, t, h)
```

```
zq = 1.0
```

Requires posterior mean edge to exceed one posterior SD — approximately 84% of the posterior mass above zero edge. At daily cadence with < 500 trades/year, false positives cost more than false negatives; `zq = 1.0` balances conservatism against opportunity. Coherent with `rho = rho_h = 1.0`.

**Calibration levers:**
- If realized false-positive rate on executed trades > 45%, raise to `zq = 1.282` (~90% credible bound).
- If fill rate drops below 15% of eligible candidates while known-strong signals are being rejected, cut to `zq = 0.842` (~80% credible bound).

Total predictive uncertainty:

```
sigma_total^2(a, t) = sigma_edge^2(a, t)
                   + sigma_quote_staleness^2(a, t)
                   + sigma_liquidity_state^2(a, t)
                   + sigma_order_state^2(a, t)
                   + sigma_regime_shift^2(a, t)
```

Probability of positive after-cost value:

```
P_positive(a, t) = 1 - Phi((0 - mu_edge(a, t)) / sigma_total(a, t))
```

Probability of clearing hurdle `H_a(t)`:

```
P_clears_hurdle(a, t) = 1 - Phi((H_a(t) - mu_edge(a, t)) / sigma_total(a, t))
```

Risk-adjusted alpha:

```
risk_adjusted_alpha_i = E[mu_i | D_t] - lambda_uncertainty * Var(mu_i | D_t)
```

`lambda_uncertainty` is (unspecified in spec — TODO).

Posterior update: `P(mu_i | D_t) ∝ P(D_t | mu_i) * P(mu_i)`.

## 2.7 Full Bayesian calculation stack (§2.7)

**Research target** (label the ridge tries to predict):

```
y_i,t,h = realized_forward_return_bps(i, t, h)
        - realized_entry_cost_bps(i, t)
        - realized_exit_cost_bps(i, t+h)
        - realized_slippage_bps(i, t)
        - realized_queue_delay_value_bps(i, t)
        - realized_opportunity_cost_bps(i, t)
```

**Grouped regression:**

```
y_n = alpha + sum_g(X_n,g * beta_g) + Z_n * delta + epsilon_n
epsilon_n | sigma^2 ~ N(0, sigma^2 / w_n)
```

**Posterior precision (conditional on active group set A):**

```
Lambda_A = X_A' * W * X_A / sigma^2 + V_0,A^(-1)
V_A = Lambda_A^(-1)
m_A = V_A * (X_A' * W * y / sigma^2 + V_0,A^(-1) * m_0,A)
theta_A | D_t, A, sigma^2 ~ N(m_A, V_A)
```

**Plug-in sigma^2 treatment:** Throughout §2.6–§2.8, `sigma^2` is treated as known (fixed empirical Bayes plug-in from historical feature-return data). The `sigma^2` in `Lambda_A` uses the point estimate, not a posterior mean. Under this plug-in, `theta` is Gaussian. With a conjugate inverse-chi-squared prior, marginal theta would be multivariate Student-t (heavier tails) and `zq` bounds would be approximate. Gaussian approximation is adequate when calibration set > 500 obs per group.

**Posterior predictive for action `a`:**

```
mu_edge(a, t)         = E[y* | x_a,t, z_a,t, D_t]
sigma_edge^2(a, t)    = E[sigma^2 | D_t]
                      + x_a,t' * Var(theta | D_t) * x_a,t
                      + sigma_model_misspecification^2(a, t)
```

**Cached group posterior summary** (persist between ticks):

```
posterior_group_summary_g = (
    P(gamma_g = 1 | D_t),
    E[beta_g | D_t],
    Var(beta_g | D_t),
    E[tau_g | D_t],
    n_effective_g,
    last_valid_update_g
)
```

**Group utility (research selection):**

```
utility_k = m_k - rho * sqrt(V_k,k)
rho = 1.0
```

A group earns positive utility only when posterior mean contribution exceeds one posterior SD (one-sigma signal-to-noise). Approximation: uses marginal SD `sqrt(V_k,k)`, ignoring off-diagonal posterior covariance between groups (joint selection would require 2^G combinatorial set-cover).

**Calibration levers:**
- Decrease `rho` toward 0.5 if known-predictive groups are being systematically excluded (`gamma_g = 0` for groups with positive OOS hit rate).
- Increase toward 1.5 if low-evidence groups are consistently included.

Cost per decision tick: `C_bayes_tick(t) = O(G * d_max^2 + n * p_active) = Theta(n)`.

## 2.8 Feature contribution accounting (§2.8)

Local, posterior-weighted per indicator:

```
indicator_contribution_j,i,t,h = E[beta_j,h | D_t-] * z_j,i,t
```

Uncertainty-discounted contribution:

```
discounted_contribution_j,i,t,h = indicator_contribution_j,i,t,h
                                 - rho_h * sqrt(Var(beta_j,h | D_t-)) * |z_j,i,t|
rho_h = 1.0    (flat)
```

**Deploy flat `rho_h = 1.0` first.** Only differentiate by horizon after observing OOS prediction stability. Optional horizon-differentiated schedule:

```
rho_h(1 trading day) = 1.0    (highest uncertainty; daily signal is noisiest)
rho_h(2-3 days)      = 0.9
rho_h(1 week)        = 0.8
rho_h(2 weeks)       = 0.7
```

Coherent with `rho = 1.0` and `zq = 1.0` — same one-sigma penalty across research selection, credible bound, and per-indicator discounting.

**Share statistics for indicator explanation reports (§3.10):**

```
signed_share_j,i,t,h = indicator_contribution_j,i,t,h / max(epsilon, sum_l |indicator_contribution_l,i,t,h|)
abs_share_j,i,t,h    = |indicator_contribution_j,i,t,h| / max(epsilon, sum_l |indicator_contribution_l,i,t,h|)
family_abs_share_g,i,t,h = sum_{j in g} |indicator_contribution_j,i,t,h| / max(epsilon, sum_l |indicator_contribution_l,i,t,h|)
```

**After-cost edge decomposed at three levels:**

```
pre_cost_edge_i,t,h   = alpha_h,t
                      + sum_j(indicator_contribution_j,i,t,h)
                      + source_role_edge_i,t,h
                      + regime_adjustment_i,t,h
after_cost_edge_i,t,h = pre_cost_edge_i,t,h
                      - execution_cost_i,t
                      - uncertainty_discount_i,t,h
                      - wait_value_hurdle_i,t
```

`after_cost_edge_i,t,h` feeds the composite `model_edge_bps` (§2.22a) and thence the gate (§2.24–§2.26).
