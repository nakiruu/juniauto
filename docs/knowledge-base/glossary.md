# Glossary — Symbols, Constants, and Section Pointers

All entries are as used in `PRINCIPLESLONG.md`. When multiple sections use the same symbol, the primary defining section is listed first.

## Greek symbols

| Symbol | Meaning | Section |
|---|---|---|
| `alpha` | Regression intercept in grouped ridge; also generic model edge input to `EV_limit` / `EV_market` (§2.14, §2.25) | §2.7, §2.14 |
| `alpha_fill` | Per-fill exponential decay factor for `recent_fill_slippage_bps`, = 0.75 (cold-start 0.90) | §2.24 |
| `beta_g` | Coefficient vector for feature group g in grouped ridge | §1.6, §2.7 |
| `beta_bucket` | Ridge coefficients from within-bucket regression in challenger | §2.42 |
| `gamma` | Source-conviction boost exponent in `b_i,t`, applied = 1 (linear) | §2.21 |
| `gamma` (decay) | Exponential decay factor for challenger sample weighting (0.75 / 0.90 / 0.95) | §2.41 |
| `gamma_g` | Spike-and-slab inclusion indicator for group g (`gamma_g ∈ {0, 1}`) | §1.6 |
| `gamma_risk(t)` | Time-varying risk aversion in `G(w)`; = 1 in operative Kelly config | §2.1, §2.29 |
| `delta_0` (posterior prior) | Prior mean on shadow delta = 0 | §2.41 |
| `delta_0(beta_g)` | Dirac point mass at zero (spike component) | §1.6 |
| `delta_post` | Normal-normal posterior mean of shadow challenger delta | §2.41 |
| `delta_post_se` | Standard error of `delta_post` | §2.41 |
| `epsilon_n` | Regression noise term, `~ N(0, sigma^2 / w_n)` | §2.7 |
| `eta_0`, `eta_role`, ... | Logit-Bayesian coefficients for adaptive friction multiplier | §2.23 |
| `theta` | Concatenated regression coefficient vector | §2.7, §3.7 |
| `kappa_0` | Shadow promotion prior strength (§2.41), = 7. **Distinct from `PRIOR_STRENGTH_KAPPA` = 20 in §2.42.** | §2.41 |
| `Lambda_A` | Posterior precision matrix for active group set A | §2.7 |
| `lambda_confirm(t)` | Confirmatory covariance reward weight (default 0.05) | §2.1 |
| `lambda_g` | Group shrinkage strength in spike-and-slab | §1.6 |
| `lambda_uncertainty` | Risk-adjusted alpha uncertainty penalty (unspecified in spec — TODO) | §2.6 |
| `mu` | Expected return vector | §2.1, §2.29 |
| `mu_i` | Confidence-adjusted expected return, per candidate | §2.5 |
| `mu_i_H` | Expected return of candidate i over horizon H | §2.3, §2.4 |
| `mu_edge(a,t)` | Posterior predictive mean edge for action a at t | §2.6, §2.7 |
| `mu_prior_i` | Neutral prior for shrinkage, = 0 | §2.5 |
| `pi_g` | Prior inclusion probability of group g | §1.6 |
| `rho` | Group utility marginal-SD penalty, = 1.0 (coherent one-sigma) | §2.7 |
| `rho_h` | Per-indicator contribution SD penalty, = 1.0 flat (optional horizon ladder 1.0/0.9/0.8/0.7) | §2.8 |
| `rho_label` | Label autocorrelation under overlapping horizons: 0.00 / 0.30 / 0.50 / 0.70 for 1d / 2-3d / 1wk / 2wk | §2.41 |
| `sigma^2` | Regression noise variance; plug-in constant in §§2.6–2.8 and §2.41 (empirical Bayes) | §2.7 |
| `sigma_edge^2` | Posterior predictive edge variance component | §2.6, §2.7 |
| `sigma_noise^2` | Empirical noise variance of shadow deltas (= `sample_var(delta_shadow_clean)`) | §2.41 |
| `sigma_post` | Standard error of shadow posterior mean | §2.41 |
| `sigma_total` | Combined predictive uncertainty for action a | §2.6 |
| `Sigma_confirm` | Confirmatory covariance object (rewarded via `lambda_confirm`) | §2.1, §2.28 |
| `Sigma_risk` | Risk covariance object (penalized via `gamma_risk`) | §2.1, §2.28 |
| `Sigma_t` | Portfolio covariance decomposed as `B*F*B' + D + S` | §2.38 |
| `tau_g` | Group-specific slab scale | §1.6 |
| `tau_f` | Alpha decay time constant per family in `log(IC_f(h)) = log(IC_f,0) - h/tau_f` | §3.6 |
| `Phi` | Standard normal CDF | §2.6 |

## Constants and thresholds

| Name | Value | Meaning | Section |
|---|---|---|---|
| `zq` | 1.0 | Quantile-style caution parameter in `conservative_edge` | §2.6 |
| `SQRT_IMPACT_COEFF` | 25 | Bouchaud square-root impact coefficient (not 9) | §2.12 |
| `PRIOR_STRENGTH_KAPPA` | 20 | Ridge prior strength in challenger bucket regression | §2.42 |
| `RIDGE_LAMBDA` | 5 | Challenger ridge L2 weight | §2.42 |
| `maxNameWeight` | 0.10 | Hard per-name weight cap (10% NAV) | §2.29 |
| `cash_floor` | 0.05 | Minimum cash weight (raise to 0.10 in adverse regimes) | §2.29 |
| `aggregateComfortableWeight` | 0.20 | Cross-horizon per-name soft cap | §2.30 |
| `evidence_threshold` (source select) | 95 bps | Minimum `G_p,t` to activate a source package | §2.20 |
| `source_selection_decay` | 1.0 | No exponential decay in active configuration | §2.20 |
| `retained_baseline_floor` | 200 bps | Membership prior for retained role | §2.19, §2.22 |
| `primary_role_signal` | 460 bps | Membership prior for primary role | §2.19, §2.22 |
| `secondary_role_signal` | 348 bps | Membership prior for secondary role | §2.19, §2.22 |
| `dynamic_friction_multiplier` seed | 0.30 | Prior on adaptive friction (all roles) | §2.19, §2.23 |
| `exit_reserve` | 1.00 | Fraction of modeled exit cost reserved at buy | §2.19, §2.24 |
| `effective_execution_horizon` | 1 trading day (390 min) | Time scale over which edge must remain meaningful | §2.19, §2.27 |
| `minimum_hurdle_bps` | 0 | Governance floor on required edge | §2.19, §2.26 |
| `minimum_holding_period` | 1 trading day | PDT-derived structural minimum | §2.19, §3.1 |
| `max_day_trades_rolling_5_day_window` | 3 | PDT hard constraint | §2.19, §3.1 |
| `BUY exit haircut` | 0.65 × 0.85 = 0.5525 | Composite factor on `base_side_cost_bps` for reserved future exit | §2.24 |
| `fallback_round_trip_cost_bps` | 60 | Reversal fallback until first ticker fill | §2.24 |
| `api_budget_cost_bps` | 2 | Alpaca REST cancel-failure proxy | §2.24 |
| `lost_queue_priority_bps` | 1 | Alpaca PFOF marginal reprice on replace | §2.24 |
| `adverse_selection_share` regular / extended / closed | 0.35 / 0.55 / 0.90 | Barclay-Hendershott spread-share | §2.24 |
| `action_memory_horizon_seconds` | 23,400 (= 390 × 60) | One trading day of market seconds | §2.24 |
| `cash_waiting_value_bps` | `(r_cash / 252) × 10000` | Opportunity value of cash per trading day | §2.24 |
| `r_cash` | broker sweep, else Fed Funds − 50 bps | Cash rate input | §2.24 |
| `session_multiplier` reg / pre / after / closed | 1.0 / 1.5 / 2.0 / 2.5 | Session cost scalar | §3.4 |
| `operational_risk_bps` cold-start | 10 (cap 40) | Buffer for latency, API, position mismatch | §2.26 |
| `kappa_0` (shadow prior strength) | 7 | Normal-normal conjugate prior weight | §2.41 |
| `min_clean_resolved_rows` (1d) | 30 | Shadow promotion sample floor | §2.41 |
| `min_positive_share` | 0.55 | Shadow promotion positive-share floor | §2.41 |
| `min_n_eff` 1d / 2-3d / 1wk / 2wk | 30 / 45 / 60 / 80 | Overlap-adjusted sample floors | §2.41 |
| Peeking `k` | 2 | Consecutive-cycle correction | §2.41 |
| Purged-CV `embargo_gap` | 11 trading days | `h_max + 1`; strictly > 10 | §3.7 |
| `PBO` threshold | 0.50 | Do-not-promote line | §3.7 |
| `DSR` threshold | 1.0 | Improvement is plausibly noise below this | §2.41, §3.7 |
| MAD scale factor | 1.4826 | Consistent SD estimator for Gaussian | §1.2 |
| Winsorization ranks | 1st / 99th percentile | Cross-sectional feature clip | §1.2 |
| Winsorization fill/label | ±5 MAD | Slippage and label clip | §1.2 |
| IC alert healthy / degraded / flipped | > 0.04 / < 0.02 (3 windows) / < −0.03 | Family-horizon IC monitoring | §3.6 |
| SRPC alert | < 0.80 | Execution rank-reordering flag | §3.6 |

## Roman variables

| Name | Meaning | Section |
|---|---|---|
| `a_pi_g`, `b_pi_g` | Beta prior hyperparameters on inclusion probability | §1.6 |
| `A_t` | Active tradable set at t (fresh price, positive signal) | §2.21 |
| `b_i,t` | Source-conviction boost factor for candidate i at t | §2.21 |
| `base_side_cost_bps` | Aggregated one-side execution cost per §2.24 formula | §2.24 |
| `bar_dollar_volume` | Intraday bar dollar volume proxy (not full ADV) | §2.24 |
| `C_wait(a,t)` | Cash waiting value per action | §2.24 |
| `d_g` | Slab dimension parameter for group g | §1.6 |
| `data_quality_i` | Composite data-quality score for candidate i | §1.2, §1.7 |
| `decision_ref_price` | Reference mid at decision time for slippage | §2.11 |
| `edge_horizon_minutes` | = 390 for daily horizon; drives action-memory horizon | §2.24 |
| `effective_model_edge_bps` | `source_member_edge_bps × dynamic_friction_multiplier` | §2.23 |
| `EV_market`, `EV_limit(L)`, `EV_wait`, `EV_hold`, `EV_cash` | Comparison EVs for order-type routing | §2.14, §3.3 |
| `F(w)` | Full portfolio construction objective | §2.28 |
| `f(...)` weight function | Observation quality weight (form unspecified) | §1.7 |
| `G(w)` | Expected log-growth per weight vector | §2.1 |
| `G_p,t` | Rolling evidence score for source package p | §2.20 |
| `G_t` | Gross target exposure inherited from selected source | §2.21 |
| `H` | Horizon | §2.4 |
| `H_a(t)` | Hurdle for action a at t | §2.6, §2.18 |
| `I_t` | Information set at t | §2.32 |
| `indicator_contribution_j,i,t,h` | `E[beta_j,h | D_t-] × z_j,i,t` | §2.8 |
| `m_A`, `m_k` | Posterior mean (group A / group k) | §2.7 |
| `M_i(t)` | Membership role of candidate i at t (primary/secondary/retained/none) | §2.22 |
| `M_t` | Compact candidate slate after pruning | §3.14 |
| `model_edge_bps(i,t,h)` | Composite: `after_cost_edge + source_member_edge × dynamic_friction_multiplier` | §2.22a |
| `n_clean` | Count of clean resolved shadow rows | §2.41 |
| `n_eff` | Effective sample size under label autocorrelation | §2.41 |
| `n_eff_decay` | Effective sample under exponential decay | §2.41 |
| `notional` | `abs(target_delta_dollars)` | §2.24 |
| `omega_m` | Fractional OOS rank of best in-sample config m in PBO | §3.7 |
| `p_i,t` | Score-derived probability, `= clamp(score_i / 100, 0, 1)` | §1.2, §2.21 |
| `positive_edge_i_H` | Long-only positive-edge form of `mu_i_H` | §2.3 |
| `positive_share` | Fraction of clean shadow rows with positive delta | §2.41 |
| `P_fill(L)` | `clamp((L − bid_bps) / spread_bps, 0, 1)` | §2.25 |
| `q_i` | Confidence in `[0, 1]` | §2.5 |
| `Q(a, t)` | Full action-value equation | §2.32 |
| `quote_age_sessions` | `age_in_minutes / 390` | §2.24 |
| `raw_weight_i` | Practical Kelly-style position weight | §2.29 |
| `recent_fill_slippage_bps` | Exponentially decayed slippage feedback per ticker | §2.24 |
| `replacement_improvement_bps` | `EV(proposed) − EV(current)` for REPLACE/CANCEL | §2.25 |
| `s_floor_j` | Standardization variance floor for feature j | §1.2 |
| `s_i` | Score in `[0, 100]` for candidate i | §1.2, §2.21 |
| `size_ratio` | `notional / bar_dollar_volume` | §2.24 |
| `source_member_edge_bps(i,t)` | Provenance prior (460 / 348 / 200 / 0) | §2.22 |
| `spread_bps` | `10000 × (ask − bid) / mid` | §2.10 |
| `stale_quote_risk_bps` | Piecewise-linear staleness ramp on `quote_age_sessions` | §2.24 |
| `SRM_statistic(t)` | Sequential Ratio Monitor canary on rolling shadow stream | §3.6 |
| `SRPC(t)` | Signal Rank Preservation Coefficient | §3.6 |
| `TradeNowValue`, `WaitValue` | Fluid timing comparison | §2.34 |
| `U(W)` | Utility = `log(W)` | §2.1 |
| `V_0`, `V_A`, `V_k,k` | Posterior covariance objects | §2.7 |
| `volatility_bps` | Annualized realized vol in bps (= `volatility_pct × 10000 × sqrt(252)`) | §2.24 |
| `volatility_bps_corrected` | Roll bid-ask-bounce-corrected version | §2.10 |
| `w_n` | Per-observation regression weight (§1.7) | §1.7, §2.7 |
| `w_kelly` | Reference Kelly weight `mu / sigma^2` | §2.29 |
| `y_i,t,h` | Realized after-cost forward return label (research target) | §2.7 |
| `Y` | Empirical scaling factor in Bouchaud impact law (`Y ∈ [0.5, 1.5]`) | §2.12 |
| `z_j,i,t` | Causally standardized feature j on candidate i at t | §1.2 |
| `Z_n` | Auxiliary control features in grouped regression | §2.7 |

## Composite objects and named quantities

| Name | Definition | Section |
|---|---|---|
| `after_cost_edge_i,t,h` | Posterior expected edge net of cost, uncertainty, wait value | §2.8 |
| `available_buy_budget_t(a)` | `cash_trade_budget_t + sell_funded_cash_t(a)` | §2.39 |
| `carry_cost` | Sum of overnight, weekend, event gap, vol drag, capital lockup | §2.16 |
| `cash_trade_budget_t` | `max(0, cash_t − cash_floor_t)` | §2.39 |
| `concentration_cost` | `concentration_scale × Σ max(w_i − comfortable_weight_i, 0)^2` | §2.30 |
| `conservative_edge` | `mu_edge − zq × sigma_total` | §2.6 |
| `dynamic_source_action_value_bps` | Challenger's per-bucket action-value estimate | §2.42 |
| `effective_signal` | Freshness-blended signal | §1.3 |
| `entry_cost_bps`, `exit_cost_raw_bps`, `exit_cost_modeled_bps`, `reserved_future_exit_cost_bps` | Action-type-specific cost pieces | §2.24 |
| `family_abs_share_g,i,t,h` | Sum of `|contribution|` in family g, normalized | §2.8 |
| `feasible action set F_t` | Actions satisfying account/broker/market constraints | §3.1 |
| `freshness_weight` | `exp(−age / halflife)` | §1.3 |
| `gap_risk_bps` | `clamp((gap_days − 1) × 1.75, 0, 25)` | §2.24 |
| `group_reliability_g` | Composite reliability tracked per group | §1.5 |
| `gross_action_edge_bps` | Action-type-specific gross edge | §2.25 |
| `liquidity_capacity_risk_bps` | `min(120, 0.25 + 25 × sqrt(min(9, size_ratio)))` | §2.24 |
| `market_impact_bps` | `max(account_relative_size_cost, liquidity_relative_impact)` | §2.12 |
| `minimum_required_edge_bps` | `max(minimum_hurdle_bps, action_cost_bps + operational_risk_bps)` | §2.26 |
| `posterior_group_summary_g` | Cached tuple of posterior stats per group | §2.7 |
| `queue_delay_risk_bps` | `min(60, 0.8 + 12 × size_ratio + 4.5 × max(0, session_multiplier − 1))` | §2.24 |
| `RoundTripEV` | `holding_return − entry_cost − expected_exit_cost − carry_cost` | §2.15 |
| `self_test_certificate` | Boolean indicator over full replay window | §3.5 |
| `signed_share`, `abs_share` | Per-indicator normalized shares | §2.8 |
| `TotalCost(a)` | Sum of all action costs | §2.9 |
| `w_i,t` | Source-conviction target weight | §2.21 |

## Acronyms

| Acronym | Expansion | Section |
|---|---|---|
| ADV | Average Daily Volume | §2.12 |
| BUY / SELL / ROTATE / REPLACE / CANCEL | Action types in gateway | §2.25 |
| CSCV | Combinatorially Symmetric Cross-Validation | §3.7 |
| DSR | Deflated Sharpe Ratio | §2.41, §3.7 |
| EOD | End-of-day | §2.19 |
| EV | Expected Value | §2.13+ |
| FA / TA | Fundamental Analysis / Technical Analysis | §3.11 |
| IC | Information Coefficient | §3.6 |
| IEX | Investors Exchange (Alpaca free-tier feed) | §1.3, §2.24 |
| MAD | Median Absolute Deviation | §1.2 |
| MM | Market Maker | §2.24 |
| NAV | Net Asset Value | §2.29 |
| OI / OS | Output Intended / Output Observed (in `self_test_certificate`) | §3.5 |
| OOS | Out-of-Sample | §2.7, §3.7 |
| PBO | Probability of Backtest Overfitting | §3.7 |
| PDT | Pattern Day Trader (SEC rule) | §2.19, §3.1 |
| PFOF | Payment For Order Flow | §2.24 |
| SRM | Sequential Ratio Monitor | §3.6 |
| SRPC | Signal Rank Preservation Coefficient | §3.6 |
| TC | Transfer Coefficient (Grinold-Kahn) | §3.6 |
