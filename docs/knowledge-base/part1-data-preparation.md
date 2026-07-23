# Part 1 — Data Preparation (§1.1–§1.8)

## 1.1 Candidate universe (§1.1)

A candidate belongs in the universe only if it can plausibly improve after-cost expected return and is reachable through the broker route.

Minimum per-candidate requirements:
- valid tradable instrument mapping
- current or sufficiently fresh price
- bid/ask available at execution time
- enough data to estimate expected return and cost
- enough liquidity to make an order realistic

Feature vector per candidate `i` at time `t`:

```
x_i,t = [technical, fundamental, event, semantic/context, liquidity, risk, execution, freshness]
```

## 1.2 Normalization and causal standardization (§1.2)

Bounded score form: `s_i ∈ [0, 100]` (0 = strongly unfavorable, 50 = neutral, 100 = strongly favorable). Probability form: `p_i = clamp(score_i / 100, 0, 1)`.

Causal standardization prevents future data from changing a past score:

```
z_j,i,t = (x_j,i,t - mu_j,t-) / max(s_j,t-, s_floor_j)
```

`mu_j,t-` and `s_j,t-` use data strictly before time `t`. Missing data receives no optimistic default; instead track `data_quality_i` and `missing_severity_i` and apply uncertainty penalties later.

### Winsorization

Applied before percentile ranking and before ridge regression.

**Cross-sectional percentile clip (feature ranks):**

```
winsorized_x_j,i,t = clamp(x_j,i,t, percentile(x_j,t, 0.01), percentile(x_j,t, 0.99))
```

**MAD-based clip (for `recent_fill_slippage_bps` and `realized_return_bps` labels):**

```
MAD_j = median(|x_j,i,t - median(x_j,t)|)
winsorized_x_j,i,t = clamp(x_j,i,t, median - 5*MAD_j, median + 5*MAD_j)
```

The ±5-MAD band preserves genuine large moves (earnings gaps) while rejecting data-error outliers that would shift ridge coefficients ~30% in a 500-sample regime.

### MAD unit-scale standardization (before ridge)

Ridge `lambda * ||beta||^2` requires scale invariance; features with different variance scales cause ridge to under/over-penalize (Hoerl & Kennard 1970). Standardize each feature to unit MAD scale before any ridge update:

```
x_standardized = (x - median_j) / (1.4826 * MAD_j)
```

The `1.4826` factor makes MAD a consistent estimator of Gaussian SD.

## 1.3 Signal freshness (§1.3)

Stale data is not neutral — it creates measurable uncertainty.

```
age = current_time - observation_time
freshness_weight = exp(-age / halflife)
effective_signal = freshness_weight * raw_signal + (1 - freshness_weight) * prior_signal
```

Conservative default: `prior_signal = 0` (neutral).

> **IEX/PDT adaptation:** With a 15-minute delayed feed and 1-trading-day minimum hold, halflives are recalibrated to **trading days**, not seconds or minutes. Quote-level data from the current day's closing session is "fresh"; prior-session data decays per family:
> - Fundamental quality: several **weeks** (business quality changes slowly)
> - Technical momentum: several **trading days** (degrades over days, not hours)
> - Event / catalyst: **hours to trading days** (evaluated in trading days, not minutes)
> - Liquidity / microstructure: **1 trading day** (session-level liquidity resets each session)
>
> Formula is unchanged; only the operative halflife units change.

## 1.4 Signal families (§1.4)

Six evidence families enter the Bayesian hierarchy as separate groups because update speed, reliability, and failure modes differ.

**4.1 Technical Momentum** — near-term continuation probability and payoff size. Evaluated on daily bars only; intraday structure is excluded because a 15-minute delayed feed cannot act on it.
- relative strength, trend slope, breakout strength, distance from MAs, price acceleration, volume confirmation, support/resistance behavior, drawdown/rebound structure, realized volatility

**4.2 Fundamental Quality** — durable strength vs noisy price motion (useful over multi-day horizons).
- earnings quality, revenue growth, profitability, balance sheet strength, valuation quality, analyst revision direction, institutional sponsorship, industry leadership, liquidity-adjusted quality

**4.3 Event and Catalyst** — when new information changed expected return. Catalyst freshness is evaluated in **trading days**, not minutes, consistent with the 1-day minimum hold.
- earnings surprises, guidance changes, news catalysts, sector catalysts, company-specific announcements, regulatory events, product launches, analyst up/downgrades, unusual volume around event, freshness of catalyst, confirmation by price and volume
- Penalize for: staleness, ambiguity, binary outcome risk, negative overhang, already-priced-in movement, wide spreads during news shocks

**4.4 Semantic and Contextual** — explains why a move should continue.
- company-specific vs generic catalyst, sentiment direction, event confirming broader theme, sector context, freshness of reason, unresolved negative overhang
- Missing / generic / stale context reduces confidence rather than inventing edge

**4.5 Liquidity and Execution Quality** — whether paper return survives costs.
- bid, ask, spread, quote freshness, volume, relative volume, dollar volume, market depth proxy, historical slippage, participation rate, fill probability, order-size sensitivity

**4.6 Risk and Crowding** — protects compounded return from volatility drag and adverse path dependence.
- realized volatility, beta, gap risk, weekend/news risk, crowding, short squeeze / unwind risk, correlation to existing holdings, drawdown sensitivity, regime sensitivity, sector concentration
- Volatility is acceptable when compensated by return per unit of after-cost risk

## 1.5 Feature groups for Bayesian shrinkage (§1.5)

Related features bundled into evidence families to prevent double-counting.

```
X_i,t,g = { z_j,i,t : j belongs to group g }
```

Group examples:
- TA momentum (trend, breakout, relative strength, torque)
- TA chart structure (support, extension, oversold, range)
- Volatility / range (realized volatility, ATR, volume shock)
- Liquidity / microstructure (spread, quote age, dollar volume, partial-fill risk)
- Fundamental quality (growth-quality, earnings proxy, liquidity quality)
- Source / provenance role (primary, secondary, retained membership)
- Event / regime context (catalyst, sector stabilization, session bucket)
- Portfolio / account state (current weight, target delta, cash, open orders, recent fills)
- Realized execution telemetry (recent slippage, fill quality, queue delay)

Group reliability summary:

```
group_reliability_g = f(posterior_mean_g, posterior_sd_g, out_of_sample_hit_rate_g,
                       cost_adjusted_forward_edge_g, sample_count_g, regime_condition_g)
```

## 1.6 Grouped Bayesian prior (§1.6) — spike-and-slab

Feature groups receive spike-and-slab shrinkage priors so weak groups are suppressed as a group, not feature by feature.

```
beta_g | gamma_g, tau_g ~ gamma_g * N(0, sigma^2 * tau_g^2 * I_g) + (1 - gamma_g) * delta_0(beta_g)
gamma_g | pi_g ~ Bernoulli(pi_g)
pi_g ~ Beta(a_pi_g, b_pi_g)
tau_g^2 | lambda_g ~ Gamma((d_g + 1)/2, lambda_g^2 / 2)
sigma^2 ~ Inv-Gamma(a_sigma, b_sigma)
```

Terms:
- `gamma_g`: whether group g is allowed to matter
- `delta_0`: point mass at zero (group excluded)
- `tau_g`: group-specific slab scale (size of included coefficients)
- `lambda_g`: group shrinkage strength
- `pi_g`: prior inclusion probability

Prior hyperparameters `a_pi_g, b_pi_g, d_g, lambda_g, a_sigma, b_sigma` are (unspecified in spec — TODO).

## 1.7 Observation quality weighting (§1.7)

Cleaner rows get higher weight in Bayesian regression.

```
w_n = f(quote_freshness_n, spread_quality_n, fill_resolution_n,
        missing_data_flags_n, order_state_cleanliness_n)
```

A fully resolved regular-session action with quotes from the current session's delayed feed is high weight. Rows with prior-session-close proxy quotes, missing spread, partial unresolved fills, or uncertain account state get low weight.

> **IEX/PDT adaptation:** All IEX-feed data is inherently >= 15 minutes old at receipt. "Quality" here means originating from today's session rather than yesterday's close used as a proxy. The 15-minute floor is not a downgrade signal — it is the baseline. Downgrade signals fire when data is from a prior session or unresolved.

Explicit functional form for `f(...)` is (unspecified in spec — TODO).

## 1.8 Streaming feature construction cost (§1.8)

Feature computation is near-linear.

```
C_feature(t) = Theta(nnz(X_t)) <= Theta(n * p)
```

Lower bound: `C_feature(t) = Omega(n)` (each security inspected at least once).
