# Part 3 — Cost Model (§2.9–§2.18)

## 2.9 Total cost decomposition (§2.9)

A paper return is not a real return; every cost converting paper signal to account equity is subtracted.

```
TotalCost(a) = EntryCost + ExitCost + SlippageCost + SpreadCost + MarketImpactCost
             + PartialFillCost + MissedFillCost + OpportunityCost + CarryCost
             + UncertaintyCost + OperationalCost
```

All terms in basis points or dollars of action notional.

## 2.10 Spread cost (§2.10)

Mid and spread:

```
mid_i        = (bid_i + ask_i) / 2
spread_bps_i = 10000 * (ask_i - bid_i) / mid_i
```

Marketable orders pay half-spread on either side:

```
buy_spread_cost_bps  ≈ spread_bps_i / 2
sell_spread_cost_bps ≈ spread_bps_i / 2
```

Limit order expected spread cost:

```
expected_spread_cost = fill_probability * limit_fill_spread_cost
                     + (1 - fill_probability) * missed_fill_cost
```

Use limit over marketable order only if: `expected_spread_savings > missed_fill_cost`.

### Roll bid-ask bounce correction

Consecutive closes alternate bid/ask, inducing spurious negative serial autocorrelation (Roll 1984). This inflates `Var(r_close)` and therefore `volatility_bps` used in `base_side_cost_bps` (§2.24). Correct before using bar returns for volatility:

```
Var(r_corrected)          = Var(r_close) - s^2 / 4
volatility_bps_corrected  = 10000 * sqrt(max(0, Var(r_corrected)) * bars_per_year)
                            (bars_per_year = 252 for daily bars)
```

where `s = spread_bps / 10000` (decimal spread). Use `Var(r_corrected)` in any downstream volatility markout. Uncorrected `Var(r_close)` biases cost model upward for wide-spread mid-caps.

## 2.11 Slippage cost (§2.11)

```
slippage_bps_buy  = 10000 * (fill_price - decision_ref_price) / decision_ref_price
slippage_bps_sell = 10000 * (decision_ref_price - fill_price) / decision_ref_price
```

Priority order for reference sample (most-specific reliable first):
1. same symbol + same side + same session
2. same symbol + same side
3. same liquidity bucket + same session
4. global session average
5. conservative default

If realized slippage worsens, required edge rises automatically via the `recent_fill_slippage_bps` term inside `base_side_cost_bps` (§2.24).

## 2.12 Market impact and size cost (§2.12)

Account-relative size cost proxy:

```
size_cost_bps = min(size_cost_cap, size_cost_slope * N / V)
```

`size_cost_cap` and `size_cost_slope` are (unspecified in spec — TODO; likely calibrated from realized fills).

**Liquidity-relative impact (Bouchaud et al. 2018; Kissell 2013):**

```
impact_cost_bps = SQRT_IMPACT_COEFF * sqrt(N / ADV)
SQRT_IMPACT_COEFF = 25    (calibrated; NOT 9)
```

Derivation: `Y * sigma_daily * 10000 * sqrt(Q/ADV)` bps with `Y ∈ [0.5, 1.5]` (Bouchaud ch. 12; Frazzini-Israel-Moskowitz 2018). At `sigma_daily = 1.5%` and unit participation, yields 75–225 bps → coefficient ≈ 25. Coefficient 9 underestimates by roughly an order of magnitude and must not be used.

Combined:

```
market_impact_bps = max(account_relative_size_cost_bps, liquidity_relative_impact_bps)
```

If full target size fails the edge test, test smaller sizes (partial positive-EV execution principle): `N*_i = argmax_N BuyEV_i(N)` subject to `min_order <= N <= target_notional_i`.

## 2.13 Partial fill cost (§2.13)

```
EV_order = P(full_fill) * EV(full_fill)
         + P(partial_fill) * EV(partial_fill)
         + P(no_fill) * EV(no_fill)

partial_fill_cost = EV(ideal_full_fill) - EV_order
```

Place trade only if: `EV_order > EV(best_alternative)`.

Partial fill risk increases with wide spread, low liquidity, large order, premarket/after-hours session, passive limit, volatile symbol, stale quote.

## 2.14 Missed fill cost (§2.14)

Expected adverse move when a limit does not fill:

```
missed_fill_cost = (1 - P_fill) * E_move_if_missed
```

Limit EV:

```
EV_limit(L) = P_fill(L) * (alpha - C_fill(L)) + (1 - P_fill(L)) * (-C_miss(L))
L*          = argmax_L EV_limit(L)
```

Market EV:

```
EV_market = alpha - spread_cost - slippage_cost - impact_cost
```

Use limit only if `EV_limit(L*) > EV_market`. Use market only if `EV_market > max(EV_limit, EV_wait, EV_hold, EV_cash) + hurdle`.

## 2.15 Entry and exit modeled together (§2.15) — RoundTripEV

A buy without expected exit cost is incomplete.

```
RoundTripEV = expected_holding_return - entry_cost - expected_exit_cost - carry_cost

expected_exit_cost = E[spread_cost_exit] + E[slippage_exit] + E[market_impact_exit] + E[missed_exit_cost]
```

Buy valid only if `RoundTripEV > required_edge`.

System must estimate at entry: expected exit session, spread, slippage, liquidity, time to exit, forced-overnight probability, weekend-carry probability.

## 2.16 Carry cost (§2.16)

Cost of being exposed while waiting.

```
carry_cost = overnight_gap_risk + weekend_news_risk + event_gap_risk
           + volatility_drag + capital_lockup_cost
```

Weekend carry:

```
carry_cost_weekend = weekend_risk_scale * exposure * vulnerability
```

Vulnerability inputs: low risk-off resilience, high beta, high volatility, weak liquidity, negative overhang, binary event uncertainty, large concentration.

`weekend_risk_scale` is (unspecified in spec — TODO).

Hold through a weekend only when expected return justifies it — not merely because the position already exists.

## 2.17 Opportunity cost (§2.17)

Every action competes against alternatives.

```
OpportunityCost(A) = max_B EV_after_cost(B) - EV_after_cost(A)
```

**Rotation condition:**

```
Rotate A -> B if:
    EV_after_cost(new) - EV_after_cost(old) > sell_cost_old + buy_cost_new + transition_uncertainty + hurdle
```

**Exit to cash condition:**

```
Sell to cash if:
    EV_after_cost(cash) - EV_after_cost(hold) > sell_cost + rebound_risk + hurdle
```

**Cash EV includes option value of waiting:**

```
EV_cash = risk_free_return + option_value_of_waiting + avoided_bad_trade_cost
```

The `risk_free_return` component is realized concretely as `cash_waiting_value_bps` (§2.24).

## 2.18 Dynamic required edge hurdle (§2.18)

Trades are not placed merely because gross alpha is positive.

```
hurdle = model_error_buffer + quote_staleness_buffer
       + opportunity_cost_buffer + operational_risk_buffer
```

```
Trade only if:
    EV_after_cost(action) - EV_after_cost(best_alternative) > hurdle
```

Hurdle rises with wider spreads, stale quotes, lower liquidity, higher volatility, lower model confidence, worse recent slippage, larger trade, extended-hours session, holding crosses close/weekend.

Hurdle falls with fresh quotes, excellent liquidity, high confidence, good recent fills, large drift, unusually strong opportunity, current holding with high negative EV.

Note: the applied `minimum_hurdle_bps = 0` (§2.19) is the governance floor, not the dynamic hurdle. The dynamic hurdle is enforced through the sum of concrete cost terms in `action_cost_bps` and `operational_risk_bps` at the gate (§2.26).
