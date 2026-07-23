// costs/model.cpp — full cost decomposition per §2.24.
// see docs/knowledge-base/part4-gateway-execution.md §2.24
// see docs/knowledge-base/part3-costs.md §2.9–§2.18

#include "juniauto/costs/model.hpp"

#include <algorithm>
#include <cmath>

#include "juniauto/utils/math.hpp"

namespace juniauto::costs {

// ---- Base one-side cost (§2.24) ----
//
//  base_side_cost_bps = (0.5*spread_bps
//                       + 0.10*min(200, volatility_bps)
//                       + liquidity_capacity_risk_bps
//                       + stale_quote_risk_bps
//                       + recent_fill_slippage_bps)
//                     * session_multiplier
//                     + gap_risk_bps
//
// see docs/knowledge-base/part4-gateway-execution.md §2.24
double base_side_cost_bps(const MarketState& s, const SlippageStats& slip, const CostConfig& cfg) {

    // -- Half-spread term (§2.10) --
    const double half_spread = 0.5 * std::min(s.spread_bps, cfg.spread_bps_cap);

    // -- Volatility markout allowance (§2.24) --
    // 0.10 * min(200, vol_bps).  cap = 200 binds for nearly all liquid equities.
    const double vol_term = 0.10 * std::min(cfg.volatility_bps_cap, s.volatility_bps);

    // -- Liquidity capacity risk (§2.24) --
    // size_ratio = notional / bar_dollar_volume; handled by the caller populating MarketState.
    // Here bar_dollar_volume is stored in MarketState.bar_dollar_volume.
    // We reconstruct size_ratio conceptually: caller passes in a pre-computed MarketState.
    // The MarketState doesn't carry size_ratio directly; we compute from bar_dollar_volume.
    // Per §2.24 normalization: size_ratio = 1 if volume missing and notional > 0; else 0.
    // Since model.cpp doesn't have notional here (it's in Order), we use a conservative
    // size_ratio = 0 when bar_dollar_volume == 0 (liquidity floor applied below).
    // The full size-ratio-aware liquidity term is computed in compute_cost().
    // For base_side_cost_bps used standalone, assume size_ratio reflects the passed state.
    // We expose a helper inline below; public API computes with full order context.
    const double liquidity_bps = cfg.liq_min_bps;  // minimum floor; caller overrides via compute_cost

    // -- Stale-quote risk (§2.24, IEX recalibration) --
    // Thresholds in sessions (IEX 15-min = 0.038 sessions < 0.5 => 0 bps by design).
    // see docs/knowledge-base/part4-gateway-execution.md §2.24 stale-quote table
    double stale_bps = 0.0;
    const double age = s.quote_age_sessions;
    if (age > cfg.stale_band2_threshold) {
        // min(80, 4 + (age - 1.0) * 40)  — §2.24
        stale_bps = std::min(cfg.stale_band2_cap_bps,
                             4.0 + (age - cfg.stale_band2_threshold) * cfg.stale_band2_slope);
    } else if (age > cfg.stale_band1_threshold) {
        // linear ramp: min(20, (age - 0.5) * 40)  — §2.24
        stale_bps = std::min(cfg.stale_band1_cap_bps,
                             (age - cfg.stale_band1_threshold) * cfg.stale_band2_slope);
    }
    // else: 0 (IEX current-session quotes fall here)

    // -- Recent fill slippage feedback (§2.24) --
    // Caller populates slip.recent_fill_slippage_bps via exp_decay aggregation.
    // Clamped to [0, 50] per §2.24.
    const double slippage_bps = math::clamp(slip.recent_fill_slippage_bps, 0.0, cfg.slippage_cap_bps);

    // -- Gap risk (§2.24): clamp((gap_days - 1) * 1.75, 0, 25) --
    // see docs/knowledge-base/part4-gateway-execution.md §2.24
    const double gap_bps = math::clamp(
        (static_cast<double>(s.gap_days_to_next_session) - 1.0) * cfg.gap_slope,
        0.0, cfg.gap_cap_bps);

    // -- Assembly --
    const double inner = half_spread + vol_term + liquidity_bps + stale_bps + slippage_bps;
    return inner * s.session_multiplier + gap_bps;
}

// ---- Full cost breakdown for an action ----
// see docs/knowledge-base/part4-gateway-execution.md §2.24–§2.25
CostBreakdown compute_cost(
    const Order& order,
    const MarketState& state,
    const SlippageStats& slip,
    const CostConfig& cfg,
    double model_edge_bps)
{
    CostBreakdown bd;

    // -- Size ratio (§2.24 normalization) --
    double size_ratio = 0.0;
    const double notional_abs = std::abs(order.notional);
    if (state.bar_dollar_volume > 0.0) {
        size_ratio = notional_abs / state.bar_dollar_volume;
    } else if (notional_abs > 0.0) {
        size_ratio = 1.0;  // missing volume = worst-case per §2.24
    }

    // -- Liquidity capacity risk (§2.24) --
    // min(120, 0.25 + 25 * sqrt(min(9, size_ratio)))
    const double liq_bps = std::min(
        cfg.liq_cap_bps,
        cfg.liq_min_bps + cfg.liq_slope * std::sqrt(std::min(9.0, size_ratio)));
    // If volume missing and notional > 0: max with 35 bps floor.
    const double liquidity_capacity_risk =
        (state.bar_dollar_volume <= 0.0 && notional_abs > 0.0)
            ? std::max(liq_bps, cfg.liq_missing_volume_floor_bps)
            : liq_bps;

    // -- Stale-quote risk (§2.24) --
    double stale_bps = 0.0;
    const double age = state.quote_age_sessions;
    if (age > cfg.stale_band2_threshold) {
        stale_bps = std::min(cfg.stale_band2_cap_bps,
                             4.0 + (age - cfg.stale_band2_threshold) * cfg.stale_band2_slope);
    } else if (age > cfg.stale_band1_threshold) {
        stale_bps = std::min(cfg.stale_band1_cap_bps,
                             (age - cfg.stale_band1_threshold) * cfg.stale_band2_slope);
    }

    // -- Recent fill slippage (§2.24, clamped [0, 50]) --
    const double slippage_bps = math::clamp(slip.recent_fill_slippage_bps, 0.0, cfg.slippage_cap_bps);

    // -- Gap risk (§2.24) --
    const double gap_bps = math::clamp(
        (static_cast<double>(state.gap_days_to_next_session) - 1.0) * cfg.gap_slope,
        0.0, cfg.gap_cap_bps);

    // -- Base one-side cost (§2.24) --
    const double half_spread = 0.5 * std::min(state.spread_bps, cfg.spread_bps_cap);
    const double vol_term    = 0.10 * std::min(cfg.volatility_bps_cap, state.volatility_bps);
    const double inner = half_spread + vol_term + liquidity_capacity_risk + stale_bps + slippage_bps;
    const double base_side = inner * state.session_multiplier + gap_bps;
    bd.base_side_bps = base_side;

    // -- Entry cost by action type (§2.24) --
    // BUY, ROTATE, REPLACE pay base_side; others pay 0.
    // Action type is not in this function's signature; we apply a universal entry cost
    // and let the gateway decision layer select by action.  The header contract says
    // compute_cost fills CostBreakdown for the proposed action; we model a BUY here
    // and expose base_side so the gateway can combine per §2.25.
    // Entry = base_side (the gateway applies per-action selection).
    bd.entry_bps = base_side;

    // -- Exit reserved (§2.24) --
    // BUY: exit_reserved = buy_exit_haircut * buy_exit_future_factor * base_side
    //    = 0.65 * 0.85 * base_side = 0.5525 * base_side   (exact product, §2.24)
    // see docs/knowledge-base/part4-gateway-execution.md §2.24 (spec quirk confirmed)
    bd.exit_reserved_bps = cfg.buy_exit_haircut * cfg.buy_exit_future_factor * base_side;

    // -- Queue delay risk (§2.24) --
    // min(60, 0.8 + 12*size_ratio + 4.5*max(0, session_multiplier - 1))
    // see docs/knowledge-base/part4-gateway-execution.md §2.24
    bd.queue_delay_bps = std::min(
        cfg.queue_cap_bps,
        cfg.queue_min_bps
            + cfg.queue_slope_size * size_ratio
            + cfg.queue_slope_session * std::max(0.0, state.session_multiplier - 1.0));

    // -- Cancel/replace risk (§2.24) --
    // min(60, api_budget + lost_queue_priority + max(0, session_multiplier - 1))
    // api_budget = 2, lost_queue_priority = 1 (Alpaca PFOF, no exchange queue).
    // see docs/knowledge-base/part4-gateway-execution.md §2.24
    bd.cancel_replace_bps = std::min(
        cfg.cxl_cap_bps,
        cfg.cxl_api_budget_bps
            + cfg.cxl_lost_priority_bps
            + std::max(0.0, state.session_multiplier - 1.0));

    // -- Action memory cost (§2.24) --
    // horizon = predicted_holding_seconds if > 0, else 23400 (390*60, one trading day).
    // see docs/knowledge-base/part4-gateway-execution.md §2.24, §2.27
    const double horizon_seconds =
        (order.predicted_holding_seconds > 0.0)
            ? order.predicted_holding_seconds
            : cfg.memory_horizon_seconds;  // 23400 = 390 * 60

    // No recent opposite fill available via SlippageStats; use fallback round-trip cost.
    // When a real recent_opposite_fill_age_seconds is available the caller should compute
    // this externally and pass in a pre-adjusted slippage; the fallback is 60 bps.
    // see docs/knowledge-base/part4-gateway-execution.md §2.24
    const double fallback_decay = std::exp(-0.0 / std::max(1.0, horizon_seconds));
    bd.action_memory_bps = cfg.memory_fallback_round_trip_bps * fallback_decay;

    // -- Cash waiting value (§2.24) --
    // cash_waiting_value_bps = (r_cash / 252) * 10000
    // r_cash defaults to fed_funds - 50 bps.  We use env-provided fallback: the
    // CostConfig does not carry r_cash; use a conservative static 4.5% APY fallback.
    // Callers set this by overriding the breakdown field post-call if they have live data.
    // see docs/knowledge-base/part4-gateway-execution.md §2.24
    constexpr double R_CASH_FALLBACK = 0.045;  // 4.5% (Fed Funds ~5% minus 50 bps haircut)
    constexpr double TRADING_DAYS_PER_YEAR = 252.0;
    bd.cash_waiting_value_bps = (R_CASH_FALLBACK / TRADING_DAYS_PER_YEAR) * 10000.0;

    // -- Operational risk (§2.26 cold-start default = 10 bps) --
    bd.operational_bps = cfg.op_base_bps;

    // -- Opportunity cost (placeholder; model_edge_bps exposed for caller) --
    // Full opportunity cost requires comparing against best alternative — not computed here.
    bd.opportunity_bps = 0.0;

    // Uncertainty cost: excluded from this breakdown (part of sigma_total in the ridge).
    bd.uncertainty_bps = 0.0;

    return bd;
}

}  // namespace juniauto::costs
