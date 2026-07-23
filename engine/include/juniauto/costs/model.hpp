// costs/model.hpp — total cost decomposition (§2.9–§2.18) and its inputs.
//
// Every value in this header is in basis points of action notional unless stated.

#pragma once

#include <string>

namespace juniauto::costs {

struct MarketState {
    double mid_price = 0.0;
    double spread_bps = 0.0;
    double volatility_bps = 0.0;         // annualized realized vol in bps (Roll-corrected)
    double bar_dollar_volume = 0.0;      // used in size_ratio
    double adv_dollar = 0.0;             // 20d ADV in dollars, used in impact_cost
    double quote_age_sessions = 0.0;     // (§2.24)
    int gap_days_to_next_session = 0;    // (§2.24 gap_risk)
    double session_multiplier = 1.0;     // (§2.24)
    double adverse_selection_share = 0.35; // (§2.24 Glosten-Milgrom)
};

struct Order {
    std::string symbol;
    double notional = 0.0;               // signed for direction
    double predicted_holding_seconds = 0.0;
    bool has_open_order = false;
};

struct SlippageStats {
    // Recent per-fill slippages (k=1 most recent) — exponentially decayed in the model.
    // Empty means cold-start; caller substitutes a universe-level fallback.
    double recent_fill_slippage_bps = 0.0;
};

struct CostBreakdown {
    double base_side_bps = 0.0;
    double entry_bps = 0.0;
    double exit_reserved_bps = 0.0;
    double queue_delay_bps = 0.0;
    double cancel_replace_bps = 0.0;
    double action_memory_bps = 0.0;
    double cash_waiting_value_bps = 0.0;
    double operational_bps = 0.0;
    double uncertainty_bps = 0.0;
    double opportunity_bps = 0.0;

    double total() const noexcept {
        return entry_bps + exit_reserved_bps + queue_delay_bps + cancel_replace_bps
             + action_memory_bps + operational_bps + uncertainty_bps + opportunity_bps;
    }
};

// Config knobs corresponding to `config/production.yaml -> costs:`.
struct CostConfig {
    double spread_bps_cap = 1000.0;
    double volatility_bps_cap = 200.0;
    double sqrt_impact_coeff = 25.0;         // (§2.12)
    // liquidity_capacity_risk_bps = min(cap, min_bps + slope * sqrt(min(9, size_ratio)))
    double liq_min_bps = 0.25;
    double liq_slope = 25.0;
    double liq_cap_bps = 120.0;
    double liq_missing_volume_floor_bps = 35.0;
    // stale-quote piecewise bands (recalibrated to sessions, §2.24)
    double stale_band1_threshold = 0.5;
    double stale_band1_cap_bps = 20.0;
    double stale_band2_threshold = 1.0;
    double stale_band2_cap_bps = 80.0;
    double stale_band2_slope = 40.0;
    // gap
    double gap_slope = 1.75;
    double gap_cap_bps = 25.0;
    // queue delay
    double queue_min_bps = 0.8;
    double queue_slope_size = 12.0;
    double queue_slope_session = 4.5;
    double queue_cap_bps = 60.0;
    // cancel/replace
    double cxl_api_budget_bps = 2.0;
    double cxl_lost_priority_bps = 1.0;
    double cxl_cap_bps = 60.0;
    // action memory
    double memory_fallback_round_trip_bps = 60.0;
    double memory_horizon_seconds = 23400.0;
    // buy-side exit haircut
    double buy_exit_haircut = 0.65;
    double buy_exit_future_factor = 0.85;
    // slippage aggregation
    double slippage_per_fill_decay = 0.75;
    double slippage_cap_bps = 50.0;
    // operational
    double op_base_bps = 5.0;
    double op_cap_bps = 40.0;
};

// Compute the base one-side cost (§2.24):
//     (0.5*spread + 0.10*min(200, vol) + liquidity + stale + recent_fill_slippage)
//   * session_multiplier + gap_risk
double base_side_cost_bps(const MarketState& s, const SlippageStats& slip, const CostConfig& cfg);

// Full breakdown for an action.
CostBreakdown compute_cost(
    const Order& order,
    const MarketState& state,
    const SlippageStats& slip,
    const CostConfig& cfg,
    double model_edge_bps  // used for opportunity comparison
);

}  // namespace juniauto::costs
