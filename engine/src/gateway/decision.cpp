// gateway/decision.cpp — composite edge, membership edge, and evaluate().
// see docs/knowledge-base/part4-gateway-execution.md §2.22a, §2.22, §2.24–§2.26

#include "juniauto/gateway/decision.hpp"

#include <algorithm>
#include <cmath>

#include "juniauto/costs/model.hpp"

namespace juniauto::gateway {

// Composite model edge (§2.22a):
//   model_edge = after_cost_edge + membership_bps * friction_multiplier
// see docs/knowledge-base/part4-gateway-execution.md §2.22a
double composite_edge(double after_cost_edge_bps,
                      double membership_bps,
                      double friction_multiplier) noexcept {
    return after_cost_edge_bps + membership_bps * friction_multiplier;
}

// Provenance edge by role (§2.22):
//   primary = 460, secondary = 348, retained = 200, none = 0.
// see docs/knowledge-base/part4-gateway-execution.md §2.22
double membership_edge_bps(Role role, const GatewayConfig& cfg) noexcept {
    switch (role) {
        case Role::Primary:   return cfg.primary_bps;    // 460 bps
        case Role::Secondary: return cfg.secondary_bps;  // 348 bps
        case Role::Retained:  return cfg.retained_bps;   // 200 bps
        case Role::None:      return 0.0;
    }
    return 0.0;
}

// One-shot evaluation (§2.24–§2.26).
// Gross edge depends on action type per §2.25; net edge = gross - cost.
// see docs/knowledge-base/part4-gateway-execution.md §2.25–§2.26
ActionEvaluation evaluate(
    const std::string& symbol,
    Role role,
    double after_cost_edge_bps,
    double friction_multiplier,
    const costs::MarketState& state,
    const costs::SlippageStats& slippage,
    const costs::CostConfig& cost_cfg,
    const GatewayConfig& gw_cfg,
    ActionType proposed)
{
    ActionEvaluation ev;
    ev.symbol             = symbol;
    ev.action             = proposed;
    ev.role               = role;
    ev.friction_multiplier = friction_multiplier;

    // Composite model edge (§2.22a).
    const double mem_bps = membership_edge_bps(role, gw_cfg);
    ev.model_edge_bps = composite_edge(after_cost_edge_bps, mem_bps, friction_multiplier);

    // Cost breakdown (uses a generic Order with zero notional since the gateway
    // receives the model edge directly; size-ratio-dependent costs reflect pre-computed state).
    costs::Order ord;
    ord.symbol = symbol;
    ord.notional = 0.0;  // gateway-level evaluation; sizing handled by portfolio layer
    costs::CostBreakdown cbd = costs::compute_cost(ord, state, slippage, cost_cfg, ev.model_edge_bps);

    // Cash waiting value from the breakdown.
    const double cash_wait = cbd.cash_waiting_value_bps;

    // Gross action edge by type (§2.25).
    // see docs/knowledge-base/part4-gateway-execution.md §2.25
    double gross_edge = 0.0;
    double total_cost = 0.0;

    switch (proposed) {
        case ActionType::Buy:
            // gross = model_edge - cash_waiting_value  (§2.25)
            gross_edge = ev.model_edge_bps - cash_wait;
            // cost = entry + exit_reserved + queue_delay + action_memory
            total_cost = cbd.entry_bps
                       + cbd.exit_reserved_bps
                       + cbd.queue_delay_bps
                       + cbd.action_memory_bps;
            break;

        case ActionType::Sell:
            // gross = cash_waiting_value - model_edge  (§2.25)
            gross_edge = cash_wait - ev.model_edge_bps;
            // cost = exit_reserved + queue_delay + action_memory
            total_cost = cbd.exit_reserved_bps
                       + cbd.queue_delay_bps
                       + cbd.action_memory_bps;
            break;

        case ActionType::Rotate:
            // gross = model_edge(new) - model_edge(old) - cash_waiting_value
            // At the gateway level model_edge_bps already represents the improvement;
            // the caller must compute the net edge differential before calling evaluate().
            gross_edge = ev.model_edge_bps - cash_wait;
            total_cost = cbd.entry_bps
                       + cbd.exit_reserved_bps
                       + cbd.queue_delay_bps
                       + cbd.action_memory_bps;
            break;

        case ActionType::Replace:
            // gross = replacement_improvement_bps (passed as after_cost_edge_bps by caller)
            gross_edge = after_cost_edge_bps;
            total_cost = cbd.cancel_replace_bps
                       + cbd.queue_delay_bps
                       + cbd.action_memory_bps;
            break;

        case ActionType::Cancel:
            gross_edge = after_cost_edge_bps;
            total_cost = cbd.cancel_replace_bps;
            break;

        case ActionType::Hold:
            gross_edge = 0.0;
            total_cost = 0.0;
            break;
    }

    // Operational risk (§2.26 cold-start = 5 bps; full formula applied externally).
    total_cost += cbd.operational_bps;

    ev.gross_edge_bps = gross_edge;
    ev.total_cost_bps = total_cost;
    ev.net_edge_bps   = gross_edge - total_cost;

    // Execute iff net_edge > minimum_hurdle_bps (§2.26 applied value = 0).
    // see docs/knowledge-base/part4-gateway-execution.md §2.26
    // executes() checks net_edge_bps > 0.0 per header definition.

    return ev;
}

}  // namespace juniauto::gateway
