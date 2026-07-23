// gateway/decision.hpp — the composite-edge gate (§2.22a–§2.26).
//
// Consumes: posterior edge (Bayesian) + provenance role + dynamic friction + costs.
// Emits: a typed action (BUY / SELL / ROTATE / REPLACE / CANCEL / HOLD).

#pragma once

#include <string>

#include "juniauto/costs/model.hpp"

namespace juniauto::gateway {

enum class Role : int {
    Primary = 0,      // (§2.22) 460 bps prior
    Secondary = 1,    // (§2.22) 348 bps
    Retained = 2,     // (§2.22) 200 bps
    None = 3,
};

enum class ActionType : int {
    Buy = 0,
    Sell = 1,
    Rotate = 2,
    Replace = 3,
    Cancel = 4,
    Hold = 5,
};

struct GatewayConfig {
    // Provenance role edges (§2.22).
    double primary_bps = 460.0;
    double secondary_bps = 348.0;
    double retained_bps = 200.0;
    // Dynamic friction seeds (§2.23) — Bayesian mean of the adaptive multiplier surface.
    double friction_seed_primary = 0.30;
    double friction_seed_secondary = 0.30;
    double friction_seed_retained = 0.30;
    double friction_floor = 0.05;
    double friction_ceiling = 1.00;
    // Exit reserve fraction (§2.19).
    double exit_reserve = 1.0;
    // Minimum hurdle (§2.19).
    double minimum_hurdle_bps = 0.0;
};

struct ActionEvaluation {
    std::string symbol;
    ActionType action = ActionType::Hold;
    double gross_edge_bps = 0.0;
    double total_cost_bps = 0.0;
    double net_edge_bps = 0.0;
    double model_edge_bps = 0.0;    // composite (§2.22a)
    double friction_multiplier = 0.0;
    Role role = Role::None;
    bool executes() const noexcept { return net_edge_bps > 0.0; }
};

// Composite model edge (§2.22a):
//     model_edge = after_cost_edge + membership_bps * friction_multiplier
double composite_edge(double after_cost_edge_bps,
                      double membership_bps,
                      double friction_multiplier) noexcept;

// Provenance edge lookup (§2.22).
double membership_edge_bps(Role role, const GatewayConfig& cfg) noexcept;

// One-shot evaluation of a candidate (§2.24–§2.26).
ActionEvaluation evaluate(
    const std::string& symbol,
    Role role,
    double after_cost_edge_bps,   // §2.8
    double friction_multiplier,   // §2.23
    const costs::MarketState& state,
    const costs::SlippageStats& slippage,
    const costs::CostConfig& cost_cfg,
    const GatewayConfig& gw_cfg,
    ActionType proposed
);

}  // namespace juniauto::gateway
