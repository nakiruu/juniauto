// test_gateway.cpp — unit tests for the gateway composite edge.
// Asserts: BUY with role=Primary, friction=0.30 yields expected composite edge.
// see docs/knowledge-base/part4-gateway-execution.md §2.22a, §2.23

#include <cassert>
#include <cmath>
#include <cstdio>

#include "juniauto/gateway/decision.hpp"
#include "juniauto/costs/model.hpp"

static bool approx_eq(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) <= tol;
}

int main() {
    juniauto::gateway::GatewayConfig gw;  // defaults: primary=460, friction_seed=0.30
    const juniauto::costs::CostConfig cost_cfg;

    // ---- Test 1: composite_edge formula (§2.22a) ----
    // model_edge = after_cost_edge + membership_bps * friction_multiplier
    // Primary: membership = 460, friction seed = 0.30 => membership_contribution = 138 bps.
    const double after_cost = 50.0;
    const double friction   = 0.30;
    const double mem_bps    = juniauto::gateway::membership_edge_bps(
        juniauto::gateway::Role::Primary, gw);  // expect 460

    assert(approx_eq(mem_bps, 460.0, 1e-12) && "primary membership must be 460 bps");

    const double comp = juniauto::gateway::composite_edge(after_cost, mem_bps, friction);
    // Expected: 50 + 460 * 0.30 = 50 + 138 = 188 bps.
    assert(approx_eq(comp, 188.0, 1e-9) &&
           "composite_edge for primary+0.30 friction must be 188 bps");

    // ---- Test 2: evaluate() produces net_edge > 0 for a strongly positive BUY ----
    juniauto::costs::MarketState s;
    s.spread_bps           = 5.0;
    s.volatility_bps       = 80.0;
    s.bar_dollar_volume    = 2'000'000.0;
    s.quote_age_sessions   = 0.038;  // IEX floor
    s.gap_days_to_next_session = 1;
    s.session_multiplier   = 1.0;
    s.adverse_selection_share = 0.35;

    juniauto::costs::SlippageStats slip;
    slip.recent_fill_slippage_bps = 3.0;

    const juniauto::gateway::ActionEvaluation ev = juniauto::gateway::evaluate(
        "AAPL",
        juniauto::gateway::Role::Primary,
        /*after_cost_edge_bps=*/after_cost,
        /*friction_multiplier=*/friction,
        s, slip, cost_cfg, gw,
        juniauto::gateway::ActionType::Buy);

    // model_edge should equal composite_edge.
    assert(approx_eq(ev.model_edge_bps, comp, 1e-9) &&
           "evaluate() model_edge_bps must match composite_edge()");

    // net_edge = gross - total_cost; must be positive (executes() == true) for 188 bps edge.
    assert(ev.executes() && "strong primary BUY with 188 bps model edge must execute");

    // ---- Test 3: Role::None produces zero membership edge ----
    const double none_mem = juniauto::gateway::membership_edge_bps(
        juniauto::gateway::Role::None, gw);
    assert(approx_eq(none_mem, 0.0, 1e-12) && "Role::None membership must be 0");

    std::puts("test_gateway: all assertions passed");
    return 0;
}
