// test_costs.cpp — unit tests for the cost model.
// Key assertion: BUY exit_reserved_bps = 0.5525 * base_side_bps
// (= 0.65 * 0.85 * base_side_bps, exact product per §2.24 spec quirk).
// see docs/knowledge-base/part4-gateway-execution.md §2.24

#include <cassert>
#include <cmath>
#include <cstdio>

#include "juniauto/costs/model.hpp"

static bool approx_eq(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) <= tol;
}

int main() {
    // ---- Canonical input for the BUY exit_reserved test ----
    // Chosen so base_side_cost is straightforward to verify.
    juniauto::costs::MarketState s;
    s.spread_bps            = 10.0;        // half-spread = 5 bps
    s.volatility_bps        = 100.0;       // vol term = 0.10 * 100 = 10 bps
    s.bar_dollar_volume     = 1'000'000.0; // notional 0, size_ratio = 0
    s.adv_dollar            = 5'000'000.0;
    s.quote_age_sessions    = 0.038;       // IEX 15-min floor (0 stale bps by design)
    s.gap_days_to_next_session = 1;        // gap_bps = max(0, (1-1)*1.75) = 0
    s.session_multiplier    = 1.0;         // regular session
    s.adverse_selection_share = 0.35;

    juniauto::costs::SlippageStats slip;
    slip.recent_fill_slippage_bps = 5.0;  // clamped to [0,50]; contributes 5 bps

    juniauto::costs::CostConfig cfg;  // defaults from header

    // base_side = (0.5*10 + 0.10*100 + 0.25 [liq_min] + 0 [stale] + 5 [slip])
    //           * 1.0 [session_mult] + 0 [gap]
    //           = (5 + 10 + 0.25 + 0 + 5) * 1.0
    //           = 20.25 bps
    const double expected_base = 20.25;

    juniauto::costs::Order ord;
    ord.symbol  = "TEST";
    ord.notional = 0.0;
    ord.predicted_holding_seconds = 23400.0;

    const juniauto::costs::CostBreakdown bd =
        juniauto::costs::compute_cost(ord, s, slip, cfg, /*model_edge_bps=*/100.0);

    // Verify base_side.
    assert(approx_eq(bd.base_side_bps, expected_base, 1e-9) &&
           "base_side_bps mismatch");

    // Key assertion: exit_reserved = 0.65 * 0.85 * base_side = 0.5525 * base_side.
    // see docs/knowledge-base/part4-gateway-execution.md §2.24 spec quirk #1
    const double expected_exit_reserved = 0.5525 * expected_base;
    assert(approx_eq(bd.exit_reserved_bps, expected_exit_reserved, 1e-9) &&
           "exit_reserved_bps must equal 0.5525 * base_side_bps (0.65 * 0.85 product)");

    // Confirm the product decomposition exactly.
    const double via_product = cfg.buy_exit_haircut * cfg.buy_exit_future_factor * expected_base;
    assert(approx_eq(via_product, expected_exit_reserved, 1e-12) &&
           "buy_exit_haircut * buy_exit_future_factor must yield exactly 0.5525");

    // Stale-quote = 0 for IEX 15-min (0.038 sessions < 0.5 threshold).
    // Validated implicitly: base_side matches expected_base which assumes stale=0.

    std::puts("test_costs: all assertions passed");
    return 0;
}
