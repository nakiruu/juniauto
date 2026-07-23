// portfolio/sizing.cpp — Kelly-fraction sizing with hard caps and concentration penalty.
// see docs/knowledge-base/part4-gateway-execution.md §2.29–§2.30

#include "juniauto/portfolio/sizing.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "juniauto/utils/math.hpp"

namespace juniauto::portfolio {

// Kelly weight = mu / sigma^2 (§2.29), gamma_risk = 1 (log-utility / pure Kelly).
// see docs/knowledge-base/part4-gateway-execution.md §2.29
double kelly_weight(double edge_bps, double variance_bps_sq) noexcept {
    // Epsilon floor prevents division by zero; 1e-9 is negligible vs real bps^2 values.
    constexpr double EPS = 1.0e-9;
    return edge_bps / std::max(EPS, variance_bps_sq);
}

// Concentration penalty (§2.30):
//   concentration_cost = scale * sum_i max(w_i - comfortable_weight, 0)^2
// Numeric scale is unspecified in spec (§2.30 TODO); use 1.0 as a neutral unit.
// Returns bps (caller interprets as cost to subtract from gross edge).
// see docs/knowledge-base/part4-gateway-execution.md §2.30
double concentration_penalty_bps(const Eigen::VectorXd& weights, double comfortable_weight) {
    constexpr double CONCENTRATION_SCALE = 1.0;  // §2.30 TODO — calibrate from replay
    double penalty = 0.0;
    for (Eigen::Index i = 0; i < weights.size(); ++i) {
        const double excess = std::max(0.0, weights[i] - comfortable_weight);
        penalty += excess * excess;
    }
    return CONCENTRATION_SCALE * penalty;
}

// Full single-symbol sizing (§2.29):
//   1. Kelly raw weight
//   2. Confidence multiplier
//   3. Per-name hard cap (10%)
//   4. Cross-horizon aggregate cap (20%)
//   5. Cash floor check (5%)
// see docs/knowledge-base/part4-gateway-execution.md §2.29–§2.30
SizingResult size_position(
    const std::string& /*symbol*/,
    const SizingInput& in,
    const std::unordered_map<int, double>& horizon_weights,
    const SizingConfig& cfg)
{
    SizingResult res;

    if (in.account_equity <= 0.0 || in.mid_price <= 0.0) return res;

    // Step 1: Kelly weight.
    double w = kelly_weight(in.edge_bps, in.variance_bps_sq);

    // Step 2: Apply confidence multiplier (§2.5 / §2.29).
    w *= math::clamp(in.confidence, 0.0, 1.0);

    // Negative edges should not produce long positions (long-only system).
    if (w <= 0.0) return res;

    // Step 3: Per-name hard cap (§2.29, MacLean-Thorp-Ziemba).
    bool at_name = false;
    if (w > cfg.max_name_weight) {
        w = cfg.max_name_weight;
        at_name = true;
    }

    // Step 4: Cross-horizon aggregate cap (§2.30).
    // aggregate_weight_i = sum_h w_i,h.
    double existing_aggregate = 0.0;
    for (const auto& [/*h*/ _, hw] : horizon_weights) {
        existing_aggregate += hw;
    }
    bool at_agg = false;
    const double max_agg = cfg.aggregate_comfortable_weight;  // 0.20
    if (existing_aggregate + w > max_agg) {
        w = std::max(0.0, max_agg - existing_aggregate);
        at_agg = true;
    }

    // Step 5: Cash floor check (§2.29).
    // If adding this position would push investable weight above (1 - cash_floor),
    // reduce the weight proportionally.  Simplified: if w alone exceeds the headroom,
    // cap it.
    bool at_cash = false;
    const double max_investable = 1.0 - cfg.cash_floor;
    if (w > max_investable) {
        w = max_investable;
        at_cash = true;
    }

    res.weight       = w;
    res.notional     = w * in.account_equity;
    res.shares       = (in.mid_price > 0.0) ? res.notional / in.mid_price : 0.0;
    res.at_name_cap      = at_name;
    res.at_aggregate_cap = at_agg;
    res.at_cash_floor    = at_cash;

    return res;
}

}  // namespace juniauto::portfolio
