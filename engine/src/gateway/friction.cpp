// gateway/friction.cpp — Bayesian friction multiplier update (§2.23).
// see docs/knowledge-base/part4-gateway-execution.md §2.23

#include "juniauto/gateway/decision.hpp"
#include "juniauto/utils/math.hpp"
#include "juniauto/utils/stats.hpp"

namespace juniauto::gateway {

// Stateless friction update: take prior multiplier + evidence and return updated multiplier.
// Logit-Bayesian form (§2.23):
//   logit(m_new) = logit(m_prior) + shrink(realized - predicted)
// shrink() is a dampened update to avoid single-fill overreaction.
// Clamp result to [friction_floor, friction_ceiling].
// Not declared in the public header; exposed for use by the orchestrator Python layer
// via a future pybind binding if needed.
// see docs/knowledge-base/part4-gateway-execution.md §2.23
namespace {

double update_friction_impl(double prior_multiplier,
                            double realized_return_bps,
                            double predicted_return_bps,
                            const GatewayConfig& cfg) noexcept {
    // Single fills must not move the multiplier; require a sign-consistent signal.
    // Shrinkage factor 0.05 damps the update — a conservative Bayesian step size.
    // Single evidence point: delta_logit = shrink_rate * sign-normalized residual.
    constexpr double SHRINK_RATE = 0.05;

    const double residual   = realized_return_bps - predicted_return_bps;
    const double delta_logit = SHRINK_RATE * residual;

    const double clamped_prior = math::clamp(
        prior_multiplier,
        cfg.friction_floor + 1.0e-6,
        cfg.friction_ceiling - 1.0e-6);
    const double logit_prior  = stats::logit(clamped_prior);
    const double logit_new    = logit_prior + delta_logit;
    const double new_mult     = stats::sigmoid(logit_new);

    return math::clamp(new_mult, cfg.friction_floor, cfg.friction_ceiling);
}

}  // anonymous namespace

// Public-linkage wrapper so the symbol is reachable from other TUs in the library
// without requiring a header declaration.
double update_friction(double prior_multiplier,
                       double realized_return_bps,
                       double predicted_return_bps,
                       const GatewayConfig& cfg) noexcept {
    return update_friction_impl(prior_multiplier, realized_return_bps, predicted_return_bps, cfg);
}

}  // namespace juniauto::gateway
