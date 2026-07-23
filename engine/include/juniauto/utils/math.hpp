// math.hpp — small header-only numerical helpers used across the engine.
// PRINCIPLESLONG.md references appear inline as (§X.Y).

#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace juniauto::math {

// clamp(x, lo, hi)
template <typename T>
constexpr T clamp(T x, T lo, T hi) noexcept {
    return std::max(lo, std::min(hi, x));
}

// Winsorization at the [lo, hi] percentile band (§1.2). Assumes sorted `values`.
inline double winsorize_percentile(const Eigen::VectorXd& sorted_values, double p_lo, double p_hi,
                                   double x) noexcept {
    if (sorted_values.size() == 0) return x;
    const auto n = static_cast<Eigen::Index>(sorted_values.size());
    const auto idx_lo = static_cast<Eigen::Index>(std::floor(p_lo * (n - 1)));
    const auto idx_hi = static_cast<Eigen::Index>(std::ceil(p_hi * (n - 1)));
    return clamp(x, sorted_values[idx_lo], sorted_values[idx_hi]);
}

// MAD-based winsorization at ±k * MAD (§1.2). `median` and `mad` computed once per feature.
inline double winsorize_mad(double x, double median_val, double mad, double k = 5.0) noexcept {
    const double lo = median_val - k * mad;
    const double hi = median_val + k * mad;
    return clamp(x, lo, hi);
}

// Unit-MAD standardization: (x - median) / (1.4826 * MAD) (§1.2).
// The 1.4826 factor makes MAD a consistent estimator of sigma under Gaussian data.
inline double standardize_mad(double x, double median_val, double mad) noexcept {
    constexpr double MAD_TO_SIGMA = 1.4826;
    const double denom = MAD_TO_SIGMA * mad;
    return denom > 0.0 ? (x - median_val) / denom : 0.0;
}

// Freshness weight: exp(-age / halflife) (§1.3). Halflife in the same unit as age.
inline double freshness_weight(double age, double halflife) noexcept {
    return halflife > 0.0 ? std::exp(-age / halflife) : 0.0;
}

// Effective sample size correction for overlapping labels (§2.41):
// n_eff = n / (1 + 2 * rho_label).
inline double n_effective(double n_raw, double rho_label) noexcept {
    return n_raw / (1.0 + 2.0 * rho_label);
}

}  // namespace juniauto::math
