// stats.hpp — reusable statistical primitives (median, MAD, Roll bounce-corrected variance).
// PRINCIPLESLONG.md references inline as (§X.Y).

#pragma once

#include <Eigen/Dense>

namespace juniauto::stats {

// Median of a vector (mutates a working copy). O(n).
double median(const Eigen::VectorXd& v);

// Median absolute deviation about the median. Consistent estimator: sigma ≈ 1.4826 * MAD.
double mad(const Eigen::VectorXd& v);

// (median, MAD) pair — one pass for the median, one for MAD.
std::pair<double, double> median_and_mad(const Eigen::VectorXd& v);

// Roll (1984) bid-ask bounce correction (§2.10):
//     Var(r_corrected) = max(0, Var(r_close) - s^2/4)
// where s is the decimal spread (spread_bps / 10000).
double roll_corrected_variance(double variance_close, double spread_decimal) noexcept;

// Annualized volatility in bps from a corrected daily variance:
//     10000 * sqrt(max(0, Var) * 252)
double annualized_vol_bps(double corrected_daily_variance) noexcept;

// Exponentially-decayed weighted mean over a series (§2.24 recent_fill_slippage_bps
// aggregation): weight of k-th sample = decay^(k-1), k=1 = most recent.
double exp_decay_mean(const Eigen::VectorXd& samples, double decay) noexcept;

// Sigmoid / logit helpers used in the friction multiplier (§2.23).
inline double sigmoid(double x) noexcept {
    return 1.0 / (1.0 + std::exp(-x));
}
inline double logit(double p) noexcept {
    return std::log(p / (1.0 - p));
}

}  // namespace juniauto::stats
