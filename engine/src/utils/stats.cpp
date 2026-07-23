// stats.cpp — statistical primitives: median, MAD, Roll correction, exp-decay mean.
// see docs/knowledge-base/part3-costs.md §2.10, part2-signals-bayesian.md §2.7

#include "juniauto/utils/stats.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace juniauto::stats {

double median(const Eigen::VectorXd& v) {
    if (v.size() == 0) return 0.0;
    std::vector<double> buf(v.data(), v.data() + v.size());
    const auto n = buf.size();
    const auto mid = n / 2;
    std::nth_element(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(mid), buf.end());
    if (n % 2 == 1) {
        return buf[mid];
    }
    // Even length: average of two middle elements (nth_element guarantees [mid] is correct;
    // we need the element just below it).
    const double upper = buf[mid];
    std::nth_element(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(mid - 1), buf.end());
    return 0.5 * (buf[mid - 1] + upper);
}

double mad(const Eigen::VectorXd& v) {
    if (v.size() == 0) return 0.0;
    const double med = median(v);
    Eigen::VectorXd deviations = (v.array() - med).abs();
    return median(deviations);
}

std::pair<double, double> median_and_mad(const Eigen::VectorXd& v) {
    if (v.size() == 0) return {0.0, 0.0};
    const double med = median(v);
    Eigen::VectorXd deviations = (v.array() - med).abs();
    const double m = median(deviations);
    return {med, m};
}

// Roll (1984) bid-ask bounce correction (§2.10):
//   Var(r_corrected) = max(0, Var(r_close) - s^2/4)
// where s = spread_decimal = spread_bps / 10000.
// see docs/knowledge-base/part3-costs.md §2.10
double roll_corrected_variance(double variance_close, double spread_decimal) noexcept {
    const double correction = (spread_decimal * spread_decimal) / 4.0;
    return std::max(0.0, variance_close - correction);
}

// 10000 * sqrt(max(0, var) * 252) — annualized vol in bps.
// see docs/knowledge-base/part3-costs.md §2.10
double annualized_vol_bps(double corrected_daily_variance) noexcept {
    // 252 trading days per year.  §2.10
    constexpr double TRADING_DAYS_PER_YEAR = 252.0;
    return 10000.0 * std::sqrt(std::max(0.0, corrected_daily_variance) * TRADING_DAYS_PER_YEAR);
}

// Exponentially-decayed weighted mean; weight of k-th sample = decay^(k-1), k=1 most recent.
// Used for recent_fill_slippage_bps aggregation (§2.24, alpha_fill = 0.75).
// see docs/knowledge-base/part4-gateway-execution.md §2.24
double exp_decay_mean(const Eigen::VectorXd& samples, double decay) noexcept {
    if (samples.size() == 0) return 0.0;
    double weighted_sum = 0.0;
    double weight_sum = 0.0;
    double w = 1.0;
    for (Eigen::Index k = 0; k < samples.size(); ++k) {
        weighted_sum += w * samples[k];
        weight_sum += w;
        w *= decay;
    }
    return weight_sum > 0.0 ? weighted_sum / weight_sum : 0.0;
}

}  // namespace juniauto::stats
