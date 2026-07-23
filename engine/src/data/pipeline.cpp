// pipeline.cpp — cross-sectional normalization, freshness weighting, observation quality.
// see docs/knowledge-base/part1-data-preparation.md §1.2–§1.7

#include "juniauto/data/pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "juniauto/utils/math.hpp"
#include "juniauto/utils/stats.hpp"

namespace juniauto::data {

// Compute col-wise 1st/99th percentile + median + MAD from a matrix of prior rows.
// Causal: caller passes X_prior which contains only rows strictly before time t.
// see docs/knowledge-base/part1-data-preparation.md §1.2
CrossSectionStats compute_cross_section_stats(const Eigen::MatrixXd& X_prior) {
    const Eigen::Index n = X_prior.rows();
    const Eigen::Index p = X_prior.cols();

    CrossSectionStats s;
    s.median = Eigen::VectorXd::Zero(p);
    s.mad    = Eigen::VectorXd::Zero(p);
    s.p01    = Eigen::VectorXd::Zero(p);
    s.p99    = Eigen::VectorXd::Zero(p);

    if (n == 0) return s;

    for (Eigen::Index j = 0; j < p; ++j) {
        Eigen::VectorXd col = X_prior.col(j);

        // Sort a copy for percentile computation.
        std::vector<double> sorted_col(col.data(), col.data() + n);
        std::sort(sorted_col.begin(), sorted_col.end());

        // 1st and 99th percentile indices (linear interpolation, floor/ceil per math.hpp).
        const auto idx01 = static_cast<Eigen::Index>(std::floor(0.01 * (n - 1)));
        const auto idx99 = static_cast<Eigen::Index>(std::ceil(0.99 * (n - 1)));
        s.p01[j] = sorted_col[static_cast<std::size_t>(idx01)];
        s.p99[j] = sorted_col[static_cast<std::size_t>(idx99)];

        // median and MAD (single-column vector from the Eigen column).
        auto [med, m] = stats::median_and_mad(col);
        s.median[j] = med;
        s.mad[j]    = m;
    }

    return s;
}

// Winsorize a feature row to [p01, p99] per column (§1.2).
// see docs/knowledge-base/part1-data-preparation.md §1.2
void winsorize_row(const CrossSectionStats& stats, Eigen::VectorXd& row) {
    const Eigen::Index p = row.size();
    for (Eigen::Index j = 0; j < p && j < stats.p01.size(); ++j) {
        row[j] = math::clamp(row[j], stats.p01[j], stats.p99[j]);
    }
}

// MAD unit-scale standardization: (x - median) / (1.4826 * MAD) per column (§1.2).
// see docs/knowledge-base/part1-data-preparation.md §1.2
void standardize_row(const CrossSectionStats& stats, Eigen::VectorXd& row) {
    const Eigen::Index p = row.size();
    for (Eigen::Index j = 0; j < p && j < stats.median.size(); ++j) {
        row[j] = math::standardize_mad(row[j], stats.median[j], stats.mad[j]);
    }
}

// Row-level freshness: weighted average of per-column freshness weights grouped by family.
// Weight of a column = exp(-age / halflife_for_its_group).
// Columns with no valid group get the minimum (technical) halflife as a safe default.
// see docs/knowledge-base/part1-data-preparation.md §1.3
double row_freshness(const std::vector<features::GroupId>& col_groups,
                     const FreshnessHalflives& hl,
                     double age_trading_days) {
    if (col_groups.empty()) return 0.0;

    double sum_w = 0.0;
    double count  = 0.0;

    for (const auto& g : col_groups) {
        double halflife = hl.technical_days;  // safe default

        switch (g) {
            case features::GroupId::FundamentalQuality:
                halflife = hl.fundamental_days;  // several weeks (§1.3)
                break;
            case features::GroupId::TechMomentum:
            case features::GroupId::TechChartStructure:
            case features::GroupId::VolatilityRange:
                halflife = hl.technical_days;    // several trading days (§1.3)
                break;
            case features::GroupId::EventRegime:
                halflife = hl.event_days;        // hours to trading days (§1.3)
                break;
            case features::GroupId::Liquidity:
                halflife = hl.liquidity_days;    // 1 trading day (§1.3)
                break;
            case features::GroupId::ProvenanceRole:
            case features::GroupId::AccountState:
            case features::GroupId::ExecutionTelemetry:
                halflife = hl.semantic_days;     // treated as semantic / risk context
                break;
        }

        sum_w += math::freshness_weight(age_trading_days, halflife);
        count += 1.0;
    }

    return count > 0.0 ? sum_w / count : 0.0;
}

// Observation quality weight w_n (§1.7).
// Bounded product of clamped [0,1] input factors.
// Quote freshness and spread quality are primary levers; missing-data penalty
// reduces weight multiplicatively; order-state cleanliness and fill resolution
// are secondary multipliers.
// Functional form is open per spec (§1.7 TODO); a product of clamped factors
// is the standard approach in weighted-least-squares literature (Gelman & Hill 2007).
// see docs/knowledge-base/part1-data-preparation.md §1.7
double observation_weight(double quote_freshness,
                          double spread_quality,
                          double fill_resolution,
                          int missing_data_flags,
                          double order_state_cleanliness) noexcept {
    // Each input clamped to [0, 1]; missing-data flag count maps to a penalty factor.
    const double q  = math::clamp(quote_freshness,        0.0, 1.0);
    const double sq = math::clamp(spread_quality,         0.0, 1.0);
    const double fr = math::clamp(fill_resolution,        0.0, 1.0);
    const double os = math::clamp(order_state_cleanliness,0.0, 1.0);

    // Each set missing-data flag halves the weight; floor at 1/16 to avoid exact zero.
    // Chosen constant 0.5 per flag: consistent with a Bayesian credibility reduction of
    // one bit of information per missing feature.
    const int capped_flags = std::min(missing_data_flags, 4);
    const double missing_factor = std::pow(0.5, static_cast<double>(capped_flags));

    // Product of all quality factors.
    return q * sq * fr * os * missing_factor;
}

}  // namespace juniauto::data
