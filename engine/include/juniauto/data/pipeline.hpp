// pipeline.hpp — data normalization, causal standardization, freshness weights.
// PRINCIPLESLONG.md §1.2–§1.7.

#pragma once

#include <Eigen/Dense>
#include <optional>
#include <string>
#include <vector>

#include "juniauto/data/features.hpp"

namespace juniauto::data {

struct CrossSectionStats {
    // Per-column median and MAD (unit-MAD standardization, §1.2).
    Eigen::VectorXd median;
    Eigen::VectorXd mad;
    // Percentile bands for feature winsorization (§1.2, 1st/99th).
    Eigen::VectorXd p01;
    Eigen::VectorXd p99;
};

// Compute causal cross-section stats from a matrix strictly prior to time t.
CrossSectionStats compute_cross_section_stats(const Eigen::MatrixXd& X_prior);

// Winsorize a feature row against a prior cross-section (§1.2).
void winsorize_row(const CrossSectionStats& stats, Eigen::VectorXd& row);

// MAD-standardize a feature row against prior median/MAD (§1.2).
void standardize_row(const CrossSectionStats& stats, Eigen::VectorXd& row);

// Freshness halflives per family, expressed in trading days (§1.3).
struct FreshnessHalflives {
    double fundamental_days = 20.0;
    double technical_days = 5.0;
    double event_days = 2.0;
    double liquidity_days = 1.0;
    double semantic_days = 3.0;
    double risk_days = 5.0;
};

// Combine per-column freshness weights into a scalar row-level freshness (§1.3).
// The result multiplies raw_signal + (1 - w) * prior_signal at the caller.
double row_freshness(const std::vector<features::GroupId>& col_groups,
                     const FreshnessHalflives& hl,
                     double age_trading_days);

// Observation quality weight w_n used in the ridge posterior (§1.7).
double observation_weight(double quote_freshness,
                          double spread_quality,
                          double fill_resolution,
                          int missing_data_flags,
                          double order_state_cleanliness) noexcept;

}  // namespace juniauto::data
