// portfolio/sizing.hpp — Kelly-fraction sizing with hard caps and concentration penalty.
// PRINCIPLESLONG.md §2.29–§2.30.

#pragma once

#include <Eigen/Dense>
#include <string>
#include <unordered_map>

namespace juniauto::portfolio {

struct SizingConfig {
    double max_name_weight = 0.10;              // (§2.29) hard per-name cap
    double cash_floor = 0.05;                   // (§2.29) minimum cash
    double aggregate_comfortable_weight = 0.20; // (§2.30) cross-horizon aggregate cap
    double gamma_risk = 1.0;                    // (§2.1) risk-aversion in G(w)
    double lambda_confirm = 0.05;               // (§2.1) confirmatory term coefficient
};

struct SizingInput {
    double edge_bps = 0.0;         // composite model edge (§2.22a) after gateway
    double variance_bps_sq = 0.0;  // σ_total^2 (§2.6)
    double confidence = 1.0;       // q_i (§2.5)
    double account_equity = 0.0;
    double mid_price = 0.0;
};

struct SizingResult {
    double weight = 0.0;      // fraction of equity
    double notional = 0.0;    // dollars
    double shares = 0.0;
    bool at_name_cap = false;
    bool at_aggregate_cap = false;
    bool at_cash_floor = false;
};

// Kelly weight = μ / σ^2 (§2.29), with caps applied afterwards.
double kelly_weight(double edge_bps, double variance_bps_sq) noexcept;

// Concentration penalty across a target weight vector (§2.30). Fed into the gateway cost.
double concentration_penalty_bps(const Eigen::VectorXd& weights, double comfortable_weight);

// One-shot sizing for a single symbol given the account state and cross-horizon holdings.
SizingResult size_position(
    const std::string& symbol,
    const SizingInput& in,
    const std::unordered_map<int, double>& horizon_weights,  // horizon-id -> current weight
    const SizingConfig& cfg
);

}  // namespace juniauto::portfolio
