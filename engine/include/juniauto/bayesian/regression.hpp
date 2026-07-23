// regression.hpp — grouped ridge / spike-and-slab Bayesian regression.
// PRINCIPLESLONG.md §2.6–§2.8.

#pragma once

#include <Eigen/Dense>
#include <optional>
#include <unordered_map>
#include <vector>

#include "juniauto/data/features.hpp"

namespace juniauto::bayes {

struct GroupPosterior {
    // (§2.7 cached group summary)
    double gamma = 0.0;              // P(gamma_g = 1 | D_t)
    Eigen::VectorXd beta_mean;       // E[beta_g | D_t]
    Eigen::MatrixXd beta_cov;        // Var(beta_g | D_t)
    double tau = 0.0;                // E[tau_g | D_t]
    double n_effective = 0.0;        // effective sample size
    // Utility used for group selection (§2.7):
    //     utility = m_k - rho * sqrt(V_k,k)
    double utility_score = 0.0;
};

struct RegressionConfig {
    double zq = 1.0;                     // (§2.6) conservative-edge quantile
    double rho = 1.0;                    // (§2.7) group-utility skepticism
    double ridge_lambda = 5.0;           // (§2.42)
    double prior_strength_kappa = 20.0;  // (§2.42) empirical-Bayes shrinkage toward group fallback
    double sigma_sq = 1.0;               // plug-in noise variance (§2.7)
    double sigma_model_misspec_sq = 0.0; // added to sigma_edge^2 (§2.7)
};

// Grouped ridge regression with per-group spike-and-slab prior. Online-updateable
// via update() so posteriors persist across decision ticks (§2.7).
class GroupedRidge {
public:
    explicit GroupedRidge(RegressionConfig cfg);

    // One batch update: X (n×p), y (n), weights w (n). Weights are per-row observation
    // quality (§1.7). Assumes X has already been MAD-unit-standardized (§1.2).
    void update(const Eigen::MatrixXd& X,
                const Eigen::VectorXd& y,
                const Eigen::VectorXd& weights,
                const std::vector<features::GroupId>& col_groups);

    // Exponential decay of the effective sample size and shrinkage back toward the prior
    // (§2.42 challenger tuning; also used by the primary ridge for regime adaptation).
    void decay(double factor);

    // Posterior predictive mean μ_edge and variance σ_edge^2 for a candidate row (§2.7).
    double predict_mean(const Eigen::VectorXd& x) const;
    double predict_variance(const Eigen::VectorXd& x) const;

    // σ_total^2 combines σ_edge^2 with the additional uncertainty terms in §2.6.
    double sigma_total(double x_predict_variance,
                       double sigma_quote_stale_sq,
                       double sigma_liquidity_sq,
                       double sigma_order_state_sq,
                       double sigma_regime_shift_sq) const;

    // Conservative edge: μ − zq·σ_total (§2.6).
    double conservative_edge(double mu_edge, double sigma_total_val) const;

    // Per-group cached summary + utility score for group-level research reports (§2.7).
    const GroupPosterior& group_posterior(features::GroupId g) const;

    // Total posterior for reporting / persistence.
    Eigen::VectorXd beta_mean() const;
    Eigen::MatrixXd beta_cov() const;

private:
    RegressionConfig cfg_;
    // Sufficient statistics accumulated across updates (weighted normal equations):
    //     A = X'WX + λI, b = X'Wy
    Eigen::MatrixXd A_;
    Eigen::VectorXd b_;
    double effective_n_ = 0.0;
    std::unordered_map<int, GroupPosterior> group_state_;
};

}  // namespace juniauto::bayes
