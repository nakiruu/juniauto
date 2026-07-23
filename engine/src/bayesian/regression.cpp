// regression.cpp — GroupedRidge Bayesian regression.
// see docs/knowledge-base/part2-signals-bayesian.md §2.6–§2.8, §2.42

#include "juniauto/bayesian/regression.hpp"

#include <algorithm>
#include <cmath>

#include "juniauto/utils/stats.hpp"

namespace juniauto::bayes {

// ---- Construction ----

GroupedRidge::GroupedRidge(RegressionConfig cfg)
    : cfg_(cfg) {
    // A and b are sized on first update(); zero-sized until then.
}

// ---- Update (online weighted normal equations) ----
// Sufficient statistics:
//   A = X'WX + λI   (precision-like matrix)
//   b = X'Wy
// Accumulated across calls; effective_n_ tracks weighted observation count.
// see docs/knowledge-base/part2-signals-bayesian.md §2.7
void GroupedRidge::update(const Eigen::MatrixXd& X,
                          const Eigen::VectorXd& y,
                          const Eigen::VectorXd& weights,
                          const std::vector<features::GroupId>& col_groups) {
    if (X.rows() == 0 || X.cols() == 0) return;

    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();

    // Lazy initialisation on first call.
    if (A_.rows() != p) {
        // Ridge: A = λI initially (prior precision).
        // see docs/knowledge-base/part2-signals-bayesian.md §2.42
        A_ = cfg_.ridge_lambda * Eigen::MatrixXd::Identity(p, p);
        b_ = Eigen::VectorXd::Zero(p);
    }

    // Diagonal weight matrix W applied as elementwise scaling.
    const Eigen::VectorXd w_vec = weights.head(n).cwiseMax(0.0);

    // A += X' * diag(w) * X
    A_ += X.transpose() * w_vec.asDiagonal() * X;
    // b += X' * diag(w) * y
    b_ += X.transpose() * (w_vec.array() * y.head(n).array()).matrix();

    effective_n_ += w_vec.sum();

    // Refresh per-group cached posteriors.
    // Build group -> column index mapping from col_groups.
    std::unordered_map<int, std::vector<Eigen::Index>> group_cols;
    for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(col_groups.size()) && j < p; ++j) {
        group_cols[static_cast<int>(col_groups[static_cast<std::size_t>(j)])].push_back(j);
    }

    // Global posterior (full solve).
    // A is symmetric PD by construction (ridge); use LDLT for stability.
    const Eigen::LDLT<Eigen::MatrixXd> ldlt(A_);
    const Eigen::VectorXd beta_m = ldlt.solve(b_);
    const Eigen::MatrixXd beta_V = cfg_.sigma_sq * ldlt.solve(Eigen::MatrixXd::Identity(p, p));

    for (auto& [gid, cols] : group_cols) {
        GroupPosterior& gp = group_state_[gid];
        const Eigen::Index d = static_cast<Eigen::Index>(cols.size());

        gp.beta_mean = Eigen::VectorXd(d);
        gp.beta_cov  = Eigen::MatrixXd(d, d);

        for (Eigen::Index a = 0; a < d; ++a) {
            gp.beta_mean[a] = beta_m[cols[static_cast<std::size_t>(a)]];
            for (Eigen::Index b_idx = 0; b_idx < d; ++b_idx) {
                gp.beta_cov(a, b_idx) =
                    beta_V(cols[static_cast<std::size_t>(a)],
                           cols[static_cast<std::size_t>(b_idx)]);
            }
        }

        // tau: average marginal posterior SD as a proxy for group slab scale.
        gp.tau = std::sqrt(std::max(0.0, gp.beta_cov.diagonal().mean()));

        // Effective sample size for this group (share of global n_eff).
        gp.n_effective = effective_n_;

        // Group utility: m_k - rho * sqrt(V_k,k) averaged over group columns (§2.7).
        // Uses marginal SD (diagonal of beta_cov), ignoring off-diagonal per §2.7 note.
        double utility_sum = 0.0;
        for (Eigen::Index a = 0; a < d; ++a) {
            const double m_k = gp.beta_mean[a];
            const double V_kk = gp.beta_cov(a, a);
            utility_sum += m_k - cfg_.rho * std::sqrt(std::max(0.0, V_kk));
        }
        gp.utility_score = d > 0 ? utility_sum / static_cast<double>(d) : 0.0;

        // gamma: P(gamma_g = 1 | D_t).
        // Approximated as sigmoid of standardized utility (§2.7 cached-summary formulation).
        // Standardize by (utility - 0) / 1 — utility is already in bps-of-beta units.
        gp.gamma = stats::sigmoid(gp.utility_score);
    }
}

// Decay: scale A and b by factor to shrink effective sample size (§2.42).
// see docs/knowledge-base/part2-signals-bayesian.md §2.42
void GroupedRidge::decay(double factor) {
    if (A_.rows() == 0) return;
    const Eigen::Index p = A_.rows();
    // Re-inject prior precision to prevent A from decaying to near-zero.
    // After scaling, reset the diagonal ridge floor.
    A_ = factor * A_;
    b_ = factor * b_;
    effective_n_ *= factor;
    // Re-add the base ridge prior (λI) so that (1-factor)*λI stays as a prior floor.
    // This implements "shrink toward prior" on decay: A stays >= λI.
    A_ += (1.0 - factor) * cfg_.ridge_lambda * Eigen::MatrixXd::Identity(p, p);
}

// Posterior predictive mean: x · β_mean.
// see docs/knowledge-base/part2-signals-bayesian.md §2.7
double GroupedRidge::predict_mean(const Eigen::VectorXd& x) const {
    if (A_.rows() == 0 || x.size() == 0) return 0.0;
    const Eigen::LDLT<Eigen::MatrixXd> ldlt(A_);
    const Eigen::VectorXd beta_m = ldlt.solve(b_);
    return x.dot(beta_m);
}

// Posterior predictive variance: x' Σ x + σ² (§2.7).
// sigma_sq plays the role of the noise variance plug-in.
// see docs/knowledge-base/part2-signals-bayesian.md §2.7
double GroupedRidge::predict_variance(const Eigen::VectorXd& x) const {
    if (A_.rows() == 0 || x.size() == 0) return cfg_.sigma_sq;
    const Eigen::LDLT<Eigen::MatrixXd> ldlt(A_);
    const Eigen::VectorXd Ainv_x = ldlt.solve(x);
    return cfg_.sigma_sq * x.dot(Ainv_x) + cfg_.sigma_sq;
}

// sigma_total^2 = x'Σx + σ_model_misspec^2 + σ_quote_stale^2 + ... (§2.6).
// see docs/knowledge-base/part2-signals-bayesian.md §2.6
double GroupedRidge::sigma_total(double x_predict_variance,
                                 double sigma_quote_stale_sq,
                                 double sigma_liquidity_sq,
                                 double sigma_order_state_sq,
                                 double sigma_regime_shift_sq) const {
    const double var_total = x_predict_variance
                           + cfg_.sigma_model_misspec_sq
                           + sigma_quote_stale_sq
                           + sigma_liquidity_sq
                           + sigma_order_state_sq
                           + sigma_regime_shift_sq;
    return std::sqrt(std::max(0.0, var_total));
}

// Conservative edge: mu - zq * sigma_total (§2.6).
// zq = 1.0 per §2.6 applied config.
// see docs/knowledge-base/part2-signals-bayesian.md §2.6
double GroupedRidge::conservative_edge(double mu_edge, double sigma_total_val) const {
    return mu_edge - cfg_.zq * sigma_total_val;
}

// Per-group cached posterior (§2.7).
const GroupPosterior& GroupedRidge::group_posterior(features::GroupId g) const {
    const auto it = group_state_.find(static_cast<int>(g));
    if (it == group_state_.end()) {
        // Return a default-constructed zero posterior for unseen groups.
        static const GroupPosterior kEmpty{};
        return kEmpty;
    }
    return it->second;
}

// Global posterior accessors.
Eigen::VectorXd GroupedRidge::beta_mean() const {
    if (A_.rows() == 0) return Eigen::VectorXd{};
    const Eigen::LDLT<Eigen::MatrixXd> ldlt(A_);
    return ldlt.solve(b_);
}

Eigen::MatrixXd GroupedRidge::beta_cov() const {
    if (A_.rows() == 0) return Eigen::MatrixXd{};
    const Eigen::Index p = A_.rows();
    const Eigen::LDLT<Eigen::MatrixXd> ldlt(A_);
    return cfg_.sigma_sq * ldlt.solve(Eigen::MatrixXd::Identity(p, p));
}

}  // namespace juniauto::bayes
