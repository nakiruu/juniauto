// features.hpp — feature vector layout and the six-family group ids used
// by the grouped spike-and-slab prior (§1.4–1.5).

#pragma once

#include <Eigen/Dense>
#include <array>
#include <string>
#include <vector>

namespace juniauto::features {

// Group ids match the semantic groups in PRINCIPLESLONG.md §1.5.
enum class GroupId : int {
    TechMomentum = 0,       // trend, breakout, RS, torque
    TechChartStructure = 1, // support, extension, oversold, range
    VolatilityRange = 2,    // realized vol, ATR, volume shock
    Liquidity = 3,          // spread, quote age, dollar volume, partial-fill risk
    FundamentalQuality = 4, // growth-quality, earnings proxy, liquidity quality
    ProvenanceRole = 5,     // primary / secondary / retained indicators
    EventRegime = 6,        // catalyst, sector stabilization, session bucket
    AccountState = 7,       // current weight, target delta, cash, open orders, recent fills
    ExecutionTelemetry = 8, // recent slippage, fill quality, queue delay
};

inline constexpr std::size_t kNumGroups = 9;
std::string group_name(GroupId g);

// Wide row vector fed into the Bayesian regression. Column ordering is fixed
// by BuildFeatureLayout() so the pybind bindings and QuestDB schema agree.
struct FeatureVector {
    Eigen::VectorXd values;               // length = layout().total_dim
    std::vector<GroupId> column_groups;   // per-column group tag, same length as values
    double freshness_weight = 0.0;        // (§1.3)
    double data_quality = 1.0;            // (§1.7)
};

// Immutable layout: names + groups for every feature column.
struct FeatureColumn {
    std::string name;
    GroupId group;
};

const std::vector<FeatureColumn>& layout();
int feature_dim();

}  // namespace juniauto::features
