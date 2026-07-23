// features.cpp — fixed feature layout matching the QuestDB `features` table
// in orchestrator/juniauto/db/schema.sql (same column order, same group tags).
// see docs/knowledge-base/part1-data-preparation.md §1.4–§1.5

#include "juniauto/data/features.hpp"

#include <stdexcept>

namespace juniauto::features {

// Column order mirrors schema.sql §1.4:
//   Technical (§1.4.1), Fundamental (§1.4.2), Event (§1.4.3),
//   Semantic (§1.4.4), Liquidity (§1.4.5), Risk (§1.4.6).
// Freshness/data_quality weights are row-level scalars, not regression features;
// they are stored in schema.sql but excluded from the feature vector fed to ridge.
static const std::vector<FeatureColumn> kLayout = {
    // ---- Technical Momentum (§1.4.1) ----
    {"trend_slope",           GroupId::TechMomentum},
    {"relative_strength",     GroupId::TechMomentum},
    {"breakout_strength",     GroupId::TechMomentum},
    {"ma_distance",           GroupId::TechMomentum},
    {"price_acceleration",    GroupId::TechMomentum},
    {"volume_confirmation",   GroupId::TechMomentum},
    // ---- Technical Chart Structure (§1.5) ----
    {"support_defense",       GroupId::TechChartStructure},
    // ---- Fundamental Quality (§1.4.2) ----
    {"earnings_quality",      GroupId::FundamentalQuality},
    {"revenue_growth",        GroupId::FundamentalQuality},
    {"profitability",         GroupId::FundamentalQuality},
    {"balance_sheet_strength",GroupId::FundamentalQuality},
    {"valuation_quality",     GroupId::FundamentalQuality},
    {"analyst_revision",      GroupId::FundamentalQuality},
    // ---- Event / Regime (§1.4.3) ----
    {"catalyst_score",        GroupId::EventRegime},
    {"earnings_surprise",     GroupId::EventRegime},
    {"guidance_change",       GroupId::EventRegime},
    // ---- Semantic (§1.4.4 / EventRegime group — context features) ----
    {"context_alignment",     GroupId::EventRegime},
    {"sector_context",        GroupId::EventRegime},
    // ---- Liquidity / Microstructure (§1.4.5) ----
    {"spread_bps",            GroupId::Liquidity},
    {"dollar_volume",         GroupId::Liquidity},
    {"relative_volume",       GroupId::Liquidity},
    {"depth_proxy",           GroupId::Liquidity},
    // ---- Volatility / Risk (§1.4.6) ----
    {"realized_vol_bps",      GroupId::VolatilityRange},
    {"beta",                  GroupId::VolatilityRange},
    {"gap_risk",              GroupId::VolatilityRange},
    {"crowding",              GroupId::VolatilityRange},
};

const std::vector<FeatureColumn>& layout() {
    return kLayout;
}

int feature_dim() {
    return static_cast<int>(kLayout.size());
}

std::string group_name(GroupId g) {
    switch (g) {
        case GroupId::TechMomentum:       return "TechMomentum";
        case GroupId::TechChartStructure: return "TechChartStructure";
        case GroupId::VolatilityRange:    return "VolatilityRange";
        case GroupId::Liquidity:          return "Liquidity";
        case GroupId::FundamentalQuality: return "FundamentalQuality";
        case GroupId::ProvenanceRole:     return "ProvenanceRole";
        case GroupId::EventRegime:        return "EventRegime";
        case GroupId::AccountState:       return "AccountState";
        case GroupId::ExecutionTelemetry: return "ExecutionTelemetry";
    }
    return "Unknown";
}

}  // namespace juniauto::features
