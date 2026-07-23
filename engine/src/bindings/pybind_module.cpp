// pybind_module.cpp — Python bindings for the JuniAuto engine.
// The Python orchestrator imports this as `quant_engine`.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "juniauto/bayesian/regression.hpp"
#include "juniauto/costs/model.hpp"
#include "juniauto/data/features.hpp"
#include "juniauto/data/pipeline.hpp"
#include "juniauto/gateway/decision.hpp"
#include "juniauto/portfolio/sizing.hpp"

namespace py = pybind11;

PYBIND11_MODULE(quant_engine, m) {
    m.doc() = "JuniAuto C++ core engine (PRINCIPLESLONG.md implementation)";

    // ---------------- Features ----------------
    py::enum_<juniauto::features::GroupId>(m, "GroupId")
        .value("TechMomentum", juniauto::features::GroupId::TechMomentum)
        .value("TechChartStructure", juniauto::features::GroupId::TechChartStructure)
        .value("VolatilityRange", juniauto::features::GroupId::VolatilityRange)
        .value("Liquidity", juniauto::features::GroupId::Liquidity)
        .value("FundamentalQuality", juniauto::features::GroupId::FundamentalQuality)
        .value("ProvenanceRole", juniauto::features::GroupId::ProvenanceRole)
        .value("EventRegime", juniauto::features::GroupId::EventRegime)
        .value("AccountState", juniauto::features::GroupId::AccountState)
        .value("ExecutionTelemetry", juniauto::features::GroupId::ExecutionTelemetry);

    py::class_<juniauto::features::FeatureVector>(m, "FeatureVector")
        .def(py::init<>())
        .def_readwrite("values", &juniauto::features::FeatureVector::values)
        .def_readwrite("column_groups", &juniauto::features::FeatureVector::column_groups)
        .def_readwrite("freshness_weight", &juniauto::features::FeatureVector::freshness_weight)
        .def_readwrite("data_quality", &juniauto::features::FeatureVector::data_quality);

    m.def("feature_dim", &juniauto::features::feature_dim);

    // ---------------- Pipeline (winsorize / standardize) ----------------
    py::class_<juniauto::data::CrossSectionStats>(m, "CrossSectionStats")
        .def(py::init<>())
        .def_readwrite("median", &juniauto::data::CrossSectionStats::median)
        .def_readwrite("mad", &juniauto::data::CrossSectionStats::mad)
        .def_readwrite("p01", &juniauto::data::CrossSectionStats::p01)
        .def_readwrite("p99", &juniauto::data::CrossSectionStats::p99);

    m.def("compute_cross_section_stats", &juniauto::data::compute_cross_section_stats,
          py::arg("X_prior"));
    m.def("winsorize_row", &juniauto::data::winsorize_row,
          py::arg("stats"), py::arg("row"));
    m.def("standardize_row", &juniauto::data::standardize_row,
          py::arg("stats"), py::arg("row"));

    // ---------------- Bayesian ----------------
    py::class_<juniauto::bayes::RegressionConfig>(m, "RegressionConfig")
        .def(py::init<>())
        .def_readwrite("zq", &juniauto::bayes::RegressionConfig::zq)
        .def_readwrite("rho", &juniauto::bayes::RegressionConfig::rho)
        .def_readwrite("ridge_lambda", &juniauto::bayes::RegressionConfig::ridge_lambda)
        .def_readwrite("prior_strength_kappa", &juniauto::bayes::RegressionConfig::prior_strength_kappa)
        .def_readwrite("sigma_sq", &juniauto::bayes::RegressionConfig::sigma_sq)
        .def_readwrite("sigma_model_misspec_sq", &juniauto::bayes::RegressionConfig::sigma_model_misspec_sq);

    py::class_<juniauto::bayes::GroupPosterior>(m, "GroupPosterior")
        .def_readonly("gamma", &juniauto::bayes::GroupPosterior::gamma)
        .def_readonly("beta_mean", &juniauto::bayes::GroupPosterior::beta_mean)
        .def_readonly("beta_cov", &juniauto::bayes::GroupPosterior::beta_cov)
        .def_readonly("tau", &juniauto::bayes::GroupPosterior::tau)
        .def_readonly("n_effective", &juniauto::bayes::GroupPosterior::n_effective)
        .def_readonly("utility_score", &juniauto::bayes::GroupPosterior::utility_score);

    py::class_<juniauto::bayes::GroupedRidge>(m, "GroupedRidge")
        .def(py::init<juniauto::bayes::RegressionConfig>(), py::arg("cfg"))
        .def("update", &juniauto::bayes::GroupedRidge::update,
             py::arg("X"), py::arg("y"), py::arg("weights"), py::arg("col_groups"))
        .def("decay", &juniauto::bayes::GroupedRidge::decay, py::arg("factor"))
        .def("predict_mean", &juniauto::bayes::GroupedRidge::predict_mean, py::arg("x"))
        .def("predict_variance", &juniauto::bayes::GroupedRidge::predict_variance, py::arg("x"))
        .def("sigma_total", &juniauto::bayes::GroupedRidge::sigma_total,
             py::arg("x_predict_variance"),
             py::arg("sigma_quote_stale_sq"),
             py::arg("sigma_liquidity_sq"),
             py::arg("sigma_order_state_sq"),
             py::arg("sigma_regime_shift_sq"))
        .def("conservative_edge", &juniauto::bayes::GroupedRidge::conservative_edge,
             py::arg("mu_edge"), py::arg("sigma_total"))
        .def("group_posterior", &juniauto::bayes::GroupedRidge::group_posterior,
             py::arg("group"), py::return_value_policy::reference_internal)
        .def("beta_mean", &juniauto::bayes::GroupedRidge::beta_mean)
        .def("beta_cov", &juniauto::bayes::GroupedRidge::beta_cov);

    // ---------------- Costs ----------------
    py::class_<juniauto::costs::CostConfig>(m, "CostConfig")
        .def(py::init<>());  // fields exposed via named args on the Python side if needed

    py::class_<juniauto::costs::MarketState>(m, "MarketState")
        .def(py::init<>())
        .def_readwrite("mid_price", &juniauto::costs::MarketState::mid_price)
        .def_readwrite("spread_bps", &juniauto::costs::MarketState::spread_bps)
        .def_readwrite("volatility_bps", &juniauto::costs::MarketState::volatility_bps)
        .def_readwrite("bar_dollar_volume", &juniauto::costs::MarketState::bar_dollar_volume)
        .def_readwrite("adv_dollar", &juniauto::costs::MarketState::adv_dollar)
        .def_readwrite("quote_age_sessions", &juniauto::costs::MarketState::quote_age_sessions)
        .def_readwrite("gap_days_to_next_session", &juniauto::costs::MarketState::gap_days_to_next_session)
        .def_readwrite("session_multiplier", &juniauto::costs::MarketState::session_multiplier)
        .def_readwrite("adverse_selection_share", &juniauto::costs::MarketState::adverse_selection_share);

    py::class_<juniauto::costs::Order>(m, "Order")
        .def(py::init<>())
        .def_readwrite("symbol", &juniauto::costs::Order::symbol)
        .def_readwrite("notional", &juniauto::costs::Order::notional)
        .def_readwrite("predicted_holding_seconds", &juniauto::costs::Order::predicted_holding_seconds)
        .def_readwrite("has_open_order", &juniauto::costs::Order::has_open_order);

    py::class_<juniauto::costs::SlippageStats>(m, "SlippageStats")
        .def(py::init<>())
        .def_readwrite("recent_fill_slippage_bps", &juniauto::costs::SlippageStats::recent_fill_slippage_bps);

    py::class_<juniauto::costs::CostBreakdown>(m, "CostBreakdown")
        .def_readonly("base_side_bps", &juniauto::costs::CostBreakdown::base_side_bps)
        .def_readonly("entry_bps", &juniauto::costs::CostBreakdown::entry_bps)
        .def_readonly("exit_reserved_bps", &juniauto::costs::CostBreakdown::exit_reserved_bps)
        .def_readonly("queue_delay_bps", &juniauto::costs::CostBreakdown::queue_delay_bps)
        .def_readonly("cancel_replace_bps", &juniauto::costs::CostBreakdown::cancel_replace_bps)
        .def_readonly("action_memory_bps", &juniauto::costs::CostBreakdown::action_memory_bps)
        .def_readonly("cash_waiting_value_bps", &juniauto::costs::CostBreakdown::cash_waiting_value_bps)
        .def_readonly("operational_bps", &juniauto::costs::CostBreakdown::operational_bps)
        .def_readonly("uncertainty_bps", &juniauto::costs::CostBreakdown::uncertainty_bps)
        .def_readonly("opportunity_bps", &juniauto::costs::CostBreakdown::opportunity_bps)
        .def("total", &juniauto::costs::CostBreakdown::total);

    m.def("base_side_cost_bps", &juniauto::costs::base_side_cost_bps,
          py::arg("state"), py::arg("slip"), py::arg("cfg"));
    m.def("compute_cost", &juniauto::costs::compute_cost,
          py::arg("order"), py::arg("state"), py::arg("slip"), py::arg("cfg"), py::arg("model_edge_bps"));

    // ---------------- Gateway ----------------
    py::enum_<juniauto::gateway::Role>(m, "Role")
        .value("Primary", juniauto::gateway::Role::Primary)
        .value("Secondary", juniauto::gateway::Role::Secondary)
        .value("Retained", juniauto::gateway::Role::Retained)
        .value("None_", juniauto::gateway::Role::None);

    py::enum_<juniauto::gateway::ActionType>(m, "ActionType")
        .value("BUY", juniauto::gateway::ActionType::Buy)
        .value("SELL", juniauto::gateway::ActionType::Sell)
        .value("ROTATE", juniauto::gateway::ActionType::Rotate)
        .value("REPLACE", juniauto::gateway::ActionType::Replace)
        .value("CANCEL", juniauto::gateway::ActionType::Cancel)
        .value("HOLD", juniauto::gateway::ActionType::Hold);

    py::class_<juniauto::gateway::GatewayConfig>(m, "GatewayConfig").def(py::init<>());

    py::class_<juniauto::gateway::ActionEvaluation>(m, "ActionEvaluation")
        .def_readonly("symbol", &juniauto::gateway::ActionEvaluation::symbol)
        .def_readonly("action", &juniauto::gateway::ActionEvaluation::action)
        .def_readonly("gross_edge_bps", &juniauto::gateway::ActionEvaluation::gross_edge_bps)
        .def_readonly("total_cost_bps", &juniauto::gateway::ActionEvaluation::total_cost_bps)
        .def_readonly("net_edge_bps", &juniauto::gateway::ActionEvaluation::net_edge_bps)
        .def_readonly("model_edge_bps", &juniauto::gateway::ActionEvaluation::model_edge_bps)
        .def_readonly("friction_multiplier", &juniauto::gateway::ActionEvaluation::friction_multiplier)
        .def_readonly("role", &juniauto::gateway::ActionEvaluation::role)
        .def("executes", &juniauto::gateway::ActionEvaluation::executes);

    m.def("composite_edge", &juniauto::gateway::composite_edge,
          py::arg("after_cost_edge_bps"), py::arg("membership_bps"), py::arg("friction_multiplier"));
    m.def("membership_edge_bps", &juniauto::gateway::membership_edge_bps,
          py::arg("role"), py::arg("cfg"));
    m.def("evaluate_gateway", &juniauto::gateway::evaluate,
          py::arg("symbol"), py::arg("role"), py::arg("after_cost_edge_bps"),
          py::arg("friction_multiplier"), py::arg("state"), py::arg("slippage"),
          py::arg("cost_cfg"), py::arg("gw_cfg"), py::arg("proposed"));

    // ---------------- Portfolio sizing ----------------
    py::class_<juniauto::portfolio::SizingConfig>(m, "SizingConfig").def(py::init<>());
    py::class_<juniauto::portfolio::SizingInput>(m, "SizingInput").def(py::init<>());
    py::class_<juniauto::portfolio::SizingResult>(m, "SizingResult")
        .def_readonly("weight", &juniauto::portfolio::SizingResult::weight)
        .def_readonly("notional", &juniauto::portfolio::SizingResult::notional)
        .def_readonly("shares", &juniauto::portfolio::SizingResult::shares)
        .def_readonly("at_name_cap", &juniauto::portfolio::SizingResult::at_name_cap)
        .def_readonly("at_aggregate_cap", &juniauto::portfolio::SizingResult::at_aggregate_cap)
        .def_readonly("at_cash_floor", &juniauto::portfolio::SizingResult::at_cash_floor);

    m.def("kelly_weight", &juniauto::portfolio::kelly_weight,
          py::arg("edge_bps"), py::arg("variance_bps_sq"));
    m.def("concentration_penalty_bps", &juniauto::portfolio::concentration_penalty_bps,
          py::arg("weights"), py::arg("comfortable_weight"));
    m.def("size_position", &juniauto::portfolio::size_position,
          py::arg("symbol"), py::arg("in_"), py::arg("horizon_weights"), py::arg("cfg"));
}
