"""Historical backtest engine for JuniAuto.

Event-driven bar-by-bar replay that composes the live decision-cycle
modules (signals, Bayesian, C++ gateway, top-K, weighting) without
touching production Alpaca or QuestDB live tables. All state lives in
`backtest_*` tables keyed by `run_id`.

Entry point:
    python -m juniauto.backtest --start 2024-01-01 --end 2026-07-01 \\
                                --run-id kelly_v3 --fill next_open

See docs/knowledge-base/part6-operational.md and the coordinator design
review notes (2026-07-24) for the full architecture rationale.

Public surface:
    SimClock        -- deterministic trading-day iterator with pandas_market_calendars
    SimBroker       -- Alpaca-shaped simulated broker with configurable fill models
    HistoricalSnapshotLoader
                    -- assembles MarketSnapshot from QuestDB bars point-in-time
    BacktestEngine  -- (commit 2) seven-step decision cycle over a date range
    compute_metrics -- (commit 3) CAPM, Fama-French, Sharpe, drawdown, ...
"""
from juniauto.backtest.benchmarks import (
    benchmark_equal_weight,
    benchmark_fixed5,
    benchmark_spy,
)
from juniauto.backtest.broker import SimBroker, SimBrokerFill
from juniauto.backtest.clock import SimClock
from juniauto.backtest.engine import BacktestEngine
from juniauto.backtest.loader import HistoricalSnapshotLoader
from juniauto.backtest.metrics import MetricRow, compute_and_persist_metrics

__all__ = [
    "SimClock",
    "SimBroker",
    "SimBrokerFill",
    "HistoricalSnapshotLoader",
    "BacktestEngine",
    "MetricRow",
    "compute_and_persist_metrics",
    "benchmark_spy",
    "benchmark_equal_weight",
    "benchmark_fixed5",
]
