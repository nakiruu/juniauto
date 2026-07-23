"""Prometheus metric objects for JuniAuto operational monitoring.

Thin wrapper around prometheus_client. Do NOT call start_http_server here;
main.py already does that. This module only defines the metric objects and
exposes them for import by the decision cycle and signal computers.

Metrics align with §3.6 (performance metrics) and §2.41 (shadow monitor).
See docs/knowledge-base/part6-operational.md §3.6.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram


# ---- Account state (§3.1) ----

equity_gauge: Gauge = Gauge(
    "juniauto_equity_dollars",
    "Current account equity in dollars",
)

cash_gauge: Gauge = Gauge(
    "juniauto_cash_dollars",
    "Current cash balance in dollars",
)

day_trade_count_gauge: Gauge = Gauge(
    "juniauto_day_trade_count",
    "Rolling 5-day day-trade count (PDT limit = 3)",
)

position_count_gauge: Gauge = Gauge(
    "juniauto_position_count",
    "Number of open positions",
)


# ---- Execution quality (§3.6) ----

net_edge_hist: Histogram = Histogram(
    "juniauto_net_edge_bps",
    "Net edge in basis points for each executed action",
    buckets=(-500, -200, -100, -50, -20, -10, 0, 10, 20, 50, 100, 200, 500),
)

cost_component_counter: Counter = Counter(
    "juniauto_cost_component_bps_total",
    "Cumulative cost component in basis points by component label",
    labelnames=["component"],
)


# ---- Shadow monitor (§2.41) ----

shadow_delta_gauge: Gauge = Gauge(
    "juniauto_shadow_delta_post",
    "Posterior mean Δnet_bps for the shadow challenger",
    labelnames=["horizon"],
)

shadow_promotion_ready_gauge: Gauge = Gauge(
    "juniauto_shadow_promotion_ready",
    "1 if shadow challenger is promotion-ready, 0 otherwise",
    labelnames=["horizon"],
)


# ---- Decision cycle latency (§3.13) ----

decision_cycle_seconds_hist: Histogram = Histogram(
    "juniauto_decision_cycle_seconds",
    "Wall-clock time of each full decision cycle in seconds",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)


# ---- PDT enforcement (§3.1) ----

pdt_blocked_total: Counter = Counter(
    "juniauto_pdt_blocked_total",
    "Total number of orders blocked by PDT day-trade limit",
)


# ---- Convenience helper ----

class Metrics:
    """Namespace re-export for callers that prefer attribute access.

    Usage:
        from juniauto.monitoring import Metrics
        Metrics.equity_gauge.set(account["equity"])
    """

    equity_gauge = equity_gauge
    cash_gauge = cash_gauge
    day_trade_count_gauge = day_trade_count_gauge
    position_count_gauge = position_count_gauge
    net_edge_hist = net_edge_hist
    cost_component_counter = cost_component_counter
    shadow_delta_gauge = shadow_delta_gauge
    shadow_promotion_ready_gauge = shadow_promotion_ready_gauge
    decision_cycle_seconds_hist = decision_cycle_seconds_hist
    pdt_blocked_total = pdt_blocked_total
