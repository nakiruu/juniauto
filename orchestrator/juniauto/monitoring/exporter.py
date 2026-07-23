"""Prometheus metrics exporter facade.

Wraps the metric objects defined in `juniauto.monitoring.metrics` (sibling
module) behind small, semantic helpers used by the trading loop. Metric
objects are imported lazily so this file is safe to import before
`metrics.py` lands, and any metrics failure is logged and swallowed —
never break the trading loop over telemetry.

Cross-refs:
- Metric list: docs/knowledge-base/part6-operational.md § 3.6
- Cost components: docs/knowledge-base/part3-costs.md §2.9–2.18
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from juniauto.utils.logging import get_logger

if TYPE_CHECKING:
    from juniauto.config import MetricsConfig

_log = get_logger(__name__)


class MetricsExporter:
    def __init__(self, config: "MetricsConfig") -> None:
        self._config = config
        self._enabled = bool(config.enabled)
        self._port = int(config.port)
        self._interval_seconds = int(config.interval_seconds)
        self._server_started = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def port(self) -> int:
        return self._port

    def start(self) -> None:
        if not self._enabled or self._server_started:
            return
        try:
            from prometheus_client import start_http_server

            start_http_server(self._port)
            self._server_started = True
            _log.info("metrics.exporter.started", port=self._port)
        except Exception as exc:  # noqa: BLE001
            _log.warning("metrics.exporter.start_failed", error=str(exc))

    def push(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        if not self._enabled:
            return
        try:
            from juniauto.monitoring import metrics as _m

            metric: Any | None = getattr(_m, name, None)
            if metric is None:
                _log.warning("metrics.push.unknown_metric", name=name)
                return
            target = metric.labels(**labels) if labels else metric
            if hasattr(target, "set"):
                target.set(float(value))
            elif hasattr(target, "observe"):
                target.observe(float(value))
            elif hasattr(target, "inc"):
                target.inc(float(value))
            else:
                _log.warning("metrics.push.unsupported_metric", name=name)
        except ImportError as exc:
            _log.warning("metrics.push.metrics_module_missing", error=str(exc), name=name)
        except Exception as exc:  # noqa: BLE001
            _log.warning("metrics.push.failed", error=str(exc), name=name)

    def record_action(
        self,
        symbol: str,
        action_type: str,
        executed: bool,
        net_edge_bps: float,
        reject_reason: str | None = None,
    ) -> None:
        if not self._enabled:
            return
        try:
            from juniauto.monitoring import metrics as _m

            _m.net_edge_bps.observe(float(net_edge_bps))
            if executed:
                _m.actions_executed_total.labels(action_type=action_type).inc()
            else:
                _m.actions_rejected_total.labels(
                    action_type=action_type,
                    reason=reject_reason or "unknown",
                ).inc()
        except ImportError as exc:
            _log.warning("metrics.record_action.metrics_missing", error=str(exc))
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "metrics.record_action.failed",
                error=str(exc),
                symbol=symbol,
                action_type=action_type,
            )

    def record_cycle_duration(self, seconds: float, phase: str = "full") -> None:
        if not self._enabled:
            return
        try:
            from juniauto.monitoring import metrics as _m

            _m.decision_cycle_seconds.labels(phase=phase).observe(float(seconds))
        except ImportError as exc:
            _log.warning("metrics.record_cycle_duration.metrics_missing", error=str(exc))
        except Exception as exc:  # noqa: BLE001
            _log.warning("metrics.record_cycle_duration.failed", error=str(exc), phase=phase)

    def record_account(
        self,
        equity: float,
        cash: float,
        day_trade_count: int,
        position_count: int,
    ) -> None:
        if not self._enabled:
            return
        try:
            from juniauto.monitoring import metrics as _m

            _m.equity_gauge.set(float(equity))
            _m.cash_gauge.set(float(cash))
            _m.day_trade_count_gauge.set(float(day_trade_count))
            _m.position_count_gauge.set(float(position_count))
        except ImportError as exc:
            _log.warning("metrics.record_account.metrics_missing", error=str(exc))
        except Exception as exc:  # noqa: BLE001
            _log.warning("metrics.record_account.failed", error=str(exc))

    def record_shadow(
        self,
        horizon: str,
        delta_post: float,
        promotion_ready: bool,
    ) -> None:
        if not self._enabled:
            return
        try:
            from juniauto.monitoring import metrics as _m

            _m.shadow_delta_post_gauge.labels(horizon=horizon).set(float(delta_post))
            _m.shadow_promotion_ready_gauge.labels(horizon=horizon).set(1.0 if promotion_ready else 0.0)
        except ImportError as exc:
            _log.warning("metrics.record_shadow.metrics_missing", error=str(exc))
        except Exception as exc:  # noqa: BLE001
            _log.warning("metrics.record_shadow.failed", error=str(exc), horizon=horizon)
