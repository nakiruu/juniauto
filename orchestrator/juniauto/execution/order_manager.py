"""Order manager — bridges gateway `ActionEvaluation` → Alpaca REST → PDT tracker → DB.

Order routing rule (§3.3): use marketable when `EV_market > max(EV_limit, ...) + hurdle`;
otherwise place a limit at the model's `L*`. This module wraps that final routing plus
all state bookkeeping: PDT counter, open orders, execution telemetry.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from juniauto.data.alpaca_feed import AlpacaFeed
from juniauto.db import QuestDBClient
from juniauto.execution.pdt import PDTTracker
from juniauto.utils import get_logger
from juniauto.utils.time_utils import to_et

log = get_logger(__name__)

Side = Literal["buy", "sell"]


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Result of routing an action to a concrete order (or a rejection)."""
    symbol: str
    action_type: str
    executed: bool
    order_id: str | None
    limit_price: float | None
    reject_reason: str | None


class OrderManager:
    """Owns the last-mile of trade execution: PDT gate → order type → submit → record."""

    def __init__(self, alpaca: AlpacaFeed, db: QuestDBClient, pdt: PDTTracker) -> None:
        self._alpaca = alpaca
        self._db = db
        self._pdt = pdt

    def route(
        self,
        *,
        symbol: str,
        action_type: str,        # BUY | SELL | ROTATE | REPLACE | CANCEL
        side: Side,
        qty: float,
        model_edge_bps: float,
        decision_ref_price: float,
        limit_price: float | None,
        horizon: str,
        now: datetime,
    ) -> RoutingDecision:
        # ---- PDT gate (§3.1) ----
        if action_type in ("SELL", "ROTATE"):
            if not self._pdt.can_close_today(symbol, now):
                return self._reject(symbol, action_type, "pdt_day_trade_cap")
            if not self._pdt.min_hold_satisfied(symbol, now):
                return self._reject(symbol, action_type, "pdt_min_hold")

        # ---- Order type (§3.3) ----
        try:
            order_id: str
            if action_type == "CANCEL":
                # caller supplies the order id via the `symbol` slot when action_type==CANCEL
                self._alpaca.cancel_order(symbol)
                order_id = symbol
            elif limit_price is not None:
                order_id = self._alpaca.submit_limit(symbol, qty, side, limit_price)
            else:
                order_id = self._alpaca.submit_market(symbol, qty, side)
        except Exception as e:  # noqa: BLE001 — record + surface, don't kill the loop
            log.error("submit_order_failed", symbol=symbol, error=str(e))
            return self._reject(symbol, action_type, f"submit_error:{type(e).__name__}")

        # ---- PDT bookkeeping ----
        if action_type in ("BUY", "ROTATE"):
            self._pdt.note_open(symbol, now)
        # (SELL / ROTATE close-side is recorded when the fill lands, via note_close)

        # ---- Telemetry ----
        self._record_execution(
            order_id=order_id,
            symbol=symbol,
            action_type=action_type,
            side=side,
            qty=qty,
            fill_price=limit_price or decision_ref_price,  # updated later at fill
            decision_ref_price=decision_ref_price,
            model_edge_bps=model_edge_bps,
            horizon=horizon,
            ts=now,
        )

        return RoutingDecision(
            symbol=symbol,
            action_type=action_type,
            executed=True,
            order_id=order_id,
            limit_price=limit_price,
            reject_reason=None,
        )

    # ---- Internals ----
    def _reject(self, symbol: str, action_type: str, reason: str) -> RoutingDecision:
        log.warning("order_reject", symbol=symbol, action=action_type, reason=reason)
        return RoutingDecision(
            symbol=symbol,
            action_type=action_type,
            executed=False,
            order_id=None,
            limit_price=None,
            reject_reason=reason,
        )

    def _record_execution(
        self,
        *,
        order_id: str,
        symbol: str,
        action_type: str,
        side: Side,
        qty: float,
        fill_price: float,
        decision_ref_price: float,
        model_edge_bps: float,
        horizon: str,
        ts: datetime,
    ) -> None:
        # decision-time slippage is 0; recomputed at fill time by the resolution loop.
        try:
            with self._db.sender() as s:
                s.row(
                    "executions",
                    symbols={
                        "order_id": order_id,
                        "symbol": symbol,
                        "action_type": action_type,
                        "side": side,
                        "horizon": horizon,
                        "session": "regular",
                    },
                    columns={
                        "qty": qty,
                        "fill_price": fill_price,
                        "decision_ref_price": decision_ref_price,
                        "slippage_bps": 0.0,
                        "spread_bps": 0.0,
                        "market_impact_bps": 0.0,
                        "model_edge_bps": model_edge_bps,
                        "realized_return_bps": 0.0,
                        "day_trade": False,
                    },
                    at=to_et(ts),
                )
        except Exception as e:  # noqa: BLE001
            log.error("record_execution_failed", order_id=order_id, error=str(e))
