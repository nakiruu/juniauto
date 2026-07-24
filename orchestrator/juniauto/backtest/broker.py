"""Simulated broker for backtest — Alpaca-shape API, no network I/O.

Contract: the returned dicts have the SAME KEYS as `AlpacaFeed.get_account`,
`get_positions`, `get_open_orders`, `submit_market`, and `close_position`.
This lets the BacktestEngine reuse the live orchestrator's step-6 order-
routing branch verbatim.

Order settlement model:
    - submit_market(...) records a *pending* order with the decision-time
      reference price and returns an order id.
    - settle(next_ts, bars_provider) drains the pending queue using the
      configured fill_model to compute a fill price from the next day's
      bar (or same-day close, depending on the model), updates cash +
      positions, and appends a fill record.
    - The engine is expected to call settle() between cycles so the next
      cycle's positions view reflects the previous cycle's fills.

Fill models (default `next_open` per coordinator review Q4):
    next_open     : fill at next trading day's open bar
    close         : fill at the same trading day's close bar (OPTIMISTIC —
                    uses information from the very bar that generated the
                    signal; only for sensitivity analysis)
    vwap          : fill at next trading day's vwap
    delayed_mid   : fill at next open + 15 bp haircut on the buy side,
                    15 bp premium on the sell side (models IEX delayed
                    quote uncertainty from the live pipeline)
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Literal

from juniauto.data.alpaca_feed import Bar
from juniauto.utils import get_logger

log = get_logger(__name__)


FillModel = Literal["next_open", "close", "vwap", "delayed_mid"]


@dataclass
class SimBrokerFill:
    """One materialized fill produced by settle()."""
    order_id: str
    ts: datetime                           # bar timestamp used for the fill
    symbol: str
    side: str                              # "buy" | "sell"
    qty: float
    fill_price: float
    decision_ref_price: float              # price at the time submit() was called
    slippage_bps: float                    # 10_000 * (fill - ref) / ref, signed for buy
    fill_model: FillModel


@dataclass
class _PendingOrder:
    order_id: str
    submitted_at: datetime
    symbol: str
    side: str
    qty: float
    decision_ref_price: float
    is_full_exit: bool                     # True when this is a close_position call


@dataclass
class _Position:
    qty: float = 0.0
    avg_entry_price: float = 0.0
    total_cost: float = 0.0                # cumulative cost basis (for avg_entry recompute)


# Bars-provider signature: given (symbol, target_date), return the bar for
# that date if known, else None. The engine wires this to the historical
# bars loader.
BarsProvider = Callable[[str, date], Bar | None]


class SimBroker:
    """In-process broker replacement for AlpacaFeed during backtest."""

    def __init__(
        self,
        initial_cash: float,
        fill_model: FillModel = "next_open",
        bars_provider: BarsProvider | None = None,
    ) -> None:
        self._cash = float(initial_cash)
        self._fill_model = fill_model
        self._bars_provider = bars_provider
        self._positions: dict[str, _Position] = {}
        self._pending: list[_PendingOrder] = []
        self._fills: list[SimBrokerFill] = []
        self._closed_orders: list[str] = []
        # Track for cost-basis-based average entry (weighted average of buys).
        self._last_mark_prices: dict[str, float] = {}

    # ---- Configuration ----
    @property
    def fill_model(self) -> FillModel:
        return self._fill_model

    def set_bars_provider(self, provider: BarsProvider) -> None:
        self._bars_provider = provider

    # ---- Account / positions (Alpaca-shape) ----
    def get_account(self) -> dict[str, Any]:
        # Mark positions using last known reference price; if we've never
        # marked a symbol, assume avg_entry_price. equity = cash + Σ q*p.
        mv = 0.0
        for sym, pos in self._positions.items():
            if pos.qty <= 0:
                continue
            px = self._last_mark_prices.get(sym, pos.avg_entry_price)
            mv += pos.qty * px
        equity = self._cash + mv
        return {
            "equity": equity,
            "cash": self._cash,
            "buying_power": self._cash,           # no margin in the sim
            "day_trade_count": 0,                 # tracked externally by PDTTracker
            "pattern_day_trader": False,
            "portfolio_value": equity,
        }

    def get_positions(self) -> list[dict[str, Any]]:
        out = []
        for sym, pos in self._positions.items():
            if pos.qty <= 0:
                continue
            px = self._last_mark_prices.get(sym, pos.avg_entry_price)
            out.append({
                "symbol": sym,
                "qty": float(pos.qty),
                "qty_available": float(pos.qty),
                "avg_entry_price": float(pos.avg_entry_price),
                "market_value": float(pos.qty * px),
                "unrealized_pl": float(pos.qty * (px - pos.avg_entry_price)),
                "side": "long",
            })
        return out

    def get_open_orders(self) -> list[dict[str, Any]]:
        return [
            {
                "id": o.order_id,
                "symbol": o.symbol,
                "qty": o.qty,
                "side": o.side,
                "type": "market",
                "limit_price": None,
                "submitted_at": o.submitted_at,
            }
            for o in self._pending
        ]

    # ---- Order routing ----
    def submit_market(self, symbol: str, qty: float, side: str) -> str:
        """Queue a market order. Cash is NOT debited until settle() fills it —
        matches the live-broker asynchronous fill semantics closely enough
        for daily-cadence PnL to be correct."""
        if qty <= 0:
            raise ValueError(f"submit_market qty must be > 0, got {qty}")
        oid = f"sim-{uuid.uuid4().hex[:12]}"
        ref = float(self._last_mark_prices.get(symbol, 0.0))
        self._pending.append(_PendingOrder(
            order_id=oid,
            submitted_at=datetime.now(),
            symbol=symbol,
            side=side,
            qty=qty,
            decision_ref_price=ref,
            is_full_exit=False,
        ))
        return oid

    def close_position(self, symbol: str) -> str:
        """Queue a full-exit sell. Distinguished from submit_market so the
        settle path knows to sell exactly the held quantity (avoids the
        4th-decimal rounding issues the live pipeline had)."""
        pos = self._positions.get(symbol)
        if pos is None or pos.qty <= 0:
            raise ValueError(f"close_position: no long position in {symbol}")
        oid = f"sim-close-{uuid.uuid4().hex[:12]}"
        ref = float(self._last_mark_prices.get(symbol, pos.avg_entry_price))
        self._pending.append(_PendingOrder(
            order_id=oid,
            submitted_at=datetime.now(),
            symbol=symbol,
            side="sell",
            qty=pos.qty,
            decision_ref_price=ref,
            is_full_exit=True,
        ))
        return oid

    def cancel_order(self, order_id: str) -> None:
        self._pending = [o for o in self._pending if o.order_id != order_id]

    # ---- Settlement ----
    def settle(self, next_ts: datetime) -> list[SimBrokerFill]:
        """Drain the pending queue using the configured fill model, applying
        each fill to cash + positions. Returns the list of fills produced.

        `next_ts` is the timestamp of the NEXT trading day (or the current
        cycle date, depending on the fill model). For `next_open`, the
        fill uses the open of the bar dated `next_ts.date()`. For `close`,
        the fill uses the close of the bar dated `submitted_at.date()`.
        """
        if self._bars_provider is None:
            raise RuntimeError("SimBroker.set_bars_provider must be called before settle()")

        fills: list[SimBrokerFill] = []
        unfilled: list[_PendingOrder] = []

        for order in self._pending:
            bar_date, fill_px = self._resolve_fill(order, next_ts)
            if fill_px is None:
                # No bar available for the target date — leave pending; the
                # engine can retry on the next settle() call.
                unfilled.append(order)
                continue

            # Apply fill.
            if order.side == "buy":
                cost = order.qty * fill_px
                if cost > self._cash:
                    # Not enough cash — degrade to the largest whole-position
                    # buy that fits. Rare in practice because the sizing step
                    # already respects cash_floor, but insurance is cheap.
                    max_qty = max(0.0, math.floor((self._cash / fill_px) * 10000) / 10000)
                    if max_qty <= 0:
                        log.warning(
                            "sim_buy_cash_short",
                            symbol=order.symbol, need=cost, have=self._cash,
                        )
                        continue
                    order.qty = max_qty
                    cost = order.qty * fill_px
                self._cash -= cost
                pos = self._positions.setdefault(order.symbol, _Position())
                pos.total_cost += cost
                pos.qty += order.qty
                pos.avg_entry_price = pos.total_cost / pos.qty if pos.qty > 0 else 0.0
            else:  # sell
                pos = self._positions.get(order.symbol)
                if pos is None or pos.qty <= 0:
                    log.warning("sim_sell_no_position", symbol=order.symbol)
                    continue
                sell_qty = min(order.qty, pos.qty) if not order.is_full_exit else pos.qty
                proceeds = sell_qty * fill_px
                self._cash += proceeds
                # Reduce basis proportionally.
                if pos.qty > 0:
                    pos.total_cost *= max(0.0, 1.0 - sell_qty / pos.qty)
                pos.qty -= sell_qty
                if pos.qty < 1e-9:
                    # Clean out dust.
                    del self._positions[order.symbol]

            ref = order.decision_ref_price or fill_px
            slippage = 10_000.0 * (fill_px - ref) / ref if ref > 0 else 0.0
            # Sign convention: BUY at higher-than-ref = positive slippage cost.
            # SELL at lower-than-ref = negative slippage (bad for us) → flip sign.
            if order.side == "sell":
                slippage = -slippage
            fill = SimBrokerFill(
                order_id=order.order_id,
                ts=next_ts,
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
                fill_price=fill_px,
                decision_ref_price=ref,
                slippage_bps=slippage,
                fill_model=self._fill_model,
            )
            fills.append(fill)
            self._fills.append(fill)
            self._closed_orders.append(order.order_id)

        self._pending = unfilled
        return fills

    def mark_to_market(self, prices: dict[str, float]) -> None:
        """Update the last-known reference price used for equity/position
        marking. Called by the engine at the end of each cycle."""
        for sym, px in prices.items():
            if px > 0:
                self._last_mark_prices[sym] = float(px)

    # ---- Reporting ----
    def all_fills(self) -> list[SimBrokerFill]:
        return list(self._fills)

    # ---- Internals ----
    def _resolve_fill(self, order: _PendingOrder, next_ts: datetime) -> tuple[date, float | None]:
        assert self._bars_provider is not None
        model = self._fill_model
        submit_d = order.submitted_at.date() if isinstance(order.submitted_at, datetime) else next_ts.date()

        if model == "close":
            target = submit_d
            bar = self._bars_provider(order.symbol, target)
            return target, (float(bar.close) if bar else None)

        # next_open / vwap / delayed_mid all use the NEXT trading day's bar.
        target = next_ts.date()
        bar = self._bars_provider(order.symbol, target)
        if bar is None:
            return target, None

        if model == "next_open":
            return target, float(bar.open)
        if model == "vwap":
            vwap = bar.vwap if bar.vwap is not None and bar.vwap > 0 else bar.open
            return target, float(vwap)
        if model == "delayed_mid":
            # 15 bp haircut on buy (fill higher), premium on sell (fill lower).
            base = float(bar.open)
            bump_bps = 15.0
            if order.side == "buy":
                return target, base * (1.0 + bump_bps / 10_000.0)
            return target, base * (1.0 - bump_bps / 10_000.0)

        raise ValueError(f"Unknown fill_model: {model}")
