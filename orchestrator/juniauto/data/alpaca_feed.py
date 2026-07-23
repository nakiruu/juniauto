"""Alpaca REST feed — IEX 15-minute delayed data + account/positions/orders.

Uses the `alpaca-py` SDK. The IEX feed is the free tier and is the ONLY feed we
target — per PRINCIPLESLONG.md §2.27, the 15-minute delay is a first-class assumption
throughout the cost model, so we never pretend to have SIP.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
)

from juniauto.config import AlpacaConfig
from juniauto.utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Bar:
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float | None
    trade_count: int | None


@dataclass(frozen=True, slots=True)
class Quote:
    symbol: str
    ts: datetime
    bid: float
    ask: float
    bid_size: int
    ask_size: int

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2 if self.bid > 0 and self.ask > 0 else 0.0

    @property
    def spread_bps(self) -> float:
        m = self.mid
        if m <= 0:
            return 0.0
        return 10_000 * (self.ask - self.bid) / m


class AlpacaFeed:
    """Thin wrapper: strong types out, batch-friendly, no retries here (see caller)."""

    def __init__(self, cfg: AlpacaConfig) -> None:
        self._data = StockHistoricalDataClient(cfg.api_key, cfg.secret_key)
        self._trading = TradingClient(cfg.api_key, cfg.secret_key, paper=cfg.paper)
        self._feed = cfg.feed  # "iex"

    # ---- Market data ----
    def get_bars(
        self,
        symbols: list[str],
        *,
        days: int = 252,
        timeframe: TimeFrame = TimeFrame.Day,
    ) -> dict[str, list[Bar]]:
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=datetime.utcnow() - timedelta(days=int(days * 1.5)),  # buffer for holidays
            end=datetime.utcnow(),
            feed=self._feed,  # type: ignore[arg-type]
            adjustment="split",
        )
        resp = self._data.get_stock_bars(req)
        out: dict[str, list[Bar]] = {s: [] for s in symbols}
        for sym, bars in resp.data.items():
            for b in bars:
                out[sym].append(
                    Bar(
                        symbol=sym,
                        ts=b.timestamp,
                        open=float(b.open),
                        high=float(b.high),
                        low=float(b.low),
                        close=float(b.close),
                        volume=int(b.volume),
                        vwap=float(b.vwap) if b.vwap is not None else None,
                        trade_count=int(b.trade_count) if b.trade_count is not None else None,
                    )
                )
        return out

    def get_latest_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        req = StockLatestQuoteRequest(symbol_or_symbols=symbols, feed=self._feed)  # type: ignore[arg-type]
        resp = self._data.get_stock_latest_quote(req)
        out: dict[str, Quote] = {}
        for sym, q in resp.items():
            out[sym] = Quote(
                symbol=sym,
                ts=q.timestamp,
                bid=float(q.bid_price or 0),
                ask=float(q.ask_price or 0),
                bid_size=int(q.bid_size or 0),
                ask_size=int(q.ask_size or 0),
            )
        return out

    # ---- Account ----
    def get_account(self) -> dict[str, Any]:
        acct = self._trading.get_account()
        return {
            "equity": float(acct.equity or 0),           # type: ignore[union-attr]
            "cash": float(acct.cash or 0),               # type: ignore[union-attr]
            "buying_power": float(acct.buying_power or 0),  # type: ignore[union-attr]
            "day_trade_count": int(acct.daytrade_count or 0),  # type: ignore[union-attr]
            "pattern_day_trader": bool(acct.pattern_day_trader),  # type: ignore[union-attr]
            "portfolio_value": float(acct.portfolio_value or 0),  # type: ignore[union-attr]
        }

    def get_positions(self) -> list[dict[str, Any]]:
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "side": p.side.value if hasattr(p.side, "value") else str(p.side),
            }
            for p in self._trading.get_all_positions()  # type: ignore[union-attr]
        ]

    def get_open_orders(self) -> list[dict[str, Any]]:
        req = GetOrdersRequest(status="open")  # type: ignore[arg-type]
        return [
            {
                "id": str(o.id),
                "symbol": o.symbol,
                "qty": float(o.qty or 0),
                "side": o.side.value if hasattr(o.side, "value") else str(o.side),
                "type": o.order_type.value if hasattr(o.order_type, "value") else str(o.order_type),
                "limit_price": float(o.limit_price) if o.limit_price else None,
                "submitted_at": o.submitted_at,
            }
            for o in self._trading.get_orders(filter=req)  # type: ignore[union-attr]
        ]

    # ---- Order routing (§3.3) ----
    def submit_market(self, symbol: str, qty: float, side: str) -> str:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = self._trading.submit_order(req)
        log.info("order_submit_market", symbol=symbol, qty=qty, side=side, id=str(order.id))  # type: ignore[union-attr]
        return str(order.id)  # type: ignore[union-attr]

    def submit_limit(self, symbol: str, qty: float, side: str, limit_price: float) -> str:
        req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
        )
        order = self._trading.submit_order(req)
        log.info(
            "order_submit_limit",
            symbol=symbol, qty=qty, side=side, limit=limit_price, id=str(order.id),  # type: ignore[union-attr]
        )
        return str(order.id)  # type: ignore[union-attr]

    def cancel_order(self, order_id: str) -> None:
        self._trading.cancel_order_by_id(order_id)
        log.info("order_cancel", id=order_id)

    # ---- Clock / calendar ----
    def get_clock(self) -> dict[str, Any]:
        c = self._trading.get_clock()
        return {"is_open": bool(c.is_open), "next_open": c.next_open, "next_close": c.next_close}  # type: ignore[union-attr]
