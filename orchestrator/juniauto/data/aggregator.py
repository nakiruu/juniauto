"""Data aggregator — pulls bars + quotes + fundamentals, hands off to signal families.

Owns the freshness weighting logic per §1.3 (halflives in trading days) and the
observation-quality weighting per §1.7. Persists observations to QuestDB via ILP.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

import pandas as pd

from juniauto.config import JuniAutoConfig
from juniauto.data.alpaca_feed import AlpacaFeed, Bar, Quote
from juniauto.data.yahoo_feed import Fundamentals, YahooFeed
from juniauto.db import QuestDBClient
from juniauto.utils import get_logger
from juniauto.utils.time_utils import quote_age_sessions, session_of

log = get_logger(__name__)


@dataclass(slots=True)
class MarketSnapshot:
    ts: datetime
    bars: dict[str, list[Bar]] = field(default_factory=dict)
    quotes: dict[str, Quote] = field(default_factory=dict)
    fundamentals: dict[str, Fundamentals] = field(default_factory=dict)

    def bars_df(self) -> pd.DataFrame:
        rows = []
        for sym, series in self.bars.items():
            for b in series:
                rows.append({
                    "symbol": sym,
                    "ts": b.ts,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                    "vwap": b.vwap,
                    "trade_count": b.trade_count,
                })
        if not rows:
            return pd.DataFrame(
                columns=["symbol", "ts", "open", "high", "low", "close", "volume", "vwap", "trade_count"]
            )
        return pd.DataFrame(rows).sort_values(["symbol", "ts"]).reset_index(drop=True)

    def quotes_df(self) -> pd.DataFrame:
        if not self.quotes:
            return pd.DataFrame(columns=["symbol", "ts", "bid", "ask", "bid_size", "ask_size", "spread_bps", "mid"])
        return pd.DataFrame(
            [
                {
                    "symbol": q.symbol,
                    "ts": q.ts,
                    "bid": q.bid,
                    "ask": q.ask,
                    "bid_size": q.bid_size,
                    "ask_size": q.ask_size,
                    "spread_bps": q.spread_bps,
                    "mid": q.mid,
                }
                for q in self.quotes.values()
            ]
        )


class DataAggregator:
    def __init__(
        self,
        cfg: JuniAutoConfig,
        alpaca: AlpacaFeed,
        yahoo: YahooFeed,
        db: QuestDBClient,
    ) -> None:
        self._cfg = cfg
        self._alpaca = alpaca
        self._yahoo = yahoo
        self._db = db

    # ---- Fetch ----
    def snapshot(self, symbols: Iterable[str], now: datetime) -> MarketSnapshot:
        symbols = list(symbols)
        bars = self._alpaca.get_bars(symbols, days=self._cfg.alpaca.history_bars)
        quotes = self._alpaca.get_latest_quotes(symbols)
        fundamentals = self._yahoo.get_fundamentals(symbols)
        snap = MarketSnapshot(ts=now, bars=bars, quotes=quotes, fundamentals=fundamentals)
        self._persist(snap)
        return snap

    # ---- Persistence via ILP ----
    def _persist(self, snap: MarketSnapshot) -> None:
        try:
            with self._db.sender() as s:
                for sym, series in snap.bars.items():
                    for b in series:
                        s.row(
                            "bars",
                            symbols={"symbol": sym, "session": session_of(b.ts)},
                            columns={
                                "open": b.open,
                                "high": b.high,
                                "low": b.low,
                                "close": b.close,
                                "volume": b.volume,
                                "vwap": b.vwap if b.vwap is not None else 0.0,
                                "trade_count": b.trade_count if b.trade_count is not None else 0,
                            },
                            at=b.ts,
                        )
                for q in snap.quotes.values():
                    s.row(
                        "quotes",
                        symbols={"symbol": q.symbol},
                        columns={
                            "bid": q.bid,
                            "ask": q.ask,
                            "bid_size": q.bid_size,
                            "ask_size": q.ask_size,
                            "quote_age_min": quote_age_sessions(q.ts, snap.ts) * 390.0,
                        },
                        at=q.ts,
                    )
        except Exception as e:  # noqa: BLE001 — never break the trading loop over ingest
            log.error("persist_snapshot_failed", error=str(e))
