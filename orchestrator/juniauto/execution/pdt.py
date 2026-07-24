"""PDT rule enforcement — the hardest binding constraint in the whole system.

Per PRINCIPLESLONG.md §3.1 and §2.27:
    - Account equity < $25,000 → Pattern Day Trader rule applies.
    - day_trades_in_rolling_5_trading_day_window ≤ 3.
    - minimum_holding_period = 1 trading day (position opened on day D cannot generate a
      sell for the *same security* on day D — that would be a day trade).

Day-trade definition (SEC): opening and closing the same security on the same calendar day.
The tracker must survive a container restart, so it persists to QuestDB (`day_trades` table).

**Test/paper override:** setting env `PDT_ENFORCE=false` makes both check methods
return True unconditionally. Read per-call so a one-off `docker exec -e
PDT_ENFORCE=false ...` bypasses the gate for that invocation only — no restart
needed. Never set this in a live-money deployment; a bright log warning fires
whenever it's active.
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

from juniauto.utils import get_logger
from juniauto.utils.time_utils import ET, to_et, trading_days_between

log = get_logger(__name__)


def _pdt_enforce_active() -> bool:
    """Env-driven kill switch, read per-call so a one-off `docker exec -e
    PDT_ENFORCE=false ...` takes effect without restarting the container."""
    raw = os.environ.get("PDT_ENFORCE", "true").strip().lower()
    return raw not in ("false", "0", "no", "off")


@dataclass(frozen=True, slots=True)
class DayTrade:
    symbol: str
    open_ts: datetime
    close_ts: datetime

    @property
    def trade_date(self) -> date:
        return to_et(self.close_ts).date()


class PDTTracker:
    """Rolling 5-trading-day window over completed day trades.

    Reload from persistence on startup via `hydrate()`.
    """

    WINDOW_TRADING_DAYS = 5
    MAX_DAY_TRADES = 3

    def __init__(self) -> None:
        self._trades: deque[DayTrade] = deque(maxlen=64)
        self._open_dates: dict[str, date] = {}

    # ---- Persistence ----
    def hydrate(self, prior_trades: Iterable[DayTrade], open_dates: dict[str, date]) -> None:
        for t in prior_trades:
            self._trades.append(t)
        self._open_dates.update(open_dates)

    # ---- Position lifecycle ----
    def note_open(self, symbol: str, ts: datetime) -> None:
        d = to_et(ts).date()
        # First open of the session sets the floor; do not overwrite an earlier same-day open.
        self._open_dates.setdefault(symbol, d)

    def note_close(self, symbol: str, open_ts: datetime, close_ts: datetime) -> DayTrade | None:
        """Return a DayTrade if the close created one, else None."""
        open_d = to_et(open_ts).date()
        close_d = to_et(close_ts).date()
        self._open_dates.pop(symbol, None)
        if open_d != close_d:
            return None
        trade = DayTrade(symbol=symbol, open_ts=open_ts, close_ts=close_ts)
        self._trades.append(trade)
        return trade

    # ---- Queries ----
    def count_in_window(self, as_of: datetime | None = None) -> int:
        """Number of day trades within the rolling 5-*trading-day* window (not calendar days)."""
        anchor = (as_of or datetime.now(tz=ET)).date()
        return sum(
            1
            for t in self._trades
            if trading_days_between(t.trade_date, anchor) < self.WINDOW_TRADING_DAYS
        )

    def can_close_today(self, symbol: str, now: datetime | None = None) -> bool:
        """Would closing `symbol` right now be a day trade, and if so, are we already at the cap?

        Returns True if either (a) closing is not a day trade, or (b) it is but we have room.
        Env override: PDT_ENFORCE=false bypasses the check entirely (paper/testing only).
        """
        if not _pdt_enforce_active():
            log.warning("PDT_BYPASS_ACTIVE", check="can_close_today", symbol=symbol)
            return True
        now_et = to_et(now or datetime.now(tz=ET))
        opened = self._open_dates.get(symbol)
        if opened is None or opened != now_et.date():
            # not a day trade
            return True
        return self.count_in_window(now_et) < self.MAX_DAY_TRADES

    def min_hold_satisfied(self, symbol: str, now: datetime | None = None) -> bool:
        """§3.1: position opened on day D cannot sell same security on day D.
        Env override: PDT_ENFORCE=false bypasses (paper/testing only)."""
        if not _pdt_enforce_active():
            log.warning("PDT_BYPASS_ACTIVE", check="min_hold_satisfied", symbol=symbol)
            return True
        opened = self._open_dates.get(symbol)
        if opened is None:
            return True
        d = to_et(now or datetime.now(tz=ET)).date()
        return trading_days_between(opened, d) >= 1

    # ---- Introspection ----
    def snapshot(self) -> dict[str, object]:
        return {
            "day_trade_count": self.count_in_window(),
            "trades": [
                {"symbol": t.symbol, "open": t.open_ts.isoformat(), "close": t.close_ts.isoformat()}
                for t in self._trades
            ],
            "open_dates": {s: d.isoformat() for s, d in self._open_dates.items()},
        }
