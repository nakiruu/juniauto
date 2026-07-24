"""Deterministic trading-day iterator for the backtest engine.

Uses the same pandas_market_calendars NYSE schedule as the live pipeline,
so bar-count semantics between backtest and live are identical. All
timestamps are timezone-aware ET.

Design contract:
    - `now()` is stable across calls until `advance()` is invoked.
    - Each cycle timestamp is set to 15:55 ET (matching the live
      `model.decision_time_et` default) so downstream code sees the
      same time-of-day it would see in production.
    - Phantom cycle times (09:40, 12:30) are exposed via
      `phantom_iter()` for the engine's optional multi-cycle-per-day
      mode; the base iterator emits only the 15:55 tick.
"""
from __future__ import annotations

from datetime import date, datetime, time
from typing import Iterator

import pandas_market_calendars as mcal

from juniauto.utils.time_utils import ET


_NYSE = mcal.get_calendar("NYSE")
_LIVE_DECISION_TIME = time(15, 55)


class SimClock:
    """Trading-day iterator scoped to [start_date, end_date].

    Trading days come from the NYSE schedule (respects holidays + early
    closes). `now()` returns the current cycle timestamp at 15:55 ET;
    `advance()` moves to the next trading day and returns True if one
    exists, False when the window is exhausted.

    Usage:
        clock = SimClock(date(2024, 1, 2), date(2026, 7, 1))
        while True:
            engine.run_cycle(clock.now())
            if not clock.advance():
                break
    """

    def __init__(
        self,
        start_date: date,
        end_date: date,
        decision_time: time = _LIVE_DECISION_TIME,
    ) -> None:
        if end_date < start_date:
            raise ValueError(f"end_date {end_date} < start_date {start_date}")
        self._decision_time = decision_time
        sched = _NYSE.schedule(start_date=start_date, end_date=end_date)
        # sched.index is DatetimeIndex of session open timestamps (UTC).
        # Convert to date-only ET so we anchor at decision_time cleanly.
        self._days: list[date] = [ts.tz_convert(ET).date() for ts in sched.index]
        if not self._days:
            raise ValueError(
                f"No trading days between {start_date} and {end_date}. "
                "Widen the window or check calendar (weekends only?)"
            )
        self._i = 0

    # ---- Cursor state ----
    def now(self) -> datetime:
        return datetime.combine(self._days[self._i], self._decision_time, tzinfo=ET)

    def today(self) -> date:
        return self._days[self._i]

    def advance(self) -> bool:
        """Move to the next trading day. Return True on success, False when
        the window is exhausted (further advance() calls are no-ops)."""
        if self._i + 1 >= len(self._days):
            return False
        self._i += 1
        return True

    def reset(self) -> None:
        self._i = 0

    # ---- Introspection ----
    def __len__(self) -> int:
        return len(self._days)

    def remaining(self) -> int:
        return len(self._days) - self._i - 1

    def index(self) -> int:
        return self._i

    def days(self) -> list[date]:
        """Return the full list of trading days in the window (immutable copy)."""
        return list(self._days)

    def previous_day(self) -> date | None:
        return self._days[self._i - 1] if self._i > 0 else None

    def next_day(self) -> date | None:
        return self._days[self._i + 1] if self._i + 1 < len(self._days) else None

    # ---- Phantom cycle helper (optional multi-cycle-per-day mode) ----
    def phantom_iter(self, phantom_times: list[time]) -> Iterator[tuple[str, datetime]]:
        """Emit (cycle_label, timestamp) for the phantom cycles on the current
        day, e.g. ("0940", 09:40 ET), ("1230", 12:30 ET). Used when the
        engine is configured to simulate the live phantom-cadence pattern."""
        today = self._days[self._i]
        for t in phantom_times:
            label = f"{t.hour:02d}{t.minute:02d}"
            yield label, datetime.combine(today, t, tzinfo=ET)
