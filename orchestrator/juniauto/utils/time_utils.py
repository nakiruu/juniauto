"""Trading-day / session utilities.

PRINCIPLESLONG.md is emphatic that time is measured in **trading sessions and days**,
not wall-clock seconds. This module is the single place that answers:
    - Is `t` inside the regular US equities session?
    - How many trading days between two timestamps?
    - What is the next decision tick at 15:55 ET?
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

ET = ZoneInfo("America/New_York")
NYSE = mcal.get_calendar("NYSE")

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
DECISION_TIME = time(15, 55)
MINUTES_PER_TRADING_DAY = 390  # (§2.24)


def now_et() -> datetime:
    return datetime.now(tz=ET)


def to_et(ts: datetime) -> datetime:
    return ts.astimezone(ET) if ts.tzinfo else ts.replace(tzinfo=ET)


def is_trading_day(d: date) -> bool:
    sched = NYSE.schedule(start_date=d, end_date=d)
    return not sched.empty


def previous_trading_day(d: date) -> date:
    sched = NYSE.schedule(start_date=d - timedelta(days=10), end_date=d - timedelta(days=1))
    return sched.index[-1].date()  # type: ignore[no-any-return]


def next_trading_day(d: date) -> date:
    sched = NYSE.schedule(start_date=d + timedelta(days=1), end_date=d + timedelta(days=10))
    return sched.index[0].date()  # type: ignore[no-any-return]


def trading_days_between(a: date, b: date) -> int:
    if a > b:
        a, b = b, a
    sched = NYSE.schedule(start_date=a, end_date=b)
    return len(sched)


def session_of(ts: datetime) -> str:
    """Classify a timestamp into a trading session bucket used in §2.24 session_multiplier."""
    et = to_et(ts)
    if not is_trading_day(et.date()):
        return "closed"
    t = et.time()
    if t < time(9, 30):
        return "premarket" if t >= time(4, 0) else "closed"
    if t < time(16, 0):
        return "regular"
    if t < time(20, 0):
        return "after_hours"
    return "closed"


def quote_age_sessions(quote_ts: datetime, now: datetime | None = None) -> float:
    """Age in trading sessions (§2.24 stale_quote_risk).

    A same-session quote from today's regular hours reads as ~0.
    A quote from the prior session's close reads as ~1.0.
    """
    n = now or now_et()
    age_min = max(15.0, (to_et(n) - to_et(quote_ts)).total_seconds() / 60.0)
    return age_min / MINUTES_PER_TRADING_DAY
