"""Unit tests for the PDT tracker — the hardest binding constraint in the system."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from juniauto.execution import DayTrade, PDTTracker
from juniauto.utils.time_utils import ET


def _et(y: int, m: int, d: int, h: int = 10, mi: int = 0) -> datetime:
    return datetime(y, m, d, h, mi, tzinfo=ET)


def test_open_close_same_day_is_day_trade() -> None:
    p = PDTTracker()
    p.note_open("AAPL", _et(2026, 7, 20, 9, 35))
    trade = p.note_close("AAPL", _et(2026, 7, 20, 9, 35), _et(2026, 7, 20, 15, 45))
    assert isinstance(trade, DayTrade)
    assert p.count_in_window(_et(2026, 7, 20, 16)) == 1


def test_open_next_day_is_not_day_trade() -> None:
    p = PDTTracker()
    p.note_open("MSFT", _et(2026, 7, 20, 10))
    trade = p.note_close("MSFT", _et(2026, 7, 20, 10), _et(2026, 7, 21, 10))
    assert trade is None
    assert p.count_in_window() == 0


def test_min_hold_blocks_same_day_close() -> None:
    p = PDTTracker()
    p.note_open("GOOG", _et(2026, 7, 20, 9, 45))
    assert not p.min_hold_satisfied("GOOG", _et(2026, 7, 20, 15, 30))
    assert p.min_hold_satisfied("GOOG", _et(2026, 7, 21, 9, 30))


def test_max_three_in_rolling_five_days() -> None:
    p = PDTTracker()
    # simulate 3 day trades on consecutive trading days
    days = [_et(2026, 7, 20), _et(2026, 7, 21), _et(2026, 7, 22)]
    for i, d in enumerate(days):
        sym = f"SYM{i}"
        p.note_open(sym, d)
        p.note_close(sym, d, d + timedelta(hours=5))
    # 3 day trades in window → at cap; next same-day close should be blocked
    p.note_open("BLOCK", _et(2026, 7, 23, 10))
    assert not p.can_close_today("BLOCK", _et(2026, 7, 23, 15))


def test_window_slides_out_after_five_trading_days() -> None:
    p = PDTTracker()
    early = _et(2026, 7, 10, 10)  # far in the past
    p._trades.append(DayTrade("OLD", early, early + timedelta(hours=5)))
    # anchor 2026-07-23 (10 calendar days later, well over 5 trading days)
    assert p.count_in_window(_et(2026, 7, 23, 16)) == 0
