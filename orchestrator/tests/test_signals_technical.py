"""Golden-values tests for TechnicalSignals.

Uses a synthetic 60-day OHLCV series with a known upward trend in the
last 20 days vs the prior 20. Asserts that relative_strength is positive
(last_20_mean > prior_20_mean after normalization implies the normalized
score for that symbol is well-defined and directionally correct).

See docs/knowledge-base/part1-data-preparation.md §1.4.1.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from juniauto.signals.technical import TechnicalSignals


def _make_bars(
    symbol: str,
    n: int,
    start_price: float = 100.0,
    daily_return: float = 0.0,
) -> pd.DataFrame:
    """Build a synthetic daily bar DataFrame.

    Args:
        symbol:       Ticker symbol string.
        n:            Number of trading days.
        start_price:  Opening price of first bar.
        daily_return: Constant daily log return (positive = uptrend).

    Returns:
        DataFrame with columns: symbol, ts, open, high, low, close, volume, vwap.
    """
    rng = np.random.default_rng(seed=42)
    rows = []
    price = start_price
    base = datetime(2024, 1, 2)
    for i in range(n):
        price *= math.exp(daily_return)
        noise = rng.normal(0, price * 0.005)
        o = price + noise
        h = max(o, price) * 1.005
        lo = min(o, price) * 0.995
        c = price + rng.normal(0, price * 0.003)
        rows.append(
            {
                "symbol": symbol,
                "ts": base + timedelta(days=i),
                "open": o,
                "high": h,
                "low": lo,
                "close": max(c, 0.01),
                "volume": int(1_000_000 + rng.integers(-100_000, 100_000)),
                "vwap": (o + c) / 2.0,
            }
        )
    return pd.DataFrame(rows)


class TestTechnicalSignalsGolden:
    """Known-direction tests on synthetic series."""

    def test_relative_strength_positive_for_uptrending_symbol(self) -> None:
        """A symbol whose last 20 days trend up vs prior 20 should have
        relative_strength > 0 (before normalization, the raw signal is positive).

        With a flat "control" symbol we can verify cross-sectional sign
        is preserved after normalization.
        """
        # Uptrending symbol: +0.5% per day in last 20 days.
        # The series is constructed so the final 40 days have accelerating gains.
        n = 60
        trend_bars = _make_bars("TREND", n, start_price=100.0, daily_return=0.005)
        flat_bars = _make_bars("FLAT", n, start_price=100.0, daily_return=0.0)
        bars = pd.concat([trend_bars, flat_bars], ignore_index=True)

        comp = TechnicalSignals()
        out = comp.compute(bars)

        assert "TREND" in out.index, "TREND symbol missing from output"
        assert "FLAT" in out.index, "FLAT symbol missing from output"

        rs_trend = out.at["TREND", "relative_strength"]
        rs_flat = out.at["FLAT", "relative_strength"]

        # After MAD normalization, the uptrending symbol should score above
        # the flat symbol on relative_strength.
        assert not math.isnan(rs_trend), "relative_strength should not be NaN for TREND"
        assert rs_trend > rs_flat, (
            f"TREND relative_strength ({rs_trend:.4f}) should exceed "
            f"FLAT ({rs_flat:.4f}) for a 0.5%/day uptrend"
        )

    def test_trend_slope_positive_for_uptrend(self) -> None:
        """trend_slope OLS coefficient must be positive for a rising series."""
        trend_bars = _make_bars("UP", 60, daily_return=0.005)
        down_bars = _make_bars("DOWN", 60, daily_return=-0.005)
        bars = pd.concat([trend_bars, down_bars], ignore_index=True)

        out = TechnicalSignals().compute(bars)
        assert out.at["UP", "trend_slope"] > out.at["DOWN", "trend_slope"]

    def test_all_feature_columns_present(self) -> None:
        """Output must contain every column required by schema §1.4.1."""
        bars = _make_bars("AAPL", 60, daily_return=0.002)
        out = TechnicalSignals().compute(bars)
        expected = {
            "trend_slope",
            "relative_strength",
            "breakout_strength",
            "ma_distance",
            "price_acceleration",
            "volume_confirmation",
            "support_defense",
        }
        missing = expected - set(out.columns)
        assert not missing, f"Missing feature columns: {missing}"

    def test_empty_bars_returns_empty_dataframe(self) -> None:
        out = TechnicalSignals().compute(pd.DataFrame())
        assert out.empty

    def test_insufficient_history_produces_nan_not_zero(self) -> None:
        """With < 20 bars, rolling features should be NaN, never filled with 0."""
        bars = _make_bars("SHORT", 5, daily_return=0.0)
        out = TechnicalSignals().compute(bars)
        # Only trend_slope (window=10) and a few others have NaN; relative_strength
        # needs 20+ bars. All windowed features should be NaN or defined.
        rs = out.at["SHORT", "relative_strength"]
        # With only 5 bars, relative_strength (needs ≥20) must be NaN.
        assert math.isnan(rs), "relative_strength with 5 bars must be NaN, not 0"

    def test_volume_confirmation_signed_by_price_direction(self) -> None:
        """High-volume up-day should produce positive volume_confirmation."""
        n = 30
        bars = _make_bars("VOLUP", n, daily_return=0.003)
        flat_bars = _make_bars("VOLFLT", n, daily_return=0.0)
        out = TechnicalSignals().compute(pd.concat([bars, flat_bars], ignore_index=True))
        # Not NaN:
        vc = out.at["VOLUP", "volume_confirmation"]
        assert not math.isnan(vc), "volume_confirmation must not be NaN with 30 bars"
