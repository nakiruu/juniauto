"""Event and catalyst signal family computer.

Catalyst freshness is evaluated in trading days (not minutes) consistent with
the IEX 15-minute delay + 1-day minimum hold. See §1.3 (freshness weighting)
and §1.4.3 (event/catalyst family).

Output columns match `features` table §1.4.3:
    catalyst_score, earnings_surprise, guidance_change.
    gap_days_to_next_earnings is computed but stored as a raw (non-normalized)
    integer-valued feature aligned to the schema column `gap_risk` in risk.py;
    here it is returned as an additional column for informational use.

Freshness halflife = 2 trading days (config key: freshness_halflife_days["event"]).
See docs/knowledge-base/part1-data-preparation.md §1.3, §1.4.3.
"""
from __future__ import annotations

import math
from datetime import date, datetime

import pandas as pd

from juniauto.data.yahoo_feed import Fundamentals
from juniauto.signals._norm import _normalize
from juniauto.utils import get_logger

log = get_logger(__name__)

# Halflife for event catalyst freshness weighting (trading days).
# config key: freshness_halflife_days["event"]; §1.3 IEX/PDT adaptation.
_EVENT_HALFLIFE_TRADING_DAYS: int = 2

_MISSING = float("nan")


def _freshness_weight(age_trading_days: float, halflife: float) -> float:
    """Exponential freshness decay.

    freshness_weight = exp(-age / halflife)
    See docs/knowledge-base/part1-data-preparation.md §1.3.
    """
    if age_trading_days < 0:
        age_trading_days = 0.0
    return math.exp(-age_trading_days / halflife)


class EventSignals:
    """Compute event/catalyst signal family.

    Takes fundamentals (for next_earnings_date and earnings_growth as a
    surprise proxy) and optional age context. Applies freshness weighting
    per §1.3 with a 2-trading-day halflife for events.

    Missing or stale catalyst data returns NaN, not zero.
    See docs/knowledge-base/part1-data-preparation.md §1.4.3.
    """

    def __init__(self, halflife_days: int = _EVENT_HALFLIFE_TRADING_DAYS) -> None:
        # config key: freshness_halflife_days["event"]; §1.3
        self._halflife = float(halflife_days)

    def compute(
        self,
        fundamentals: dict[str, Fundamentals],
        as_of_date: date | None = None,
    ) -> pd.DataFrame:
        """Compute event features for each symbol.

        Args:
            fundamentals: Mapping of symbol -> Fundamentals (from YahooFeed).
                fetched_at is used to compute catalyst age in trading days
                (approximated as calendar days * 5/7 for simplicity; the
                system does not need intraday precision here).
            as_of_date: Decision date. Defaults to today. Must be strictly
                before the decision timestamp (no look-ahead).

        Returns:
            DataFrame indexed by symbol with columns:
            catalyst_score, earnings_surprise, guidance_change,
            gap_days_to_next_earnings.
            Values are freshness-weighted and cross-sectionally normalized.
        """
        if not fundamentals:
            return pd.DataFrame()

        if as_of_date is None:
            as_of_date = datetime.utcnow().date()

        rows: list[dict[str, object]] = []
        for sym, f in fundamentals.items():
            row: dict[str, object] = {"symbol": sym}

            # Age of fundamental data in approximate trading days.
            try:
                fetched_dt = datetime.fromisoformat(f.fetched_at).date()
                age_cal = max(0, (as_of_date - fetched_dt).days)
                # approximate calendar-to-trading-day conversion (5/7)
                age_td = age_cal * (5.0 / 7.0)
            except Exception:
                age_td = float("nan")

            fw = _freshness_weight(age_td, self._halflife) if not math.isnan(age_td) else 0.0

            # catalyst_score: freshness-weighted earnings growth as a general
            # catalyst signal. If data is too stale, the weight collapses to ~0.
            # §1.4.3 "catalyst freshness"
            if f.earnings_growth is not None and fw > 0:
                row["catalyst_score"] = fw * f.earnings_growth
            else:
                row["catalyst_score"] = _MISSING

            # earnings_surprise: earnings_growth as a surprise proxy.
            # TODO: replace with actual EPS surprise from a dedicated source
            # (yfinance does not provide beat/miss magnitude directly).
            # Freshness-weighted.
            # §1.4.3 "earnings surprises"
            if f.earnings_growth is not None and fw > 0:
                row["earnings_surprise"] = fw * f.earnings_growth
            else:
                row["earnings_surprise"] = _MISSING

            # guidance_change: placeholder — yfinance does not expose guidance.
            # Returns NaN so downstream treats as missing data (§1.4.4 note:
            # missing context → NaN, not invented edge).
            # §1.4.3 "guidance changes"
            row["guidance_change"] = _MISSING

            # gap_days_to_next_earnings: calendar days to next earnings date.
            # Used by risk.py for gap_risk; also exposed here for completeness.
            # §1.4.3 "earnings surprise" context.
            if f.next_earnings_date is not None:
                try:
                    ned = date.fromisoformat(f.next_earnings_date)
                    days_to = (ned - as_of_date).days
                    row["gap_days_to_next_earnings"] = float(max(0, days_to))
                except Exception:
                    row["gap_days_to_next_earnings"] = _MISSING
            else:
                row["gap_days_to_next_earnings"] = _MISSING

            rows.append(row)

        if not rows:
            return pd.DataFrame()

        out = pd.DataFrame(rows).set_index("symbol")

        # Normalize freshness-weighted scores cross-sectionally.
        # gap_days_to_next_earnings is a count — raw value passed to risk.py.
        for col in ["catalyst_score", "earnings_surprise"]:
            if col in out.columns:
                out[col] = _normalize(out, col)

        # guidance_change is all-NaN at this placeholder stage; skip normalization.

        log.debug("event_signals_computed", n_symbols=len(out))
        return out
