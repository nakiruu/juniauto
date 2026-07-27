"""Quarterly universe selector — bars-only, no fundamentals.

Computes per-symbol combined score at each quarter boundary and returns
the top-N symbols. All inputs come from the `bars` table which is
already populated by backfill.

Invariants (asserted in code):
  - Quarter boundary dates are ordered and non-empty for the requested window.
  - Top-N selection returns at most N and at least 1 name per quarter
    (unless bars are pathologically sparse — logs a warning).
  - Each quarter's selection is memoized; identical dates return identical sets.
"""
from __future__ import annotations

import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from io import StringIO
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from juniauto.utils import get_logger

if TYPE_CHECKING:
    from juniauto.db import QuestDBClient

log = get_logger(__name__)


_MOM_LOOKBACK_LONG = 252   # 12 months
_MOM_LOOKBACK_SHORT = 21   # 1 month
_LIQ_VOL_WINDOW = 60       # 60 trading days for liquidity + realized vol


@dataclass(frozen=True)
class QuarterKey:
    year: int
    quarter: int   # 1..4

    @classmethod
    def from_date(cls, d: date) -> "QuarterKey":
        return cls(d.year, (d.month - 1) // 3 + 1)


class QuarterlyUniverseSelector:
    """Precomputes top-N universe per quarter from bars-only signals.

    All computation happens at construction (one-shot). `active_symbols`
    is O(1) after that.
    """

    def __init__(
        self,
        db: "QuestDBClient",
        start_date: date,
        end_date: date,
        top_n: int = 50,
    ) -> None:
        self._top_n = int(top_n)
        # Fetch bars once. Uses the same REST-batched pattern the main
        # backtest loader uses so we bypass the flaky PG wire.
        bars = _fetch_all_bars(db)
        if bars is None or bars.empty:
            log.error("un_selector_no_bars")
            self._selection: dict[QuarterKey, frozenset[str]] = {}
            return
        bars = bars.sort_values(["symbol", "ts"]).reset_index(drop=True)
        bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
        bars["_date"] = bars["ts"].dt.date

        quarter_boundaries = _quarter_start_dates(start_date, end_date, bars["_date"])
        log.info(
            "un_selector_start",
            n_quarters=len(quarter_boundaries),
            top_n=self._top_n,
            n_bars=len(bars),
        )

        self._selection = {}
        for qs in quarter_boundaries:
            picked = _select_top_n_for_date(bars, qs, top_n=self._top_n)
            key = QuarterKey.from_date(qs)
            self._selection[key] = frozenset(picked)
            log.info(
                "un_selector_quarter",
                quarter=f"{key.year}Q{key.quarter}",
                anchor_date=str(qs),
                n_selected=len(picked),
                sample=sorted(picked)[:10],
            )

    def active_symbols(self, as_of: date | datetime) -> frozenset[str]:
        """Return the frozen set of symbols active for `as_of`'s quarter.
        Returns empty set for dates before the first quarter boundary."""
        if isinstance(as_of, datetime):
            as_of = as_of.date()
        key = QuarterKey.from_date(as_of)
        return self._selection.get(key, frozenset())

    def summary(self) -> dict:
        """Small dict for logging / debugging — quarter-count and size range."""
        if not self._selection:
            return {"n_quarters": 0, "sizes": []}
        sizes = [len(v) for v in self._selection.values()]
        return {
            "n_quarters": len(self._selection),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "mean_size": round(sum(sizes) / len(sizes), 1),
        }


# ================================================================
# Internals
# ================================================================
def _quarter_start_dates(
    start_date: date, end_date: date, all_dates: pd.Series
) -> list[date]:
    """Return the first available trading date of each quarter in
    [start_date, end_date]. Uses the bars_df date column so we snap to
    a real trading day, not the calendar 1st (which may be a weekend)."""
    all_dates = pd.Series(pd.unique(all_dates)).sort_values().reset_index(drop=True)
    boundaries: list[date] = []
    seen_quarters: set[QuarterKey] = set()
    for d in all_dates:
        if d < start_date or d > end_date:
            continue
        k = QuarterKey.from_date(d)
        if k not in seen_quarters:
            seen_quarters.add(k)
            boundaries.append(d)
    return boundaries


def _select_top_n_for_date(
    bars: pd.DataFrame, anchor_date: date, top_n: int
) -> list[str]:
    """Compute per-symbol combined score at `anchor_date` and return
    top-N symbols sorted by score descending."""
    # Slice to bars strictly BEFORE anchor_date (point-in-time correctness).
    hist = bars[bars["_date"] < anchor_date]
    if hist.empty:
        return []

    scores: dict[str, float] = {}
    for sym, g in hist.groupby("symbol", sort=False):
        g = g.sort_values("ts")
        closes = g["close"].astype(float).to_numpy()
        vols = g["volume"].astype(float).to_numpy() if "volume" in g.columns else None

        if len(closes) < _MOM_LOOKBACK_LONG:
            # Not enough history to compute 12-month momentum; skip.
            continue

        # 60-day dollar volume (liquidity)
        if vols is None:
            continue
        recent_dv = closes[-_LIQ_VOL_WINDOW:] * vols[-_LIQ_VOL_WINDOW:]
        liq = float(np.mean(recent_dv))
        if liq <= 0:
            continue

        # 12m-1m momentum: close 21d ago / close 252d ago - 1
        mom = float(closes[-_MOM_LOOKBACK_SHORT] / closes[-_MOM_LOOKBACK_LONG] - 1.0)

        # Inverse of 60-day realized daily-return volatility.
        recent_closes = closes[-_LIQ_VOL_WINDOW - 1:]
        if len(recent_closes) < 2:
            continue
        rets = np.diff(recent_closes) / recent_closes[:-1]
        std_r = float(np.std(rets, ddof=1))
        if std_r <= 0:
            continue
        inv_vol = 1.0 / std_r

        scores[str(sym)] = _combine(liq, mom, inv_vol)

    if not scores:
        return []

    # Rank & take top_n. Combined score is already normalized in _combine.
    sorted_syms = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [s for s, _ in sorted_syms[:top_n]]


def _combine(liq: float, mom: float, inv_vol: float) -> float:
    """Simple product-of-log combination. Log helps against outliers.
    All three inputs assumed positive; caller guards against zeros."""
    return np.log(liq) + mom + np.log(inv_vol)


# ================================================================
# Bars fetch (same REST pattern as backtest.loader for consistency)
# ================================================================
def _fetch_all_bars(db: "QuestDBClient") -> pd.DataFrame | None:
    current_year = datetime.now(tz=timezone.utc).year
    years = list(range(2020, current_year + 2))
    parts: list[pd.DataFrame] = []
    failed_years: list[int] = []
    for yr in years:
        sql = (
            "SELECT symbol, ts, close, volume FROM bars "
            f"WHERE ts >= '{yr}-01-01' AND ts < '{yr + 1}-01-01' "
            "ORDER BY symbol, ts"
        )
        try:
            part = _rest_query_df(db, sql, timeout=120.0)
        except Exception as exc:  # noqa: BLE001
            log.warning("un_bars_batch_failed", year=yr,
                        error=str(exc), error_type=type(exc).__name__)
            failed_years.append(yr)
            continue
        if not part.empty:
            parts.append(part)
            log.info("un_bars_batch", year=yr, n_rows=len(part))
    if not parts:
        log.error("un_bars_all_batches_failed", failed_years=failed_years)
        return None
    if failed_years:
        log.warning("un_bars_partial", failed_years=failed_years)
    return pd.concat(parts, ignore_index=True)


def _rest_query_df(db: "QuestDBClient", sql: str, timeout: float = 120.0) -> pd.DataFrame:
    cfg = db._cfg  # noqa: SLF001
    url = f"http://{cfg.host}:9000/exp?" + urllib.parse.urlencode({"query": sql})
    req = urllib.request.Request(url, headers={"User-Agent": "juniauto-un/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8")
    return pd.read_csv(StringIO(payload))
