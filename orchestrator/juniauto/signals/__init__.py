"""Signal family computers for JuniAuto.

Six evidence families per docs/knowledge-base/part1-data-preparation.md §1.4.
Each computer is a pure-function class; all cross-family composition is
handled by compute_all().

Usage:
    from juniauto.signals import compute_all, TechnicalSignals

    wide_df = compute_all(bars=bars_df, fundamentals=fund_dict, quotes=quotes_dict)
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from juniauto.data.alpaca_feed import Quote
from juniauto.data.yahoo_feed import Fundamentals
from juniauto.signals.event import EventSignals
from juniauto.signals.fundamental import FundamentalSignals
from juniauto.signals.liquidity import LiquiditySignals
from juniauto.signals.risk import RiskSignals
from juniauto.signals.semantic import SemanticSignals
from juniauto.signals.technical import TechnicalSignals
from juniauto.utils import get_logger

log = get_logger(__name__)

__all__ = [
    "TechnicalSignals",
    "FundamentalSignals",
    "EventSignals",
    "SemanticSignals",
    "LiquiditySignals",
    "RiskSignals",
    "compute_all",
]


def compute_all(
    bars: pd.DataFrame,
    fundamentals: dict[str, Fundamentals] | None = None,
    quotes: dict[str, Quote] | None = None,
    context_map: dict[str, float] | None = None,
    sector_map: dict[str, float] | None = None,
    as_of_date: date | None = None,
    halflife_event_days: int = 2,
) -> pd.DataFrame:
    """Compose all six signal families into a single wide feature DataFrame.

    Each family is computed independently then joined on the symbol index.
    Missing values from absent families propagate as NaN per §1.7.

    Cross-family normalization is performed within each family (not globally)
    to preserve the semantics of each family's cross-sectional z-score.

    Args:
        bars: Daily OHLCV bar DataFrame (columns: symbol, ts, open, high,
            low, close, volume, vwap). Must include SPY if beta estimation
            is desired. Rows must be strictly before the decision timestamp.
        fundamentals: Optional symbol -> Fundamentals mapping.
        quotes: Optional symbol -> Quote mapping (latest quotes).
        context_map: Optional pre-scored semantic context alignment per symbol.
        sector_map: Optional pre-scored sector context per symbol.
        as_of_date: Decision date for freshness calculations.
        halflife_event_days: Event freshness halflife in trading days (§1.3).

    Returns:
        Wide DataFrame indexed by symbol with all schema feature columns:
        trend_slope, relative_strength, breakout_strength, ma_distance,
        price_acceleration, volume_confirmation, support_defense,
        earnings_quality, revenue_growth, profitability,
        balance_sheet_strength, valuation_quality, analyst_revision,
        catalyst_score, earnings_surprise, guidance_change,
        context_alignment, sector_context,
        spread_bps, dollar_volume, relative_volume, depth_proxy,
        realized_vol_bps, beta, gap_risk, crowding.
    """
    symbols: list[str] = (
        bars["symbol"].unique().tolist() if not bars.empty else []
    )
    # Exclude SPY from the output feature set (it is a reference instrument).
    output_symbols = [s for s in symbols if s != "SPY"]

    fund = fundamentals or {}
    fund_output = {s: v for s, v in fund.items() if s != "SPY"}

    dfs: list[pd.DataFrame] = []

    # --- Technical (§1.4.1) ---
    tech = TechnicalSignals().compute(bars)
    if not tech.empty:
        # Drop SPY from output
        tech = tech[tech.index != "SPY"]
        dfs.append(tech)

    # --- Fundamental (§1.4.2) ---
    if fund_output:
        fund_df = FundamentalSignals().compute(fund_output)
        if not fund_df.empty:
            dfs.append(fund_df)

    # --- Event (§1.4.3) ---
    ev_computer = EventSignals(halflife_days=halflife_event_days)
    event_df = ev_computer.compute(fund_output, as_of_date=as_of_date)
    if not event_df.empty:
        dfs.append(event_df)

    # --- Semantic (§1.4.4) ---
    sem_df = SemanticSignals().compute(
        output_symbols, context_map=context_map, sector_map=sector_map
    )
    if not sem_df.empty:
        dfs.append(sem_df)

    # --- Liquidity (§1.4.5) ---
    liq_df = LiquiditySignals().compute(bars, quotes=quotes)
    if not liq_df.empty:
        liq_df = liq_df[liq_df.index != "SPY"]
        dfs.append(liq_df)

    # --- Risk (§1.4.6) ---
    risk_df = RiskSignals().compute(
        bars, fundamentals=fund, event_features=event_df if not event_df.empty else None
    )
    if not risk_df.empty:
        dfs.append(risk_df)

    if not dfs:
        return pd.DataFrame()

    # Outer join so missing families produce NaN, not dropped rows.
    wide = dfs[0]
    for df in dfs[1:]:
        wide = wide.join(df, how="outer")

    log.info(
        "compute_all_complete",
        n_symbols=len(wide),
        n_features=len(wide.columns),
    )
    return wide
