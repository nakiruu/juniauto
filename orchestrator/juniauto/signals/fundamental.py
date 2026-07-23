"""Fundamental signal family computer.

Consumes a dict[str, Fundamentals] from yahoo_feed.py and returns
cross-sectionally normalized z-scores for six fundamental dimensions.

Output columns match `features` table §1.4.2:
    earnings_quality, revenue_growth, profitability,
    balance_sheet_strength, valuation_quality, analyst_revision.

Cross-sectional normalization: winsorize at 1st/99th percentile then
MAD-standardize. See docs/knowledge-base/part1-data-preparation.md §1.2.
"""
from __future__ import annotations

import math

import pandas as pd

from juniauto.data.yahoo_feed import Fundamentals
from juniauto.signals._norm import _normalize
from juniauto.utils import get_logger

log = get_logger(__name__)

# sentinel for missing numeric fields — not zero, propagates as NaN downstream
_MISSING = float("nan")


class FundamentalSignals:
    """Compute fundamental signal family from Fundamentals snapshots.

    Missing fields produce NaN for that feature; they are NOT filled with
    neutral defaults. The downstream data_quality weight handles them (§1.7).
    See docs/knowledge-base/part1-data-preparation.md §1.4.2.
    """

    def compute(self, fundamentals: dict[str, Fundamentals]) -> pd.DataFrame:
        """Compute fundamental features for each symbol.

        Args:
            fundamentals: Mapping of symbol -> Fundamentals (from YahooFeed).

        Returns:
            DataFrame indexed by symbol with columns:
            earnings_quality, revenue_growth, profitability,
            balance_sheet_strength, valuation_quality, analyst_revision.
            All values are cross-sectionally MAD-normalized.
        """
        if not fundamentals:
            return pd.DataFrame()

        rows: list[dict[str, object]] = []
        for sym, f in fundamentals.items():
            row: dict[str, object] = {"symbol": sym}

            # earnings_quality: earnings growth quality proxy.
            # Use earnings_growth directly; positive = quality earnings momentum.
            # §1.4.2 "earnings quality"
            row["earnings_quality"] = f.earnings_growth if f.earnings_growth is not None else _MISSING

            # revenue_growth: raw revenue growth rate from yfinance.
            # §1.4.2 "revenue growth"
            row["revenue_growth"] = f.revenue_growth if f.revenue_growth is not None else _MISSING

            # profitability: composite of ROE and profit margins.
            # Average the two if both available; fall back to whichever is present.
            # §1.4.2 "profitability"
            roe = f.return_on_equity
            pm = f.profit_margins
            if roe is not None and pm is not None:
                row["profitability"] = (roe + pm) / 2.0
            elif roe is not None:
                row["profitability"] = roe
            elif pm is not None:
                row["profitability"] = pm
            else:
                row["profitability"] = _MISSING

            # balance_sheet_strength: penalizes high debt; rewards liquidity.
            # score = quick_ratio - debt_to_equity_normalized.
            # Normalize D/E by 100 (typical range 0–200) before differencing.
            # §1.4.2 "balance sheet strength"
            dte = f.debt_to_equity
            qr = f.quick_ratio
            if dte is not None and qr is not None:
                row["balance_sheet_strength"] = qr - (dte / 100.0)
            elif qr is not None:
                row["balance_sheet_strength"] = qr
            elif dte is not None:
                row["balance_sheet_strength"] = -(dte / 100.0)
            else:
                row["balance_sheet_strength"] = _MISSING

            # valuation_quality: lower PEG and P/B = more attractive valuation.
            # Invert sign so higher score = better value.
            # Use negative PEG if available, else negative trailing_pe normalized.
            # §1.4.2 "valuation quality"
            peg = f.peg_ratio
            pe = f.trailing_pe
            if peg is not None and not math.isnan(peg) and peg > 0:
                row["valuation_quality"] = -peg
            elif pe is not None and not math.isnan(pe) and pe > 0:
                row["valuation_quality"] = -math.log(pe)
            else:
                row["valuation_quality"] = _MISSING

            # analyst_revision: placeholder using gross_margins as a proxy for
            # whether analyst revisions tend to be positive (positive margins
            # growth → upward revisions implied).
            # TODO: replace with actual analyst revision direction from a premium
            # data source (yfinance does not expose this reliably).
            # §1.4.2 "analyst revision direction"
            gm = f.gross_margins
            row["analyst_revision"] = gm if gm is not None else _MISSING

            rows.append(row)

        if not rows:
            return pd.DataFrame()

        out = pd.DataFrame(rows).set_index("symbol")

        feature_cols = [
            "earnings_quality",
            "revenue_growth",
            "profitability",
            "balance_sheet_strength",
            "valuation_quality",
            "analyst_revision",
        ]
        for col in feature_cols:
            if col in out.columns:
                out[col] = _normalize(out, col)

        log.debug("fundamental_signals_computed", n_symbols=len(out))
        return out
