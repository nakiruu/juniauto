"""Liquidity and execution-quality signal family computer.

Computes four features from bar and quote data.
Output columns match `features` table §1.4.5:
    spread_bps, dollar_volume, relative_volume, depth_proxy.

See docs/knowledge-base/part1-data-preparation.md §1.4.5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from juniauto.data.alpaca_feed import Quote
from juniauto.signals._norm import _normalize
from juniauto.utils import get_logger

log = get_logger(__name__)

# Lookback for volume averages (trading days)
_VOL_WINDOW = 20  # average daily volume window; §1.4.5


class LiquiditySignals:
    """Compute liquidity/execution-quality signal family.

    Consumes the latest Quote per symbol for spread measurement and daily
    bar history for volume features. Missing quotes produce NaN for
    spread_bps and depth_proxy; that is correct behavior per §1.7.
    See docs/knowledge-base/part1-data-preparation.md §1.4.5.
    """

    def compute(
        self,
        bars: pd.DataFrame,
        quotes: dict[str, Quote] | None = None,
    ) -> pd.DataFrame:
        """Compute liquidity features for each symbol.

        Args:
            bars: Daily bar DataFrame with columns
                [symbol, ts, open, high, low, close, volume, vwap].
                Rows must be strictly before the decision timestamp.
            quotes: Latest Quote snapshot per symbol (from AlpacaFeed).
                Optional; missing symbols get NaN for spread_bps / depth_proxy.

        Returns:
            DataFrame indexed by symbol with columns:
            spread_bps, dollar_volume, relative_volume, depth_proxy.
        """
        if bars.empty:
            return pd.DataFrame()

        quotes = quotes or {}
        rows: list[dict[str, object]] = []

        for sym, grp in bars.sort_values("ts").groupby("symbol", sort=False):
            grp = grp.sort_values("ts").reset_index(drop=True)
            close = grp["close"].astype(float)
            vol = grp["volume"].astype(float)
            n = len(grp)

            row: dict[str, object] = {"symbol": sym}

            # spread_bps: from latest Quote.
            # §1.4.5 "bid, ask, spread"
            q = quotes.get(str(sym))
            if q is not None and q.bid > 0 and q.ask > 0:
                row["spread_bps"] = q.spread_bps
            else:
                row["spread_bps"] = float("nan")

            # dollar_volume: mean(close * volume) over last _VOL_WINDOW days.
            # §1.4.5 "dollar volume"
            if n >= _VOL_WINDOW:
                dv = (close.iloc[-_VOL_WINDOW:] * vol.iloc[-_VOL_WINDOW:]).mean()
                row["dollar_volume"] = float(dv)
            elif n >= 1:
                row["dollar_volume"] = float((close * vol).mean())
            else:
                row["dollar_volume"] = float("nan")

            # relative_volume: today's volume vs _VOL_WINDOW average.
            # §1.4.5 "relative volume"
            if n >= _VOL_WINDOW + 1:
                avg_vol = vol.iloc[-_VOL_WINDOW - 1 : -1].mean()  # exclude last day
                if avg_vol > 0:
                    row["relative_volume"] = float(vol.iloc[-1] / avg_vol)
                else:
                    row["relative_volume"] = float("nan")
            else:
                row["relative_volume"] = float("nan")

            # depth_proxy: bid_size + ask_size from latest quote, normalized
            # by average daily share volume. Approximates order book depth.
            # §1.4.5 "market depth proxy"
            if q is not None and n >= 1:
                total_size = float(q.bid_size + q.ask_size)
                avg_vol_all = float(vol.mean())
                if avg_vol_all > 0:
                    row["depth_proxy"] = total_size / avg_vol_all
                else:
                    row["depth_proxy"] = float("nan")
            else:
                row["depth_proxy"] = float("nan")

            rows.append(row)

        if not rows:
            return pd.DataFrame()

        out = pd.DataFrame(rows).set_index("symbol")

        # Cross-sectional normalization per §1.2.
        # Note: spread_bps is a cost — higher is worse. After normalization
        # a positive z-score = wider-than-median spread. Caller / cost model
        # interprets the sign appropriately.
        feature_cols = ["spread_bps", "dollar_volume", "relative_volume", "depth_proxy"]
        for col in feature_cols:
            if col in out.columns:
                out[col] = _normalize(out, col)

        log.debug("liquidity_signals_computed", n_symbols=len(out))
        return out
