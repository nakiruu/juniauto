"""Risk and crowding signal family computer.

Computes four features from daily bar history and fundamentals.
Output columns match `features` table §1.4.6:
    realized_vol_bps, beta, gap_risk, crowding.

Realized volatility uses the Roll (1984) bid-ask bounce correction:
    Var_corrected = max(0, Var(close_returns) - s^2/4)
where s = mean spread in price units (not bps).
See docs/knowledge-base/part2-signals-bayesian.md §2.10 and spec quirk #4.

Beta is estimated via rolling OLS of symbol returns vs SPY returns (252d).
SPY must be included as a symbol in the bars DataFrame (symbol == "SPY").
If SPY is absent, beta falls back to the yfinance beta from Fundamentals.

gap_risk: proximity to earnings date from EventSignals (gap_days_to_next_earnings).
crowding: placeholder — returns NaN. Replace with short-interest data later.

See docs/knowledge-base/part1-data-preparation.md §1.4.6.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from juniauto.data.yahoo_feed import Fundamentals
from juniauto.signals._norm import _normalize
from juniauto.utils import get_logger

log = get_logger(__name__)

# Lookback windows (trading days)
_VOL_WINDOW = 20       # realized vol window; §1.4.6
_BETA_WINDOW = 252     # rolling beta window; §1.4.6
_ANNUALIZE = 252.0     # annualization factor for daily variance
_MISSING = float("nan")

# Basis points conversion
_BPS = 10_000.0


class RiskSignals:
    """Compute risk/crowding signal family.

    Roll (1984) bid-ask bounce correction is applied to realized variance
    before converting to volatility. Under-correcting (i.e., not applying
    this) would inflate all volatility_bps inputs to the cost model.
    See §2.10 and spec quirk #4.
    """

    def compute(
        self,
        bars: pd.DataFrame,
        fundamentals: dict[str, Fundamentals] | None = None,
        event_features: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute risk features for each symbol.

        Args:
            bars: Daily bar DataFrame with columns
                [symbol, ts, open, high, low, close, volume, vwap].
                Include "SPY" rows for beta estimation.
                Rows must be strictly before the decision timestamp.
            fundamentals: Optional; used for beta fallback when SPY absent.
            event_features: Optional; expected to have column
                gap_days_to_next_earnings indexed by symbol (from EventSignals).

        Returns:
            DataFrame indexed by symbol with columns:
            realized_vol_bps, beta, gap_risk, crowding.
        """
        if bars.empty:
            return pd.DataFrame()

        fundamentals = fundamentals or {}

        # Build per-symbol close return series and spread proxy.
        sym_returns: dict[str, pd.Series] = {}
        sym_spread_price: dict[str, float] = {}  # spread in price units (mean high-low proxy)

        for sym, grp in bars.sort_values("ts").groupby("symbol", sort=False):
            grp = grp.sort_values("ts").reset_index(drop=True)
            close = grp["close"].astype(float)
            high = grp["high"].astype(float)
            low = grp["low"].astype(float)
            rets = close.pct_change().dropna()
            sym_returns[str(sym)] = rets

            # Spread proxy in price units: mean (high - low) / 2 as a bid-ask proxy.
            # Not a precise spread — the Roll model uses the true bid-ask spread.
            # Using (H-L)/2 as a conservative upper bound for the daily spread.
            # See §2.10 Roll (1984) correction.
            if len(grp) >= _VOL_WINDOW:
                spread_px = float((high.iloc[-_VOL_WINDOW:] - low.iloc[-_VOL_WINDOW:]).mean() / 2.0)
            else:
                spread_px = float((high - low).mean() / 2.0)
            sym_spread_price[str(sym)] = spread_px

        spy_returns = sym_returns.get("SPY")
        rows: list[dict[str, object]] = []

        for sym, grp in bars.sort_values("ts").groupby("symbol", sort=False):
            sym = str(sym)
            if sym == "SPY":
                continue  # SPY is a reference; don't emit a feature row for it

            row: dict[str, object] = {"symbol": sym}
            rets = sym_returns.get(sym, pd.Series(dtype=float))
            n = len(rets)

            # realized_vol_bps: annualized, Roll bounce-corrected.
            # Var_corrected = max(0, Var(r) - s^2/4)
            # where s is the bid-ask spread in *return* units (price spread / midprice).
            # see docs/knowledge-base/part2-signals-bayesian.md §2.10
            if n >= _VOL_WINDOW:
                r_window = rets.iloc[-_VOL_WINDOW:]
                var_raw = float(r_window.var(ddof=1))

                # Convert spread proxy from price to return units.
                # Use last close as denominator for midprice approximation.
                grp_data = bars[bars["symbol"] == sym].sort_values("ts")
                last_close = float(grp_data["close"].iloc[-1]) if not grp_data.empty else 1.0
                s_price = sym_spread_price.get(sym, 0.0)
                s_return = s_price / max(last_close, 1e-6)

                # Roll (1984): Var_corrected = max(0, Var_close - s^2/4)
                var_corrected = max(0.0, var_raw - (s_return ** 2) / 4.0)
                vol_annual = math.sqrt(var_corrected * _ANNUALIZE)
                row["realized_vol_bps"] = vol_annual * _BPS
            else:
                row["realized_vol_bps"] = _MISSING

            # beta: OLS rolling 252d vs SPY.
            # Fall back to yfinance beta from Fundamentals if SPY not present.
            # see §1.4.6 "beta"
            if spy_returns is not None and n >= _BETA_WINDOW:
                # Align on common index
                combined = pd.concat(
                    [rets.rename("sym"), spy_returns.rename("spy")], axis=1
                ).dropna()
                if len(combined) >= 20:
                    cov = combined["sym"].cov(combined["spy"])
                    var_spy = combined["spy"].var(ddof=1)
                    row["beta"] = float(cov / var_spy) if var_spy > 0 else _MISSING
                else:
                    row["beta"] = _MISSING
            else:
                # Fall back to yfinance beta
                f = fundamentals.get(sym)
                if f is not None and f.beta is not None:
                    row["beta"] = float(f.beta)
                else:
                    row["beta"] = _MISSING

            # gap_risk: derived from gap_days_to_next_earnings.
            # Higher value = more gap risk. Invert so that "imminent earnings"
            # = higher raw gap_risk score (fewer days = higher risk).
            # §1.4.6 "gap risk"
            gap_days = _MISSING
            if event_features is not None and sym in event_features.index:
                gd = event_features.at[sym, "gap_days_to_next_earnings"]
                if not (isinstance(gd, float) and math.isnan(gd)):
                    gap_days = float(gd)

            if not math.isnan(gap_days):
                # Risk is higher when earnings are near: use 1 / (1 + gap_days)
                row["gap_risk"] = 1.0 / (1.0 + gap_days)
            else:
                row["gap_risk"] = _MISSING

            # crowding: placeholder.
            # TODO: integrate short-interest ratio or institutional overlap data.
            # Returns NaN so downstream handles as missing (§1.7).
            # §1.4.6 "crowding, short squeeze / unwind risk"
            row["crowding"] = _MISSING

            rows.append(row)

        if not rows:
            return pd.DataFrame()

        out = pd.DataFrame(rows).set_index("symbol")

        # Cross-sectional normalization per §1.2.
        for col in ["realized_vol_bps", "beta", "gap_risk"]:
            if col in out.columns:
                out[col] = _normalize(out, col)
        # crowding is all-NaN; skip normalization.

        log.debug("risk_signals_computed", n_symbols=len(out))
        return out
