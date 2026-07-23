"""Technical signal family computer.

Computes seven features from daily OHLCV bars using pandas/numpy only.
No TA-Lib dependency (avoids the C build requirement).
Daily bars only — intraday structure is excluded per §2.4 (IEX/PDT adaptation).

Output columns match `features` table in orchestrator/juniauto/db/schema.sql §1.4.1:
    trend_slope, relative_strength, breakout_strength, ma_distance,
    price_acceleration, volume_confirmation, support_defense
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from juniauto.signals._norm import _normalize
from juniauto.utils import get_logger

log = get_logger(__name__)

# ---- Lookback constants (trading days) ----
_SHORT_WINDOW = 20   # momentum / MA short
_LONG_WINDOW = 60    # trend slope / MA long; §1.4.1 multi-day horizon
_TREND_WINDOW = 10   # OLS regression window for slope
_VOL_WINDOW = 20     # volume average for relative volume
_ATR_WINDOW = 14     # breakout / ATR proxy
_ACCEL_WINDOW = 5    # price acceleration look-back


class TechnicalSignals:
    """Compute technical signal family from daily bars.

    All computations are per-symbol, then cross-sectionally normalized.
    No intraday data is consumed or required.
    See docs/knowledge-base/part1-data-preparation.md §1.4.1.
    """

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Compute technical features for each symbol.

        Args:
            bars: DataFrame with columns
                [symbol, ts, open, high, low, close, volume, vwap].
                Rows must be strictly *before* the decision timestamp.
                Multiple symbols may be present; use all rows for
                cross-sectional normalization.

        Returns:
            DataFrame indexed by symbol with columns:
            trend_slope, relative_strength, breakout_strength, ma_distance,
            price_acceleration, volume_confirmation, support_defense.
            Missing values are NaN (not zero).
        """
        if bars.empty:
            return pd.DataFrame()

        results: list[dict[str, object]] = []

        for sym, grp in bars.sort_values("ts").groupby("symbol", sort=False):
            grp = grp.sort_values("ts").reset_index(drop=True)
            close = grp["close"].astype(float)
            high = grp["high"].astype(float)
            low = grp["low"].astype(float)
            vol = grp["volume"].astype(float)
            n = len(grp)

            row: dict[str, object] = {"symbol": sym}

            # -- trend_slope: OLS slope of log(close) over last _TREND_WINDOW days --
            # see §1.4.1 "trend slope"
            if n >= _TREND_WINDOW:
                y = np.log(close.iloc[-_TREND_WINDOW:].values)
                x = np.arange(_TREND_WINDOW, dtype=float)
                slope = float(np.polyfit(x, y, 1)[0])
            else:
                slope = float("nan")
            row["trend_slope"] = slope

            # -- relative_strength: 20d return vs prior 20d return (momentum) --
            # see §1.4.1 "relative strength"
            if n >= _SHORT_WINDOW * 2:
                ret_recent = close.iloc[-1] / close.iloc[-_SHORT_WINDOW] - 1.0
                ret_prior = close.iloc[-_SHORT_WINDOW] / close.iloc[-_SHORT_WINDOW * 2] - 1.0
                row["relative_strength"] = float(ret_recent - ret_prior)
            elif n >= _SHORT_WINDOW + 1:
                row["relative_strength"] = float(close.iloc[-1] / close.iloc[-_SHORT_WINDOW] - 1.0)
            else:
                row["relative_strength"] = float("nan")

            # -- breakout_strength: close vs rolling ATR-based high --
            # Proxy ATR = mean(high - low) over _ATR_WINDOW days.
            # Breakout = (close - rolling_max_close) / ATR
            # see §1.4.1 "breakout strength"
            if n >= _ATR_WINDOW:
                atr = (high.iloc[-_ATR_WINDOW:] - low.iloc[-_ATR_WINDOW:]).mean()
                roll_high = high.iloc[-_ATR_WINDOW:].max()
                if atr > 0:
                    row["breakout_strength"] = float((close.iloc[-1] - roll_high) / atr)
                else:
                    row["breakout_strength"] = float("nan")
            else:
                row["breakout_strength"] = float("nan")

            # -- ma_distance: z-score of close vs 20d and 60d MA composite --
            # (close / ma20 + close / ma60) / 2 - 1
            # see §1.4.1 "distance from MAs"
            ma20 = close.iloc[-_SHORT_WINDOW:].mean() if n >= _SHORT_WINDOW else float("nan")
            ma60 = close.iloc[-_LONG_WINDOW:].mean() if n >= _LONG_WINDOW else float("nan")
            c = float(close.iloc[-1])
            if not np.isnan(ma20) and ma20 > 0 and not np.isnan(ma60) and ma60 > 0:
                row["ma_distance"] = ((c / ma20 - 1.0) + (c / ma60 - 1.0)) / 2.0
            elif not np.isnan(ma20) and ma20 > 0:
                row["ma_distance"] = c / ma20 - 1.0
            else:
                row["ma_distance"] = float("nan")

            # -- price_acceleration: momentum of momentum --
            # rate of change of 5d return (recent) vs prior 5d return
            # see §1.4.1 "price acceleration"
            if n >= _ACCEL_WINDOW * 2 + 1:
                r1 = close.iloc[-1] / close.iloc[-_ACCEL_WINDOW - 1] - 1.0
                r2 = close.iloc[-_ACCEL_WINDOW - 1] / close.iloc[-_ACCEL_WINDOW * 2 - 1] - 1.0
                row["price_acceleration"] = float(r1 - r2)
            else:
                row["price_acceleration"] = float("nan")

            # -- volume_confirmation: recent volume vs long-run average --
            # Positive when price move is volume-confirmed; signed by last-day return.
            # see §1.4.1 "volume confirmation"
            if n >= _VOL_WINDOW:
                avg_vol = vol.iloc[-_VOL_WINDOW:].mean()
                last_vol = float(vol.iloc[-1])
                price_sign = np.sign(close.iloc[-1] - close.iloc[-2]) if n >= 2 else 0.0
                if avg_vol > 0:
                    row["volume_confirmation"] = float(price_sign * (last_vol / avg_vol - 1.0))
                else:
                    row["volume_confirmation"] = float("nan")
            else:
                row["volume_confirmation"] = float("nan")

            # -- support_defense: proximity to 20d low as a defense score --
            # Positive if close is well above 20d low (supported); negative if near 20d low.
            # see §1.4.1 "support/resistance behavior"
            if n >= _SHORT_WINDOW:
                low20 = low.iloc[-_SHORT_WINDOW:].min()
                high20 = high.iloc[-_SHORT_WINDOW:].max()
                rng = high20 - low20
                if rng > 0:
                    row["support_defense"] = float((close.iloc[-1] - low20) / rng)
                else:
                    row["support_defense"] = float("nan")
            else:
                row["support_defense"] = float("nan")

            results.append(row)

        if not results:
            return pd.DataFrame()

        out = pd.DataFrame(results).set_index("symbol")

        # Cross-sectional normalization: winsorize then MAD-standardize.
        # see §1.2 — order is mandated.
        feature_cols = [
            "trend_slope",
            "relative_strength",
            "breakout_strength",
            "ma_distance",
            "price_acceleration",
            "volume_confirmation",
            "support_defense",
        ]
        for col in feature_cols:
            if col in out.columns:
                out[col] = _normalize(out, col)

        log.debug("technical_signals_computed", n_symbols=len(out))
        return out
