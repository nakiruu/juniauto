"""Market-regime signal (observation-only, coordinator review 2026-07-24).

Blends three complementary components into a scalar stress score in [0, 1]:

  (a) SPY drawdown from N-day high, normalized to [0, 1]
        dd_score = clip(-drawdown_pct / 0.20, 0, 1)   # 20% drawdown = full stress
  (b) SPY 21d realized-vol percentile rank in trailing 252d
        vol_score = pct_rank in [0, 1]
  (c) Mean pairwise 21d return correlation of top-N by dollar volume
        corr_score = clip((corr - 0.30) / (0.90 - 0.30), 0, 1)

  stress_raw = w_dd * dd_score + w_vol * vol_score + w_corr * corr_score

EMA-smoothed across cycles to prevent whipsaws (halflife = 5 trading days
by default), then converted to an implied γ_risk multiplier:

  span = gamma_risk_max / gamma_risk_base - 1
  gamma_multiplier = 1.0 + stress_ema * span

The multiplier is logged to the `market_regime` table + Prometheus but NOT
applied to Kelly sizing while `regime.apply_to_kelly=false`. See the
promotion protocol in `docs/knowledge-base/part4-gateway-execution.md`.

This signal is deliberately kept out of `compute_all()` — it's a
portfolio-level (not per-symbol) metric, and returning a scalar from a
per-symbol pipeline would confuse consumers. Call `MarketRegimeSignals`
directly from the orchestrator after bars are refreshed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from juniauto.config import RegimeConfig
from juniauto.utils import get_logger

log = get_logger(__name__)


_MISSING = float("nan")


@dataclass(frozen=True)
class MarketRegimeSnapshot:
    """One cycle's regime observation. All fields NaN-safe."""
    spy_drawdown_pct: float          # spy_close / spy_63d_high - 1, clipped to [-1, 0]
    spy_vol_pct_rank: float          # 21d vol percentile in trailing 252d, [0, 1]
    avg_pairwise_corr: float         # mean pairwise 21d return corr of top-N, [-1, 1]
    stress_raw: float                # blended stress in [0, 1]
    stress_ema: float                # EMA-smoothed stress in [0, 1]
    gamma_multiplier: float          # 1 + stress_ema * (gamma_risk_max/gamma_risk_base - 1)
    n_symbols_corr: int              # how many names contributed to the pairwise-corr avg

    def is_missing(self) -> bool:
        """True when every stress component is NaN (bars too shallow to compute)."""
        return (
            math.isnan(self.spy_drawdown_pct)
            and math.isnan(self.spy_vol_pct_rank)
            and math.isnan(self.avg_pairwise_corr)
        )


class MarketRegimeSignals:
    """Compute portfolio-level regime stress from a daily bars DataFrame."""

    def __init__(self, cfg: RegimeConfig) -> None:
        self.cfg = cfg
        # EMA weight per cycle: alpha = 1 - 0.5**(1/halflife). For halflife=5,
        # alpha ≈ 0.1294 — a fresh stress reading contributes ~13%, prior EMA ~87%.
        h = max(float(cfg.regime_stress_ema_halflife_days), 1e-6)
        self._ema_alpha = 1.0 - 0.5 ** (1.0 / h)
        # Multiplier span. base=1.0 max=3.0 → span=2.0 → stress=1 → multiplier=3.0.
        base = max(float(cfg.gamma_risk_base), 1e-6)
        self._span = float(cfg.gamma_risk_max) / base - 1.0

    def compute(
        self,
        bars: pd.DataFrame,
        prev_stress_ema: float | None = None,
    ) -> MarketRegimeSnapshot:
        """Compute a single regime snapshot from the current bars window.

        Args:
            bars: Long-format daily bars (columns: symbol, ts, open, high,
                low, close, volume, ...). Must contain enough history for
                each lookback window; components that lack history return NaN.
            prev_stress_ema: Prior EMA value (from the last market_regime
                row). On cold start (None), stress_ema seeds to stress_raw.
        """
        if bars.empty:
            return MarketRegimeSnapshot(
                _MISSING, _MISSING, _MISSING, _MISSING, _MISSING, 1.0, 0,
            )

        ref = self.cfg.reference_symbol
        dd_pct = self._compute_drawdown(bars, ref, self.cfg.drawdown_lookback_days)
        vol_rank = self._compute_vol_pct_rank(
            bars, ref, self.cfg.vol_lookback_days, self.cfg.vol_pct_lookback_days
        )
        avg_corr, n_corr = self._compute_avg_pairwise_corr(
            bars, self.cfg.corr_top_n, self.cfg.corr_lookback_days, exclude=ref
        )

        # Component scores in [0, 1] with NaN-passthrough.
        dd_score = _clip01(-dd_pct / 0.20) if not math.isnan(dd_pct) else _MISSING
        vol_score = vol_rank  # already in [0, 1] or NaN
        corr_score = _clip01((avg_corr - 0.30) / 0.60) if not math.isnan(avg_corr) else _MISSING

        # Blended stress: weighted mean over the NON-nan components, renormalizing
        # weights so a missing component doesn't zero out the whole score. If ALL
        # components are NaN, stress_raw is NaN.
        parts = [
            (dd_score, self.cfg.weight_drawdown),
            (vol_score, self.cfg.weight_vol_percentile),
            (corr_score, self.cfg.weight_correlation),
        ]
        num = 0.0
        wsum = 0.0
        for val, w in parts:
            if not math.isnan(val):
                num += w * val
                wsum += w
        stress_raw = num / wsum if wsum > 0.0 else _MISSING

        # EMA smoothing. On cold start (prev None) or NaN, seed to stress_raw.
        if math.isnan(stress_raw):
            stress_ema = _MISSING
        elif prev_stress_ema is None or math.isnan(prev_stress_ema):
            stress_ema = stress_raw
        else:
            stress_ema = self._ema_alpha * stress_raw + (1.0 - self._ema_alpha) * prev_stress_ema

        # Implied multiplier — always defined (defaults to 1.0 when stress is NaN
        # so the observation-only column is never null in downstream Grafana).
        if math.isnan(stress_ema):
            gamma_mult = 1.0
        else:
            gamma_mult = 1.0 + max(0.0, stress_ema) * self._span

        return MarketRegimeSnapshot(
            spy_drawdown_pct=dd_pct,
            spy_vol_pct_rank=vol_rank,
            avg_pairwise_corr=avg_corr,
            stress_raw=stress_raw,
            stress_ema=stress_ema,
            gamma_multiplier=gamma_mult,
            n_symbols_corr=n_corr,
        )

    # ---- Components ----

    @staticmethod
    def _compute_drawdown(bars: pd.DataFrame, ref_symbol: str, lookback_days: int) -> float:
        """Return spy_close / spy_max_high_over_lookback - 1, clipped to [-1, 0].
        Negative-only (drawdown never positive by construction)."""
        spy = bars[bars["symbol"] == ref_symbol].sort_values("ts")
        if len(spy) < 2:
            return _MISSING
        window = spy.iloc[-lookback_days:] if len(spy) >= lookback_days else spy
        peak = float(window["high"].max())
        last_close = float(window["close"].iloc[-1])
        if peak <= 0.0:
            return _MISSING
        dd = (last_close / peak) - 1.0
        return max(-1.0, min(0.0, dd))

    @staticmethod
    def _compute_vol_pct_rank(
        bars: pd.DataFrame,
        ref_symbol: str,
        vol_window: int,
        pct_window: int,
    ) -> float:
        """Rolling std of daily returns over `vol_window`, then percentile rank
        of the LAST value within the trailing `pct_window` rolling-vol series.

        Uses raw close-to-close returns (no Roll bias correction): consistent
        with what a naive VIX proxy would produce, and we want the *rank*
        anyway, not the absolute level.
        """
        spy = bars[bars["symbol"] == ref_symbol].sort_values("ts")
        if len(spy) < vol_window + 5:
            return _MISSING
        rets = spy["close"].astype(float).pct_change().dropna()
        if len(rets) < vol_window:
            return _MISSING
        rolling_vol = rets.rolling(window=vol_window).std(ddof=1).dropna()
        if len(rolling_vol) < 2:
            return _MISSING
        # Trailing pct_window window for the rank base; last value is the "current".
        base = rolling_vol.iloc[-pct_window:] if len(rolling_vol) >= pct_window else rolling_vol
        current = float(base.iloc[-1])
        # Percentile rank: fraction of prior obs strictly ≤ current.
        # scipy-free implementation with mid-rank for ties.
        arr = base.to_numpy()
        n = len(arr)
        if n < 2:
            return _MISSING
        lt = float(np.sum(arr < current))
        eq = float(np.sum(arr == current))
        # Midrank convention: (# strictly less + 0.5 * # equal) / n.
        return (lt + 0.5 * eq) / n

    @staticmethod
    def _compute_avg_pairwise_corr(
        bars: pd.DataFrame,
        top_n: int,
        lookback_days: int,
        exclude: str,
    ) -> tuple[float, int]:
        """Mean pairwise correlation of daily returns over `lookback_days` for
        the top-N symbols ranked by dollar-volume over the same window."""
        if bars.empty:
            return _MISSING, 0

        # Pick top-N by mean(close*volume) over the last lookback_days.
        recent = bars.sort_values("ts").groupby("symbol").tail(lookback_days)
        if recent.empty:
            return _MISSING, 0
        recent = recent.assign(_dv=recent["close"].astype(float) * recent["volume"].astype(float))
        dv = recent.groupby("symbol")["_dv"].mean().sort_values(ascending=False)
        pick = [s for s in dv.index.tolist() if s != exclude][:top_n]
        if len(pick) < 3:
            return _MISSING, len(pick)

        # Build a wide returns matrix: rows=ts, cols=symbol.
        window = bars[bars["symbol"].isin(pick)].sort_values(["symbol", "ts"])
        wide = window.pivot_table(index="ts", columns="symbol", values="close", aggfunc="last")
        rets = wide.pct_change().dropna(how="all").tail(lookback_days)
        # Require enough overlap: at least 10 rows and ≥3 non-null cols.
        rets = rets.dropna(axis=1, thresh=max(10, lookback_days // 2))
        if rets.shape[0] < 10 or rets.shape[1] < 3:
            return _MISSING, int(rets.shape[1])

        corr = rets.corr(min_periods=10)
        # Mean of strictly upper-triangular entries (excludes self-corr diagonal).
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        vals = corr.to_numpy()[mask]
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            return _MISSING, int(rets.shape[1])
        return float(vals.mean()), int(rets.shape[1])


def _clip01(x: float) -> float:
    if math.isnan(x):
        return _MISSING
    return max(0.0, min(1.0, x))
