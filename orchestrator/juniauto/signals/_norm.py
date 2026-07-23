"""Shared normalization helpers used across all signal families.

Winsorize at 1st/99th cross-sectional percentile, then MAD-standardize.
Order is mandated: winsorize *then* fit MAD to the clipped distribution.
See docs/knowledge-base/part1-data-preparation.md §1.2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _winsorize(series: pd.Series) -> pd.Series:
    """Clip to cross-sectional 1st/99th percentile.

    Only uses observations in `series` — no look-ahead.
    NaN values are excluded from percentile calculation and preserved as NaN.
    See §1.2 cross-sectional percentile clip.
    """
    lo = series.quantile(0.01)
    hi = series.quantile(0.99)
    return series.clip(lower=lo, upper=hi)


def _mad_standardize(series: pd.Series) -> pd.Series:
    """MAD-unit scale: (x - median) / (1.4826 * MAD).

    Applied *after* winsorization so the MAD is fitted to the clipped
    distribution, not the raw one. See §1.2.
    """
    med = series.median()
    mad = (series - med).abs().median()
    scale = 1.4826 * mad
    if scale == 0 or np.isnan(scale):
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (series - med) / scale


def _normalize(df: pd.DataFrame, col: str) -> pd.Series:
    """Winsorize then MAD-standardize a single column of `df` in place.

    Returns the normalized series. NaN inputs propagate as NaN.
    Caller is responsible for assigning back to the output DataFrame.
    See §1.2 — order matters.
    """
    raw = df[col].copy()
    winsorized = _winsorize(raw)
    return _mad_standardize(winsorized)
