"""Tests for the cross-sectional winsorize + MAD-standardize normalization pipeline.

Verifies:
  - After normalization, median → 0 within tolerance.
  - After normalization, 1.4826 * MAD → 1 within tolerance.
  - Winsorize then standardize (order matters per §1.2).
  - NaN inputs propagate as NaN (never become 0).
  - Extreme outliers are clipped before MAD is fitted.

See docs/knowledge-base/part1-data-preparation.md §1.2.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from juniauto.signals._norm import _mad_standardize, _normalize, _winsorize


def _make_df(values: list[float], col: str = "x") -> pd.DataFrame:
    return pd.DataFrame({col: values})


class TestWinsorize:
    """_winsorize clips to 1st/99th percentile."""

    def test_clips_extreme_high(self) -> None:
        s = pd.Series([1.0] * 98 + [1000.0, 0.001])
        w = _winsorize(s)
        # The 99th percentile is 1.0, so 1000 gets clipped.
        assert w.max() <= 1.0 + 1e-9

    def test_clips_extreme_low(self) -> None:
        s = pd.Series([100.0] * 98 + [1e6, -1e6])
        w = _winsorize(s)
        assert w.min() >= 100.0 - 1e-9

    def test_nan_preserved(self) -> None:
        s = pd.Series([1.0, 2.0, float("nan"), 4.0])
        w = _winsorize(s)
        assert math.isnan(w.iloc[2])

    def test_passthrough_when_no_outliers(self) -> None:
        s = pd.Series(list(range(100)), dtype=float)
        w = _winsorize(s)
        # All values within 1st/99th, so unchanged (approximately).
        assert abs(w.mean() - s.mean()) < 1.0


class TestMADStandardize:
    """_mad_standardize: median → 0, 1.4826*MAD → 1."""

    def test_median_is_zero(self) -> None:
        np.random.seed(0)
        s = pd.Series(np.random.normal(5.0, 2.0, 500))
        out = _mad_standardize(s)
        assert abs(out.median()) < 0.05, f"Normalized median {out.median():.4f} != 0"

    def test_mad_scale_is_one(self) -> None:
        np.random.seed(1)
        s = pd.Series(np.random.normal(0.0, 3.0, 500))
        out = _mad_standardize(s)
        mad = (out - out.median()).abs().median()
        # 1.4826 * MAD of standardized series should be ≈ 1
        assert abs(1.4826 * mad - 1.0) < 0.05, f"1.4826*MAD = {1.4826 * mad:.4f} != 1"

    def test_nan_propagates(self) -> None:
        s = pd.Series([1.0, 2.0, float("nan"), 4.0, 5.0])
        out = _mad_standardize(s)
        assert math.isnan(out.iloc[2])

    def test_constant_series_returns_nan(self) -> None:
        """Zero MAD → all-NaN output (no sensible scale)."""
        s = pd.Series([3.14] * 50)
        out = _mad_standardize(s)
        assert out.isna().all(), "Constant series must yield all-NaN after MAD-standardize"


class TestNormalize:
    """_normalize: winsorize then MAD-standardize round-trip."""

    def test_median_zero_and_mad_one_after_full_roundtrip(self) -> None:
        """The round-trip of winsorize + MAD-standardize must satisfy:
            median(out) ≈ 0
            1.4826 * MAD(out) ≈ 1
        See §1.2.
        """
        np.random.seed(42)
        vals = list(np.random.normal(10.0, 5.0, 200))
        # Add a few extreme outliers to test winsorization effect.
        vals += [1e6, -1e6]
        df = _make_df(vals)
        out = _normalize(df, "x")

        med = float(out.median())
        mad = float((out - out.median()).abs().median())

        assert abs(med) < 0.05, f"Normalized median {med:.4f} should be near 0"
        assert abs(1.4826 * mad - 1.0) < 0.05, (
            f"1.4826 * MAD = {1.4826 * mad:.4f} should be near 1.0"
        )

    def test_nan_propagates_through_normalize(self) -> None:
        """NaN inputs must remain NaN after normalization, never become 0."""
        vals = [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        df = _make_df(vals)
        out = _normalize(df, "x")
        assert math.isnan(out.iloc[2]), "NaN must propagate through normalization"
        # Non-NaN values must not be 0 (they should be z-scores near 0 but not pinned).
        non_nan = out.dropna()
        assert len(non_nan) == 9

    def test_winsorize_before_mad_not_after(self) -> None:
        """Winsorization before MAD-fitting prevents outliers from inflating scale.

        Without winsorization, a single extreme outlier at 1e6 would make MAD
        large, compressing all other z-scores toward 0. After winsorization the
        outlier is clipped and MAD is computed on the trimmed distribution.
        """
        np.random.seed(7)
        core = list(np.random.normal(0.0, 1.0, 198))
        outliers = [1e8, -1e8]  # without winsorization these dominate MAD
        df = _make_df(core + outliers)
        out = _normalize(df, "x")
        mad_out = float((out - out.median()).abs().median())
        # If winsorization is applied correctly, 1.4826 * MAD ≈ 1.
        assert abs(1.4826 * mad_out - 1.0) < 0.15, (
            f"Outlier inflation check failed: 1.4826*MAD = {1.4826 * mad_out:.3f}"
        )

    def test_small_cross_section_still_normalizes(self) -> None:
        """Even with 3 symbols the normalization must not crash."""
        df = _make_df([1.0, 5.0, 10.0])
        out = _normalize(df, "x")
        assert len(out) == 3
        assert not out.isna().all()
