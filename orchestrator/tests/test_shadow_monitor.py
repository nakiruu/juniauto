"""Tests for ShadowMonitor Bayesian posterior and promotion gating.

Covers:
  (a) Prior-only posterior: n=0 → delta_post=0, delta_post_se=sigma/sqrt(7).
  (b) Horizon-aware gate: 30 clean rows at horizon="2-3d" does NOT promote
      (n_eff = 30/1.6 ≈ 18.75 < min_n_eff["2-3d"] = 45).
  (c) k=2 peeking correction: single-cycle pass does not set promotion_ready.

See docs/knowledge-base/part5-shadow-and-replay.md §2.41.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from juniauto.config import ShadowConfig
from juniauto.shadow.monitor import ShadowMonitor, ShadowStats


def _make_cfg(**overrides: object) -> ShadowConfig:
    defaults: dict[str, object] = {
        "prior_delta": 0.0,
        "prior_strength_kappa0": 7.0,
        "min_clean_rows": 30,
        "min_positive_share": 0.55,
        "required_consecutive_passes": 2,
        "min_n_eff": {
            "1d": 30,
            "2-3d": 45,
            "1wk": 60,
            "2wk": 80,
        },
        "rho_label": {
            "1d": 0.00,
            "2-3d": 0.30,
            "1wk": 0.50,
            "2wk": 0.70,
        },
    }
    defaults.update(overrides)
    return ShadowConfig.model_validate(defaults)


def _make_monitor() -> ShadowMonitor:
    cfg = _make_cfg()
    db = MagicMock()
    return ShadowMonitor(cfg=cfg, db=db)


# ------------------------------------------------------------------ #
# (a) Prior-only posterior                                            #
# ------------------------------------------------------------------ #

class TestPriorOnly:
    """n=0 → posterior must equal the prior exactly."""

    def test_delta_post_equals_prior_delta(self) -> None:
        """n=0 → delta_post = delta_0 = 0. See §2.41."""
        mon = _make_monitor()
        stats = mon.evaluate("AAPL", "1d")
        assert stats.delta_post == 0.0, "Prior-only delta_post should be 0"

    def test_delta_post_se_equals_sigma_over_sqrt_kappa0(self) -> None:
        """n=0 → delta_post_se = sqrt(sigma_noise^2 / kappa0).

        With the nominal sigma^2=1 in the prior-only branch,
        delta_post_se = sqrt(1/7) = 1/sqrt(7).
        See §2.41 posterior standard error formula.
        """
        mon = _make_monitor()
        stats = mon.evaluate("AAPL", "1d")
        expected_se = 1.0 / math.sqrt(7.0)
        assert math.isclose(
            stats.delta_post_se, expected_se, rel_tol=1e-9
        ), f"Expected {expected_se:.6f}, got {stats.delta_post_se:.6f}"

    def test_n_clean_and_n_eff_zero(self) -> None:
        mon = _make_monitor()
        stats = mon.evaluate("MSFT", "2-3d")
        assert stats.n_clean == 0
        assert stats.n_eff == 0.0

    def test_promotion_ready_false_with_no_data(self) -> None:
        mon = _make_monitor()
        stats = mon.evaluate("TSLA", "1wk")
        assert stats.promotion_ready is False


# ------------------------------------------------------------------ #
# (b) Horizon-aware min_n_eff gate                                   #
# ------------------------------------------------------------------ #

class TestHorizonGating:
    """30 raw rows at "2-3d" must NOT trigger promotion.

    n_eff = 30 / (1 + 2*0.30) = 30 / 1.60 = 18.75 < min_n_eff["2-3d"] = 45.
    See §2.41 "Minimum n_eff per horizon" and spec quirk #2.
    """

    def _make_positive_deltas(self, n: int, value: float = 10.0) -> list[float]:
        return [value] * n

    def test_30_rows_at_2_3d_does_not_promote(self) -> None:
        mon = _make_monitor()
        sym = "NVDA"
        hz = "2-3d"
        # Ingest 30 identical positive deltas — all conditions except n_eff pass.
        for d in self._make_positive_deltas(30, 10.0):
            mon.ingest(sym, hz, d)

        # Evaluate twice (to satisfy k=2 peeking threshold IF gating allowed it).
        mon.evaluate(sym, hz)
        stats = mon.evaluate(sym, hz)

        # n_eff = 18.75 < 45 → gate_neff fails → promotion_ready must be False.
        assert stats.n_eff < 45, f"Expected n_eff < 45, got {stats.n_eff}"
        assert stats.promotion_ready is False, (
            "30 clean rows at 2-3d should NOT promote: n_eff < min_n_eff[2-3d]=45"
        )

    def test_n_eff_computation_for_2_3d(self) -> None:
        """n_eff = n_clean / (1 + 2*rho_label) for "2-3d" with rho=0.30."""
        mon = _make_monitor()
        for d in self._make_positive_deltas(30, 5.0):
            mon.ingest("AMZN", "2-3d", d)
        stats = mon.evaluate("AMZN", "2-3d")
        # 30 / 1.60 = 18.75
        assert math.isclose(stats.n_eff, 18.75, rel_tol=1e-6)

    def test_1d_horizon_uses_min_n_eff_30(self) -> None:
        """For 1d, n_eff == n_clean (rho=0). 30 rows should satisfy min_n_eff=30."""
        mon = _make_monitor()
        # Ingest 30 strongly positive deltas.
        for _ in range(30):
            mon.ingest("SPY", "1d", 5.0)
        # First pass (consecutive=1 → not promoted yet).
        s1 = mon.evaluate("SPY", "1d")
        assert s1.n_eff == 30.0
        assert s1.consecutive_pass == 1
        assert s1.promotion_ready is False  # needs k=2

        # Second pass → promotion_ready=True IF all other gates pass.
        s2 = mon.evaluate("SPY", "1d")
        assert s2.promotion_ready is True, (
            "30 rows at 1d with strongly positive deltas and k=2 should promote"
        )


# ------------------------------------------------------------------ #
# (c) k=2 peeking correction                                         #
# ------------------------------------------------------------------ #

class TestPeekingCorrection:
    """A single-cycle pass must NOT set promotion_ready = True."""

    def test_first_cycle_pass_is_not_promotion_ready(self) -> None:
        """Even with n_eff >> min and all conditions met, k=2 blocks first cycle.
        See §2.41 "Peeking correction (k=2 consecutive cycles)".
        """
        mon = _make_monitor()
        sym = "GOOG"
        hz = "1d"
        # Ingest 50 strongly positive deltas (well above min_n_eff=30 for 1d).
        for _ in range(50):
            mon.ingest(sym, hz, 8.0)

        stats = mon.evaluate(sym, hz)
        assert stats.consecutive_pass == 1
        assert stats.promotion_ready is False, (
            "First cycle pass must NOT set promotion_ready (k=2 required)"
        )

    def test_second_cycle_pass_sets_promotion_ready(self) -> None:
        mon = _make_monitor()
        sym = "META"
        hz = "1d"
        for _ in range(50):
            mon.ingest(sym, hz, 8.0)

        mon.evaluate(sym, hz)  # cycle 1
        stats = mon.evaluate(sym, hz)  # cycle 2
        assert stats.consecutive_pass == 2
        assert stats.promotion_ready is True

    def test_failed_cycle_resets_consecutive_counter(self) -> None:
        """A failing cycle must reset consecutive_pass to 0."""
        mon = _make_monitor()
        sym = "AMZN"
        hz = "1d"
        for _ in range(50):
            mon.ingest(sym, hz, 5.0)
        mon.evaluate(sym, hz)  # cycle 1 pass → consecutive=1

        # Add negative deltas to flip positive_share and delta_post negative.
        for _ in range(100):
            mon.ingest(sym, hz, -20.0)

        stats = mon.evaluate(sym, hz)  # cycle 2 fail
        assert stats.consecutive_pass == 0, (
            "Consecutive counter must reset to 0 after any failing cycle"
        )
        assert stats.promotion_ready is False

    def test_stats_is_frozen_dataclass(self) -> None:
        """ShadowStats must be immutable (frozen=True)."""
        stats = ShadowStats(
            horizon="1d",
            n_clean=0,
            n_eff=0.0,
            positive_share=0.0,
            delta_post=0.0,
            delta_post_se=0.377,
            consecutive_pass=0,
            promotion_ready=False,
        )
        with pytest.raises(Exception):
            stats.horizon = "2-3d"  # type: ignore[misc]
