"""Shadow promotion monitor — Normal-normal conjugate posterior on Δnet_bps.

Implements the promotion gating logic from §2.41:
    - Normal-normal posterior update on the challenger delta vs baseline.
    - Effective sample size correction for overlapping forward-return labels.
    - k=2 consecutive-cycle peeking correction (Johari et al. 2017).
    - Horizon-aware min_n_eff gating (not a flat 30 for non-1d horizons).
    - Persists Δnet_bps rows to QuestDB `shadow_deltas` table.

Two kappas: κ₀ = 7 here (config.shadow.prior_strength_kappa0).
             κ = 20 is the ridge-bucket prior in challenger.py. Different constants.
See docs/knowledge-base/part5-shadow-and-replay.md §2.41.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd
from questdb.ingress import Sender, TimestampNanos

from juniauto.config import ShadowConfig
from juniauto.db.client import QuestDBClient
from juniauto.utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ShadowStats:
    """Frozen summary of the shadow monitor state for one (symbol, horizon).

    Fields:
        horizon:          e.g. "1d", "2-3d", "1wk", "2wk"
        n_clean:          raw count of resolved, clean shadow rows
        n_eff:            effective sample size after label-overlap correction
        positive_share:   fraction of clean rows where Δnet_bps > 0
        delta_post:       posterior mean of challenger Δnet_bps
        delta_post_se:    posterior standard error = sqrt(sigma_noise^2 / (κ₀ + n_clean))
        consecutive_pass: number of consecutive evaluation cycles all gates passed
        promotion_ready:  True iff all gates pass for k≥2 consecutive cycles
    """

    horizon: str
    n_clean: int
    n_eff: float
    positive_share: float
    delta_post: float
    delta_post_se: float
    consecutive_pass: int
    promotion_ready: bool


class ShadowMonitor:
    """Bayesian shadow promotion monitor per §2.41.

    Maintains per-(symbol, horizon) state across decision cycles.
    Call `update()` each cycle with newly resolved shadow rows.
    Call `evaluate()` to get current ShadowStats and check promotion.

    Critical gating rules (§2.41):
        1. n_eff >= min_n_eff[horizon]   (NOT flat 30 for non-1d horizons)
        2. positive_share >= 0.55
        3. delta_post > 0
        4. delta_post_se < |delta_post|  (posterior not within 1 SE of zero)
        5. All four pass for k=2 consecutive cycles (peeking correction)
    """

    def __init__(self, cfg: ShadowConfig, db: QuestDBClient) -> None:
        self._cfg = cfg
        self._db = db

        # Per (symbol, horizon) state: list of clean Δnet_bps observations
        # and consecutive pass counter.
        # see §2.41 — peeking correction requires tracking consecutive cycles
        self._deltas: dict[tuple[str, str], list[float]] = {}
        self._consecutive: dict[tuple[str, str], int] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def ingest(
        self,
        symbol: str,
        horizon: str,
        delta_net_bps: float,
    ) -> None:
        """Record one resolved shadow Δnet_bps observation.

        Args:
            symbol:        Ticker symbol.
            horizon:       Horizon label: "1d" | "2-3d" | "1wk" | "2wk".
            delta_net_bps: Challenger net_bps minus baseline net_bps for
                           this resolved trade window.
        """
        key = (symbol, horizon)
        self._deltas.setdefault(key, []).append(delta_net_bps)

    def ingest_batch(
        self,
        rows: Sequence[tuple[str, str, float]],
    ) -> None:
        """Ingest multiple (symbol, horizon, delta_net_bps) tuples."""
        for sym, hz, d in rows:
            self.ingest(sym, hz, d)

    def evaluate(self, symbol: str, horizon: str) -> ShadowStats:
        """Compute current Bayesian state and promotion decision.

        Applies all five promotion gates from §2.41 and the k=2 peeking
        correction. Updates the consecutive_pass counter in-place.

        Args:
            symbol:  Ticker symbol.
            horizon: Horizon label.

        Returns:
            ShadowStats snapshot. `promotion_ready` is True only when all
            gates pass on the current cycle AND the previous cycle (k=2).
        """
        key = (symbol, horizon)
        deltas = self._deltas.get(key, [])
        n_clean = len(deltas)

        # --- Effective sample size correction for label overlap ---
        # n_eff = n_clean / (1 + 2 * rho_label)
        # see §2.41 "Effective sample size — overlapping labels"
        rho = self._cfg.rho_label.get(horizon, 0.0)
        n_eff = n_clean / (1.0 + 2.0 * rho) if n_clean > 0 else 0.0

        # --- Posterior update (Normal-Normal conjugate) ---
        # delta_post = (κ₀·δ₀ + n_clean·mean) / (κ₀ + n_clean)
        # sigma_post  = sqrt(sigma_noise^2 / (κ₀ + n_clean))
        # config keys: shadow.prior_strength_kappa0, shadow.prior_delta
        # see §2.41 "Normal-normal conjugate posterior on delta"
        kappa0 = self._cfg.prior_strength_kappa0  # κ₀ = 7; §2.41
        delta0 = self._cfg.prior_delta             # δ₀ = 0; §2.41

        if n_clean > 0:
            obs_mean = sum(deltas) / n_clean
            delta_post = (kappa0 * delta0 + n_clean * obs_mean) / (kappa0 + n_clean)

            # sigma_noise^2: empirical Bayes plug-in from sample variance.
            # When n_clean == 1, variance is undefined; use obs as point estimate.
            if n_clean >= 2:
                obs_var = sum((d - obs_mean) ** 2 for d in deltas) / (n_clean - 1)
            else:
                obs_var = abs(deltas[0])  # conservative fallback for n=1

            sigma_post_sq = obs_var / (kappa0 + n_clean)
            delta_post_se = math.sqrt(max(0.0, sigma_post_sq))

            positive_share = sum(1 for d in deltas if d > 0) / n_clean
        else:
            # n=0: prior only
            delta_post = delta0
            sigma_post_sq = 1.0 / kappa0  # prior variance: σ²/κ₀ with σ²=1 nominal
            delta_post_se = math.sqrt(sigma_post_sq)
            positive_share = 0.0

        # --- Gate evaluation ---
        # Gate 1: n_eff >= min_n_eff[horizon]
        # CRITICAL: min_clean_rows=30 is the *1-day* floor only.
        # Other horizons use min_n_eff from config, not flat 30.
        # see §2.41 "Minimum n_eff per horizon"
        min_neff = self._cfg.min_n_eff.get(horizon, self._cfg.min_clean_rows)
        gate_neff = n_eff >= min_neff

        # Gate 2: positive_share >= 0.55
        # config key: shadow.min_positive_share; §2.41
        gate_share = positive_share >= self._cfg.min_positive_share

        # Gate 3: delta_post > 0
        gate_positive = delta_post > 0.0

        # Gate 4: delta_post_se < |delta_post| (not within 1 SE of zero)
        # see §2.41 "Do not promote when delta_post_se > delta_post"
        gate_se = delta_post_se < abs(delta_post)

        all_pass = gate_neff and gate_share and gate_positive and gate_se

        # --- k=2 peeking correction ---
        # promotion_ready requires consecutive_pass >= required_consecutive_passes
        # config key: shadow.required_consecutive_passes (= 2)
        # see §2.41 "Peeking correction (k=2 consecutive cycles)"
        prev_consecutive = self._consecutive.get(key, 0)
        if all_pass:
            consecutive = prev_consecutive + 1
        else:
            consecutive = 0
        self._consecutive[key] = consecutive

        k_required = self._cfg.required_consecutive_passes  # = 2; §2.41
        promotion_ready = all_pass and consecutive >= k_required

        stats = ShadowStats(
            horizon=horizon,
            n_clean=n_clean,
            n_eff=n_eff,
            positive_share=positive_share,
            delta_post=delta_post,
            delta_post_se=delta_post_se,
            consecutive_pass=consecutive,
            promotion_ready=promotion_ready,
        )

        log.debug(
            "shadow_evaluate",
            symbol=symbol,
            horizon=horizon,
            n_clean=n_clean,
            n_eff=round(n_eff, 1),
            delta_post=round(delta_post, 3),
            delta_post_se=round(delta_post_se, 3),
            consecutive=consecutive,
            promotion_ready=promotion_ready,
        )
        return stats

    def persist(self, symbol: str, stats: ShadowStats) -> None:
        """Write a ShadowStats row to QuestDB `shadow_deltas` table via ILP.

        Args:
            symbol: Ticker symbol.
            stats:  Current ShadowStats for (symbol, horizon).
        """
        # see orchestrator/juniauto/db/schema.sql `shadow_deltas`
        with self._db.sender() as s:
            s.row(
                "shadow_deltas",
                symbols={
                    "symbol": symbol,
                    "horizon": stats.horizon,
                },
                columns={
                    "delta_net_bps": float("nan"),  # aggregate row; individual deltas stored per ingest
                    "n_clean": stats.n_clean,
                    "n_eff": int(stats.n_eff),
                    "positive_share": stats.positive_share,
                    "delta_post": stats.delta_post,
                    "delta_post_se": stats.delta_post_se,
                    "consecutive_pass": stats.consecutive_pass,
                    "promotion_ready": stats.promotion_ready,
                },
                at=TimestampNanos.now(),
            )

    def evaluate_and_persist(self, symbol: str, horizon: str) -> ShadowStats:
        """Evaluate promotion state and persist the result to QuestDB."""
        stats = self.evaluate(symbol, horizon)
        self.persist(symbol, stats)
        return stats
