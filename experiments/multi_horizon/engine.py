"""MultiHorizonBacktestEngine — thin subclass of BacktestEngine.

Overrides ONLY `_compute_predictions` to use the multi-horizon blended
mu/sigma from `MultiHorizonModel`. Everything else — snapshot loader,
gateway evaluation, sizing, order routing, persistence, metrics — is
reused unchanged from the main backtest engine.

Reuses:
  - juniauto.backtest.engine.BacktestEngine
  - juniauto.signals + juniauto.portfolio + qe.evaluate_gateway
  - all backtest_* tables (run_id is enough to isolate this experiment)
"""
from __future__ import annotations

import math

import quant_engine as qe

from experiments.multi_horizon.training import MultiHorizonModel
from juniauto.backtest.engine import BacktestEngine
from juniauto.config import JuniAutoConfig
from juniauto.utils import get_logger

log = get_logger(__name__)


class MultiHorizonBacktestEngine(BacktestEngine):
    """Subclass that swaps the single-horizon Bayesian model for a
    multi-horizon ensemble. All other behavior identical to the parent.
    """

    def __init__(
        self,
        cfg: JuniAutoConfig,
        *,
        horizons: tuple[int, ...] = (1, 5, 20),
        blend_weights: tuple[float, ...] = (0.15, 0.50, 0.35),
        **kwargs,
    ) -> None:
        # Parent init constructs a single-horizon BayesianModel; we let
        # that run so `self.bayes` exists (its predict() is never called
        # after we override _compute_predictions below), then replace
        # with the multi-horizon model.
        super().__init__(cfg, **kwargs)
        self.mh_model = MultiHorizonModel(
            self.db, cfg,
            horizons=horizons, blend_weights=blend_weights,
        )
        log.info(
            "mh_engine_ready",
            horizons=list(horizons),
            blend_weights=list(blend_weights),
            is_trained=self.mh_model.is_trained(),
            n_samples_by_horizon={
                h: self.mh_model._n_samples[h]  # noqa: SLF001
                for h in horizons
            },
        )

    # ================================================================
    # Overridden prediction: use MultiHorizonModel.predict for blended mu/sigma
    # ================================================================
    def _compute_predictions(self, features):
        """Same signature and output shape as BacktestEngine._compute_predictions,
        but uses the multi-horizon blended prediction for mu + sigma.
        Composite edge computation unchanged.
        """
        role_enum = qe.Role.Primary
        membership_bps = qe.membership_edge_bps(role_enum, self.gw_cfg)
        friction = self.cfg.model.friction_seed_primary
        SQRT_252 = math.sqrt(252.0)
        mh_trained = self.mh_model.is_trained()
        out = []
        for symbol in features.index:
            try:
                if mh_trained:
                    mu, eps = self.mh_model.predict(features.loc[symbol])
                    after_cost_edge_bps = float(mu)
                    # Kelly denominator: same policy as parent — max of
                    # realized daily vol and epistemic sigma.
                    rvol_ann = float(features.loc[symbol].get("realized_vol_bps", 0.0))
                    if rvol_ann != rvol_ann:  # NaN
                        rvol_ann = 0.0
                    daily_vol_bps = rvol_ann / SQRT_252 if rvol_ann > 0 else 0.0
                    sigma_total_bps = max(daily_vol_bps, float(eps), 0.0)
                else:
                    after_cost_edge_bps = 0.0
                    sigma_total_bps = 0.0
            except Exception:  # noqa: BLE001
                after_cost_edge_bps = 0.0
                sigma_total_bps = 0.0
            composite = qe.composite_edge(after_cost_edge_bps, membership_bps, friction)
            out.append({
                "symbol": str(symbol),
                "role": "primary",
                "role_enum": role_enum,
                "mu_edge_bps": after_cost_edge_bps,
                "sigma_total_bps": sigma_total_bps,
                "membership_edge_bps": membership_bps,
                "friction_multiplier": friction,
                "composite_edge_bps": composite,
            })
        return out
