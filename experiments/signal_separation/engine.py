"""SignalSeparationBacktestEngine — audit Fix 1 (use μ, not composite,
for Kelly numerator) + Fix 3 (dead-feature ReducedFeatureBayesian).

Overrides:
  - Constructor: swap self.bayes for ReducedFeatureBayesian
  - `_run_curve_sizing_and_routing`: build Candidates using `mu_edge_bps`
    instead of `composite_edge_bps` for the Kelly numerator input.
    Composite is unchanged for gateway execution (§2.22a).

Everything else — snapshot loader, gateway evaluation, top-K, order
routing, persistence, metrics — reused unchanged from parent.
"""
from __future__ import annotations

import math

import numpy as np
import quant_engine as qe

from experiments.signal_separation.training import ReducedFeatureBayesian
from juniauto.backtest.engine import BacktestEngine
from juniauto.config import JuniAutoConfig
from juniauto.portfolio import (
    Candidate,
    compute_target_weights,
    edges_cv,
    select_top_k,
)
from juniauto.utils import get_logger

log = get_logger(__name__)


class SignalSeparationBacktestEngine(BacktestEngine):
    """Backtest engine implementing the two combined audit fixes."""

    def __init__(self, cfg: JuniAutoConfig, **kwargs) -> None:
        # Parent init creates a normal BayesianModel; we replace it with
        # ReducedFeatureBayesian which zeros 11 dead columns.
        super().__init__(cfg, **kwargs)
        self.bayes = ReducedFeatureBayesian(self.db, cfg)
        log.info(
            "sigsep_engine_ready",
            n_samples=self.bayes.n_samples,
            is_trained=self.bayes.is_trained(),
        )

    def _run_curve_sizing_and_routing(
        self,
        curve,
        gw_actions_base,
        snap,
        now,
    ) -> None:
        """Copy of parent _run_curve_sizing_and_routing with ONE change:
        Candidate.conservative_edge_bps uses `mu_edge_bps` (raw Bayesian μ)
        instead of `composite_edge_bps` (μ + 138 constant).

        The composite is still computed upstream and still gates the gateway
        (via _evaluate_gateway → net_edge_bps > hurdle_bps). We're only
        changing what SIZING uses.
        """
        # Deep-copy actions per curve so target_weight annotations don't collide.
        gw_actions = [dict(a) for a in gw_actions_base]

        acct = curve.broker.get_account()
        equity = float(acct["equity"])
        if equity <= 0:
            log.warning("sigsep_curve_bankrupt", curve=curve.name, equity=equity)
            return

        positions = curve.broker.get_positions()
        current_weights = {p["symbol"]: float(p["market_value"]) / equity for p in positions}
        current_qtys = {p["symbol"]: float(p["qty_available"]) for p in positions}

        # ---- FIX 1: Candidates use μ, not composite, for ranking + sizing ----
        # Parent uses composite_edge_bps. That's dominated by the +138
        # membership constant. Using mu_edge_bps directly makes the model
        # signal actually drive the ranking.
        action_by_sym = {str(a["symbol"]): a for a in gw_actions}
        executed_syms = [s for s, a in action_by_sym.items() if a["executed"]]
        candidates = [
            Candidate(
                symbol=sym,
                conservative_edge_bps=float(action_by_sym[sym].get("mu_edge_bps", 0.0)),
                sigma_total_bps=float(action_by_sym[sym].get("sigma_total_bps", 0.0)),
            )
            for sym in executed_syms
        ]

        # ---- Top-K (unchanged from parent) ----
        cv = edges_cv(candidates) if candidates else 0.0
        bayes_trained = bool(self.bayes and self.bayes.is_trained())
        top_k_active = (
            bayes_trained
            and cv >= float(self.cfg.sizing.top_k_activation_cv_threshold)
            and len(candidates) > int(self.cfg.sizing.max_holdings)
        )
        if top_k_active:
            incumbents = {s for s, w in current_weights.items() if w > 0}
            candidates = select_top_k(
                candidates,
                k=int(self.cfg.sizing.max_holdings),
                incumbents=incumbents,
                hysteresis_edge_bps=float(self.cfg.sizing.hysteresis_edge_delta_bps),
            )

        # Log the μ distribution so we can see (in the log) that the
        # signal actually varies across candidates now.
        if candidates:
            mus = [c.conservative_edge_bps for c in candidates]
            log.info(
                "sigsep_mu_distribution",
                n=len(candidates),
                mu_mean=round(float(np.mean(mus)), 2),
                mu_std=round(float(np.std(mus)), 2),
                mu_min=round(float(np.min(mus)), 2),
                mu_max=round(float(np.max(mus)), 2),
                cv=round(cv, 4),
            )

        live = compute_target_weights(
            candidates,
            max_name_weight=self.cfg.sizing.max_name_weight,
            cash_floor=self.cfg.sizing.cash_floor,
        )

        for a in gw_actions:
            sym = str(a["symbol"])
            tw = float(live.weights.get(sym, 0.0))
            cw = float(current_weights.get(sym, 0.0))
            a["target_weight"] = tw
            a["current_weight"] = cw
            a["delta_weight"] = tw - cw
            a["rebalance_kind"] = self._classify_rebalance(a, tw, cw)

        # ---- Order routing (unchanged from parent) ----
        dead_band = float(self.cfg.sizing.rebalance_dead_band)
        submitted_buy = submitted_sell = rejected = held = 0
        for a in gw_actions:
            kind = str(a["rebalance_kind"])
            delta_w = float(a["delta_weight"])
            mid = float(a.get("mid_price", 0.0))
            symbol = str(a["symbol"])
            target_w = float(a["target_weight"])
            is_full_exit = (
                kind == "trim"
                and target_w <= 0.0
                and float(current_qtys.get(symbol, 0.0)) > 0.0
            )
            if kind in ("add", "trim") and abs(delta_w) < dead_band and not is_full_exit:
                held += 1
                continue
            if kind in ("hold", "reject"):
                held += 1
                continue

            if kind == "trim":
                held_qty = float(current_qtys.get(symbol, 0.0))
                if held_qty <= 0.0:
                    held += 1
                    continue
                if curve.pdt is not None:
                    if not curve.pdt.min_hold_satisfied(symbol, now):
                        rejected += 1
                        continue
                    if not curve.pdt.can_close_today(symbol, now):
                        rejected += 1
                        continue
                if is_full_exit:
                    curve.broker.close_position(symbol)
                    submitted_sell += 1
                    continue
                target_qty = 0.0
                if target_w > 0.0 and mid > 0.0:
                    target_qty = target_w * equity / mid
                delta_qty = held_qty - target_qty
                sell_qty = math.floor(delta_qty * 10_000.0) / 10_000.0
                if sell_qty <= 0.0:
                    held += 1
                    continue
                curve.broker.submit_market(symbol, sell_qty, "sell")
                submitted_sell += 1
                continue

            if mid <= 0.0:
                rejected += 1
                continue
            notional_abs = abs(delta_w) * equity
            qty = round(notional_abs / mid, 4)
            if qty <= 0.0:
                held += 1
                continue
            curve.broker.submit_market(symbol, qty, "buy")
            submitted_buy += 1

        self._persist_gateway_actions(gw_actions, now, curve.name)
        mark_prices: dict[str, float] = {}
        for sym, bars_list in snap.bars.items():
            if bars_list:
                mark_prices[sym] = float(bars_list[-1].close)
        curve.broker.mark_to_market(mark_prices)
        self._persist_equity_point(curve, now)
        self._persist_positions(curve, now, live.weights)
        curve.prev_target_weights = dict(live.weights)
