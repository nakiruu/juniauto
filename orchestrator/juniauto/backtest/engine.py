"""Event-driven bar-by-bar backtest engine.

Composes the shipped live-side modules:
    - signals.compute_all + MarketRegimeSignals
    - bayesian.BayesianModel + resolve_stale_executions (adapted for sim)
    - qe.evaluate_gateway / qe.compute_cost / qe.concentration_penalty_bps
    - portfolio.compute_target_weights + select_top_k + edges_cv + fixed_equal_weights
    - execution.PDTTracker (for the "main" curve)

Coordinator-mandated behaviors:
    - PDT-enforced "main" curve + parallel "unconstrained" curve share the
      same signals/predictions/gateway evaluations but diverge at sizing +
      order routing (the top-K incumbents differ per curve).
    - Walk-forward Bayesian retrain every N trading days with 1-day embargo
      and purged CV (executions with fill_ts >= now - horizon are excluded).
    - Fill via SimBroker per configured fill_model (default next_open).
    - All state persisted to backtest_* tables keyed by run_id.
    - Prometheus is NOT touched — those gauges are live-loop instrumentation
      and would cross-contaminate live dashboards.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time as _time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd
import quant_engine as qe

from juniauto.backtest.broker import SimBroker, SimBrokerFill
from juniauto.backtest.clock import SimClock
from juniauto.backtest.loader import HistoricalSnapshotLoader
from juniauto.bayesian import BayesianModel
from juniauto.config import JuniAutoConfig
from juniauto.data.yahoo_feed import YahooFeed
from juniauto.db import QuestDBClient
from juniauto.execution.pdt import PDTTracker
from juniauto.portfolio import (
    Candidate,
    compute_target_weights,
    edges_cv,
    fixed_equal_weights,
    select_top_k,
)
from juniauto.signals import MarketRegimeSignals, compute_all
from juniauto.utils import get_logger
from juniauto.utils.time_utils import ET, quote_age_sessions, session_of

log = get_logger(__name__)


CURVE_MAIN = "main"
CURVE_UNCONSTRAINED = "unconstrained"


@dataclass
class _CurveState:
    """Per-curve mutable state (broker + PDT + last-cycle weights)."""
    name: str                              # "main" | "unconstrained"
    broker: SimBroker
    pdt: PDTTracker | None                 # None => unenforced
    prev_target_weights: dict[str, float]  # last cycle's target weights for top-K incumbents


class _NoopPDT:
    """PDT stand-in that always permits — used for the unconstrained curve."""

    def count_in_window(self, _now: datetime | None = None) -> int:
        return 0

    def can_close_today(self, _symbol: str, _now: datetime | None = None) -> bool:
        return True

    def min_hold_satisfied(self, _symbol: str, _now: datetime | None = None) -> bool:
        return True

    def note_open(self, _symbol: str, _ts: datetime) -> None:
        return None

    def note_close(self, _symbol: str, _open_ts: datetime, _close_ts: datetime):  # noqa: ANN201
        return None


class BacktestEngine:
    """Run a historical backtest over `[start_date, end_date]`."""

    def __init__(
        self,
        cfg: JuniAutoConfig,
        *,
        run_id: str,
        start_date: date,
        end_date: date,
        fill_model: str = "next_open",
        walkforward_days: int = 21,
        initial_cash: float = 10_000.0,
        universe: list[str] | None = None,
        cli_args: str = "",
        notes: str = "",
    ) -> None:
        self.cfg = cfg
        self.run_id = run_id
        self.start_date = start_date
        self.end_date = end_date
        self.fill_model = fill_model
        self.walkforward_days = int(walkforward_days)
        self.initial_cash = float(initial_cash)
        self.cli_args = cli_args
        self.notes = notes

        # Universe pinning: coordinator Q6 default = today's seed.
        self.universe = list(universe) if universe else list(cfg.universe.symbols)
        if not self.universe:
            raise ValueError(
                "Backtest universe is empty. Set config.universe.symbols or pass --universe."
            )
        # SPY must be in the bars fetch for regime signal + CAPM benchmark.
        if cfg.regime.reference_symbol not in self.universe:
            self.universe.append(cfg.regime.reference_symbol)

        # Infrastructure
        self.db = QuestDBClient(cfg.database)
        # NO Yahoo in backtest. The live-side backfill's Bayesian training
        # already baked fundamentals into the model, and the fundamental
        # signal family gracefully returns empty when fundamentals=None.
        # Calling YahooFeed per cycle was costing ~25s (10 cold symbols x
        # 8s per-symbol timeout across 4 workers), turning a 1144-cycle
        # backtest into a 9-hour ordeal. Disabling drops per-cycle time
        # from ~28s to ~2s and total wall-time from ~9h to ~45min.
        self.yahoo = None
        self.loader = HistoricalSnapshotLoader(
            self.db, None, history_bars=cfg.alpaca.history_bars
        )
        self.clock = SimClock(start_date, end_date)

        # C++ engine configs
        self.gw_cfg = qe.GatewayConfig()
        self.cost_cfg = qe.CostConfig()

        # Bayesian — reuses the SAME class as live; walk-forward retrains
        # explicitly via _maybe_retrain_bayes below.
        try:
            self.bayes = BayesianModel(self.db, cfg)
        except Exception as e:  # noqa: BLE001
            log.warning("backtest_bayes_init_failed", error=str(e))
            self.bayes = None  # type: ignore[assignment]

        # Regime observation (same defaults as live; observation-only during backtest too)
        self.regime = MarketRegimeSignals(cfg.regime)
        self._prev_regime_ema: float | None = None

        # Two-curve state
        broker_main = SimBroker(initial_cash, fill_model=fill_model)
        broker_uncon = SimBroker(initial_cash, fill_model=fill_model)
        broker_main.set_bars_provider(self.loader.next_bar)
        broker_uncon.set_bars_provider(self.loader.next_bar)
        self.curves: list[_CurveState] = [
            _CurveState(CURVE_MAIN, broker_main, PDTTracker(), {}),
            _CurveState(CURVE_UNCONSTRAINED, broker_uncon, _NoopPDT(), {}),  # type: ignore[list-item]
        ]

        # Walk-forward state
        self._last_retrain_day_index: int = -10 ** 9  # forces initial train

        # Cycle counter
        self.n_cycles_completed: int = 0

    # ================================================================
    # Public API
    # ================================================================
    def run(self) -> None:
        started_at = datetime.now(tz=ET)
        log.info(
            "backtest_start",
            run_id=self.run_id,
            start=str(self.start_date), end=str(self.end_date),
            fill_model=self.fill_model, walkforward_days=self.walkforward_days,
            initial_cash=self.initial_cash,
            n_trading_days=len(self.clock),
            n_universe=len(self.universe),
        )
        self._write_metadata_open(started_at)

        # Preload ALL bars for the window into memory in ONE query. This
        # replaces the ~1144 per-cycle range queries that were dropping
        # QuestDB PG-wire connections mid-response. Include 380 calendar
        # days of history before start_date so the earliest cycles have
        # enough lookback for signal computation.
        earliest = datetime.combine(self.start_date - timedelta(days=int(self.cfg.alpaca.history_bars * 1.5)),
                                    datetime.min.time(), tzinfo=ET)
        latest = datetime.combine(self.end_date + timedelta(days=2),
                                  datetime.max.time(), tzinfo=ET)
        n_bars = self.loader.preload_all_bars(self.universe, earliest, latest)
        if n_bars == 0:
            log.error("backtest_preload_empty_aborting", run_id=self.run_id)
            return

        wall_t0 = _time.monotonic()
        while True:
            try:
                self._run_one_cycle()
            except Exception as e:  # noqa: BLE001
                log.error(
                    "backtest_cycle_failed",
                    ts=str(self.clock.now()),
                    error=str(e), error_type=type(e).__name__,
                )
            self.n_cycles_completed += 1
            if not self.clock.advance():
                break

        # Final settlement pass — any pending orders from the last cycle fill
        # on the NEXT trading day, but we have no next day in the window. Mark
        # them as unfilled and let the metrics layer see them as such.
        wall_secs = _time.monotonic() - wall_t0
        ended_at = datetime.now(tz=ET)
        log.info(
            "backtest_done",
            run_id=self.run_id,
            n_cycles=self.n_cycles_completed,
            wall_secs=round(wall_secs, 1),
            per_cycle_ms=round(1000.0 * wall_secs / max(1, self.n_cycles_completed), 1),
            main_equity=self.curves[0].broker.get_account()["equity"],
            uncon_equity=self.curves[1].broker.get_account()["equity"],
        )
        self._write_metadata_close(started_at, ended_at)

    # ================================================================
    # Per-cycle body
    # ================================================================
    def _run_one_cycle(self) -> None:
        now = self.clock.now()
        # 1) Settle any orders queued from the previous cycle. Fills happen at
        #    THIS cycle's date under fill_model rules (next_open ⇒ today's open).
        for curve in self.curves:
            fills = curve.broker.settle(now)
            self._persist_fills(fills, now, curve.name)
            # Update PDT tracker for the main curve (unconstrained PDT is Noop).
            for f in fills:
                if f.side == "buy":
                    curve.pdt.note_open(f.symbol, f.ts) if curve.pdt else None

        # 2) Walk-forward Bayesian retrain (executions from prior day 1-embargoed).
        self._maybe_retrain_bayes(now)

        # 3) Snapshot + features (shared across curves)
        snap = self.loader.snapshot(self.universe, now)
        bars_df = snap.bars_df()
        if bars_df.empty:
            log.info("cycle_no_bars", ts=str(now))
            return
        try:
            features = compute_all(
                bars=bars_df,
                fundamentals=snap.fundamentals,
                quotes=snap.quotes,  # empty in backtest
                as_of_date=now.date(),
                halflife_event_days=self.cfg.freshness_halflife_days["event"],
            )
        except Exception as e:  # noqa: BLE001
            log.warning("cycle_features_failed", ts=str(now), error=str(e))
            return
        if features.empty:
            return

        # 4) Regime observation (persist to backtest_market_regime; unlike live
        #    we don't hit Prometheus). Observation-only.
        try:
            snapshot = self.regime.compute(bars_df, prev_stress_ema=self._prev_regime_ema)
            self._prev_regime_ema = (
                snapshot.stress_ema if not math.isnan(snapshot.stress_ema) else self._prev_regime_ema
            )
            self._persist_regime(snapshot, now)
        except Exception as e:  # noqa: BLE001
            log.warning("cycle_regime_failed", ts=str(now), error=str(e))

        # 5) Predictions (Bayesian μ, σ) + composite edge — shared
        predictions = self._compute_predictions(features)
        self._persist_predictions(predictions, now)

        # 6) Gateway evaluations — shared across curves (they only differ at sizing/routing)
        cycle_session = session_of(now)
        gw_actions_base = [
            self._evaluate_gateway(
                pred=p, snap=snap, now=now, session=cycle_session,
                # For backtest: no open orders (SimBroker settles synchronously each cycle).
                open_symbols=set(),
                # Wide-spread guard doesn't apply — we don't have quotes. Ignored.
            )
            for p in predictions
        ]

        # 7) Per-curve sizing + routing. Each curve maintains its own
        #    broker.get_positions() view, so incumbents differ ⇒ top-K differs.
        for curve in self.curves:
            self._run_curve_sizing_and_routing(curve, gw_actions_base, snap, now)

    # ================================================================
    # Per-curve sizing + routing
    # ================================================================
    def _run_curve_sizing_and_routing(
        self,
        curve: _CurveState,
        gw_actions_base: list[dict[str, object]],
        snap,  # MarketSnapshot
        now: datetime,
    ) -> None:
        # Deep-copy actions per curve so target_weight annotations don't collide.
        gw_actions = [dict(a) for a in gw_actions_base]

        acct = curve.broker.get_account()
        equity = float(acct["equity"])
        if equity <= 0:
            log.warning("curve_bankrupt", curve=curve.name, equity=equity)
            return

        # Current weights from THIS curve's broker
        positions = curve.broker.get_positions()
        current_weights = {p["symbol"]: float(p["market_value"]) / equity for p in positions}
        current_qtys = {p["symbol"]: float(p["qty_available"]) for p in positions}

        # ---- 5.5: top-K + Kelly ----
        action_by_sym = {str(a["symbol"]): a for a in gw_actions}
        executed_syms = [s for s, a in action_by_sym.items() if a["executed"]]
        candidates = [
            Candidate(
                symbol=sym,
                conservative_edge_bps=float(action_by_sym[sym].get("composite_edge_bps", 138.0)),
                sigma_total_bps=float(action_by_sym[sym].get("sigma_total_bps", 0.0)),
            )
            for sym in executed_syms
        ]
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

        # ---- 6: order routing via SimBroker ----
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
                # PDT gate on the main curve only
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

            # add / entry
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

        # ---- Persist per-curve records ----
        self._persist_gateway_actions(gw_actions, now, curve.name)
        # Mark broker to today's close prices for next-cycle equity computation
        mark_prices: dict[str, float] = {}
        for sym, bars_list in snap.bars.items():
            if bars_list:
                mark_prices[sym] = float(bars_list[-1].close)
        curve.broker.mark_to_market(mark_prices)
        # Record equity curve point
        self._persist_equity_point(curve, now)
        # Snapshot positions
        self._persist_positions(curve, now, live.weights)

        curve.prev_target_weights = dict(live.weights)

    # ================================================================
    # Bayesian walk-forward
    # ================================================================
    def _maybe_retrain_bayes(self, now: datetime) -> None:
        # NO-OP during backtest. Previous implementation called
        # bayes.retrain_from_db() every 21 days, but:
        #   (a) that queries the LIVE `executions` table, not
        #       backtest_executions, so it's re-reading the SAME backfilled
        #       training set every retrain — no actual walk-forward learning,
        #   (b) each retrain query has been unreliable (partition-file errors,
        #       fd-closed errors from QuestDB's internal state),
        #   (c) it was silently reducing every cycle to cold-start
        #       predictions anyway (n_samples=0 in every retrain log).
        # The Bayesian model loaded at engine __init__ (from the same
        # backfilled data) is used unchanged for the whole backtest. True
        # walk-forward would require querying backtest_executions with a
        # proper embargo — deferred until we have a stable backtest run.
        return

    # ================================================================
    # Predictions + gateway evaluation (adapted from live main.py)
    # ================================================================
    def _compute_predictions(self, features: pd.DataFrame) -> list[dict[str, object]]:
        role_enum = qe.Role.Primary
        membership_bps = qe.membership_edge_bps(role_enum, self.gw_cfg)
        friction = self.cfg.model.friction_seed_primary
        SQRT_252 = math.sqrt(252.0)
        bayes_trained = bool(self.bayes and self.bayes.is_trained())
        out = []
        for symbol in features.index:
            try:
                if bayes_trained and self.bayes is not None:
                    mu, eps = self.bayes.predict(features.loc[symbol])
                    after_cost_edge_bps = float(mu)
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

    def _evaluate_gateway(
        self, *, pred, snap, now, session, open_symbols
    ) -> dict[str, object]:
        symbol = str(pred["symbol"])
        bars_list = snap.bars.get(symbol, [])
        if not bars_list:
            return self._reject(pred, "no_bars")
        last_bar = bars_list[-1]
        mid = float(last_bar.close)  # backtest proxy for mid (no live quotes)
        if mid <= 0:
            return self._reject(pred, "no_price")

        state = qe.MarketState()
        state.mid_price = mid
        state.spread_bps = 5.0  # optimistic constant spread — replace with per-symbol later
        state.volatility_bps = self._annualized_vol_bps(bars_list)
        state.bar_dollar_volume = mid * float(last_bar.volume or 0)
        state.adv_dollar = self._adv_dollar(bars_list)
        # In backtest we don't have quote_age; treat as fresh.
        state.quote_age_sessions = 0.0
        state.gap_days_to_next_session = 0
        state.session_multiplier = self.cfg.costs.session_multiplier.get(session, 1.0)
        state.adverse_selection_share = self.cfg.costs.adverse_selection_share.get(session, 0.35)

        slippage = qe.SlippageStats()
        slippage.recent_fill_slippage_bps = 0.0

        target_weight = min(0.05, self.cfg.sizing.max_name_weight)
        notional = target_weight * float(self.curves[0].broker.get_account()["equity"])

        order = qe.Order()
        order.symbol = symbol
        order.notional = notional
        order.predicted_holding_seconds = float(self.cfg.costs.action_memory.horizon_seconds)
        order.has_open_order = symbol in open_symbols

        evaluation = qe.evaluate_gateway(
            symbol,
            pred["role_enum"],
            float(pred["mu_edge_bps"]),
            float(pred["friction_multiplier"]),
            state, slippage, self.cost_cfg, self.gw_cfg, qe.ActionType.BUY,
        )
        cost = qe.compute_cost(order, state, slippage, self.cost_cfg, float(pred["composite_edge_bps"]))
        executed = bool(evaluation.executes())
        return {
            "symbol": symbol,
            "action_type": "BUY",
            "role": str(pred["role"]),
            "gross_edge_bps": float(evaluation.gross_edge_bps),
            "entry_cost_bps": float(cost.entry_bps),
            "exit_cost_reserved": float(cost.exit_reserved_bps),
            "queue_delay_bps": float(cost.queue_delay_bps),
            "cancel_replace_bps": float(cost.cancel_replace_bps),
            "action_memory_bps": float(cost.action_memory_bps),
            "cash_waiting_value": float(cost.cash_waiting_value_bps),
            "operational_bps": float(cost.operational_bps),
            "total_cost_bps": float(evaluation.total_cost_bps),
            "net_edge_bps": float(evaluation.net_edge_bps),
            "hurdle_bps": float(self.cfg.model.minimum_hurdle_bps),
            "friction_multiplier": float(pred["friction_multiplier"]),
            "executed": executed,
            "reject_reason": "" if executed else "net_edge_below_hurdle",
            "notional": notional,
            "mid_price": mid,
            "mu_edge_bps": float(pred["mu_edge_bps"]),
            "sigma_total_bps": float(pred["sigma_total_bps"]),
            "composite_edge_bps": float(pred["composite_edge_bps"]),
        }

    @staticmethod
    def _reject(pred, reason: str) -> dict[str, object]:
        return {
            "symbol": str(pred["symbol"]),
            "action_type": "HOLD",
            "role": str(pred["role"]),
            "gross_edge_bps": 0.0, "entry_cost_bps": 0.0, "exit_cost_reserved": 0.0,
            "queue_delay_bps": 0.0, "cancel_replace_bps": 0.0, "action_memory_bps": 0.0,
            "cash_waiting_value": 0.0, "operational_bps": 0.0, "total_cost_bps": 0.0,
            "net_edge_bps": 0.0, "hurdle_bps": 0.0,
            "friction_multiplier": float(pred["friction_multiplier"]),
            "executed": False, "reject_reason": reason,
            "notional": 0.0, "mid_price": 0.0,
            "mu_edge_bps": float(pred["mu_edge_bps"]),
            "sigma_total_bps": float(pred["sigma_total_bps"]),
            "composite_edge_bps": float(pred["composite_edge_bps"]),
        }

    @staticmethod
    def _classify_rebalance(action: dict[str, object], tw: float, cw: float) -> str:
        if cw > 0.0 and tw <= 0.0:
            return "trim"
        if cw > 0.0 and tw > cw:
            return "add"
        if cw > 0.0 and tw < cw:
            return "trim"
        if cw <= 0.0 and tw > 0.0 and action["executed"]:
            return "entry"
        if not action["executed"]:
            return "reject"
        return "hold"

    @staticmethod
    def _annualized_vol_bps(bars_list) -> float:
        closes = [float(b.close) for b in bars_list if b.close and b.close > 0]
        if len(closes) < 2:
            return 0.0
        rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        n = len(rets)
        mean = sum(rets) / n
        var = sum((r - mean) ** 2 for r in rets) / max(1, n - 1)
        return 10_000.0 * math.sqrt(max(0.0, var) * 252.0)

    @staticmethod
    def _adv_dollar(bars_list, window: int = 20) -> float:
        recent = bars_list[-window:]
        if not recent:
            return 0.0
        vals = [float(b.close) * float(b.volume or 0) for b in recent if b.close and b.volume]
        return sum(vals) / len(vals) if vals else 0.0

    # ================================================================
    # Persistence (ILP writes to backtest_* tables, always with run_id)
    # ================================================================
    def _persist_gateway_actions(self, actions, ts: datetime, curve_type: str) -> None:
        if not actions:
            return
        with self.db.sender() as s:
            for a in actions:
                s.row(
                    "backtest_gateway_actions",
                    symbols={
                        "run_id": self.run_id, "curve_type": curve_type,
                        "symbol": str(a["symbol"]),
                        "action_type": str(a["action_type"]),
                        "role": str(a["role"]),
                        "horizon": "1d",
                        "reject_reason": str(a.get("reject_reason", "")) or "none",
                        "rebalance_kind": str(a.get("rebalance_kind", "reject")),
                    },
                    columns={
                        "gross_edge_bps": float(a["gross_edge_bps"]),
                        "entry_cost_bps": float(a["entry_cost_bps"]),
                        "exit_cost_reserved": float(a["exit_cost_reserved"]),
                        "queue_delay_bps": float(a["queue_delay_bps"]),
                        "cancel_replace_bps": float(a["cancel_replace_bps"]),
                        "action_memory_bps": float(a["action_memory_bps"]),
                        "cash_waiting_value": float(a["cash_waiting_value"]),
                        "operational_bps": float(a["operational_bps"]),
                        "total_cost_bps": float(a["total_cost_bps"]),
                        "net_edge_bps": float(a["net_edge_bps"]),
                        "hurdle_bps": float(a["hurdle_bps"]),
                        "friction_multiplier": float(a["friction_multiplier"]),
                        "executed": bool(a["executed"]),
                        "target_weight": float(a.get("target_weight", 0.0)),
                        "current_weight": float(a.get("current_weight", 0.0)),
                        "delta_weight": float(a.get("delta_weight", 0.0)),
                    },
                    at=ts,
                )

    def _persist_predictions(self, preds, ts: datetime) -> None:
        if not preds:
            return
        with self.db.sender() as s:
            for p in preds:
                s.row(
                    "backtest_predictions",
                    symbols={
                        "run_id": self.run_id, "symbol": str(p["symbol"]),
                        "horizon": "1d", "role": str(p["role"]),
                    },
                    columns={
                        "mu_edge_bps": float(p["mu_edge_bps"]),
                        "sigma_edge_bps": float(p["sigma_total_bps"]),
                        "sigma_total_bps": float(p["sigma_total_bps"]),
                        "p_positive": 0.5,
                        "conservative_edge": float(p["mu_edge_bps"]),
                        "membership_edge": float(p["membership_edge_bps"]),
                        "composite_edge": float(p["composite_edge_bps"]),
                    },
                    at=ts,
                )

    def _persist_fills(self, fills: list[SimBrokerFill], ts: datetime, curve_type: str) -> None:
        if not fills:
            return
        with self.db.sender() as s:
            for f in fills:
                s.row(
                    "backtest_executions",
                    symbols={
                        "run_id": self.run_id, "curve_type": curve_type,
                        # order_id is STRING; must NOT be in symbols={}
                        "symbol": f.symbol,
                        "action_type": "BUY" if f.side == "buy" else "SELL",
                        "side": f.side, "horizon": "1d",
                        "fill_model": self.fill_model,
                    },
                    columns={
                        "order_id": f.order_id,
                        "qty": float(f.qty),
                        "fill_price": float(f.fill_price),
                        "decision_ref_price": float(f.decision_ref_price),
                        "slippage_bps": float(f.slippage_bps),
                        "spread_bps": 0.0,
                        "model_edge_bps": 0.0,
                        "day_trade": False,
                    },
                    at=ts,
                )

    def _persist_equity_point(self, curve: _CurveState, ts: datetime) -> None:
        acct = curve.broker.get_account()
        equity = float(acct["equity"])
        cash = float(acct["cash"])
        invested = max(0.0, equity - cash)
        positions = curve.broker.get_positions()
        # Daily return vs previous equity in same curve: compute from equity_curve
        # table on read-side (metrics stage); write raw here.
        with self.db.sender() as s:
            s.row(
                "backtest_equity_curve",
                symbols={"run_id": self.run_id, "curve_type": curve.name},
                columns={
                    "equity": equity,
                    "cash": cash,
                    "invested": invested,
                    "position_count": len(positions),
                    "pdt_day_trade_count": int(curve.pdt.count_in_window(ts) if curve.pdt else 0),
                    "pdt_blocked": bool(curve.pdt and curve.pdt.count_in_window(ts) >= 3),
                    "daily_return_bps": 0.0,   # filled in by metrics stage
                    "cum_return_bps": 10_000.0 * (equity / self.initial_cash - 1.0),
                },
                at=ts,
            )

    def _persist_positions(
        self, curve: _CurveState, ts: datetime, target_weights: dict[str, float]
    ) -> None:
        acct = curve.broker.get_account()
        equity = float(acct["equity"])
        positions = curve.broker.get_positions()
        if not positions:
            return
        with self.db.sender() as s:
            for p in positions:
                sym = p["symbol"]
                actual_w = float(p["market_value"]) / equity if equity > 0 else 0.0
                target_w = float(target_weights.get(sym, 0.0))
                s.row(
                    "backtest_positions",
                    symbols={"run_id": self.run_id, "curve_type": curve.name, "symbol": sym},
                    columns={
                        "qty": float(p["qty"]),
                        "avg_entry_price": float(p["avg_entry_price"]),
                        "market_value": float(p["market_value"]),
                        "unrealized_pl": float(p["unrealized_pl"]),
                        "target_weight": target_w,
                        "actual_weight": actual_w,
                        "weight_drift_bps": 10_000.0 * (actual_w - target_w),
                    },
                    at=ts,
                )

    def _persist_regime(self, snap, ts: datetime) -> None:
        columns = {
            "gamma_multiplier": float(snap.gamma_multiplier),
            "n_symbols_corr": int(snap.n_symbols_corr),
        }
        for k, v in (
            ("spy_drawdown_pct", snap.spy_drawdown_pct),
            ("spy_vol_pct_rank", snap.spy_vol_pct_rank),
            ("avg_pairwise_corr", snap.avg_pairwise_corr),
            ("stress_raw", snap.stress_raw),
            ("stress_ema", snap.stress_ema),
        ):
            if v == v:  # not NaN
                columns[k] = float(v)
        with self.db.sender() as s:
            s.row(
                "backtest_market_regime",
                symbols={"run_id": self.run_id, "cycle_type": "backtest"},
                columns=columns,
                at=ts,
            )

    def _persist_bayes_snapshot(self, ts: datetime, n_samples: int) -> None:
        # Minimal shape — the full per-group posterior can be added later once
        # BayesianModel exposes an inspection API.
        try:
            y_mean = float(getattr(self.bayes, "y_mean", 0.0))
            y_std = float(getattr(self.bayes, "y_std", 0.0))
        except Exception:  # noqa: BLE001
            y_mean, y_std = 0.0, 0.0
        with self.db.sender() as s:
            s.row(
                "backtest_bayes_snapshots",
                symbols={"run_id": self.run_id, "group_id": "all"},
                columns={
                    "n_samples": int(n_samples),
                    "y_mean": y_mean, "y_std": y_std,
                    "gamma": 0.0, "beta_mean": 0.0, "beta_var": 0.0,
                    "tau": 0.0, "n_eff": float(n_samples),
                    "utility_score": 0.0,
                },
                at=ts,
            )

    # ================================================================
    # Metadata (open/close)
    # ================================================================
    def _write_metadata_open(self, started_at: datetime) -> None:
        cfg_hash = hashlib.sha256(
            json.dumps(self._cfg_snapshot(), sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        git_sha = ""
        try:
            import subprocess
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=2
            ).decode("ascii").strip()[:12]
        except Exception:  # noqa: BLE001
            pass
        start_ts = datetime.combine(self.start_date, datetime.min.time(), tzinfo=ET)
        end_ts = datetime.combine(self.end_date, datetime.min.time(), tzinfo=ET)
        with self.db.sender() as s:
            s.row(
                "backtest_metadata",
                symbols={"run_id": self.run_id, "fill_model": self.fill_model},
                columns={
                    "start_date": start_ts,
                    "end_date": end_ts,
                    "cli_args": self.cli_args,
                    "config_hash": cfg_hash,
                    "git_sha": git_sha,
                    "walkforward_days": int(self.walkforward_days),
                    "universe_size": int(len(self.universe)),
                    "n_cycles": 0,
                    "notes": self.notes,
                    "ended_at": started_at,  # placeholder; overwritten at close
                },
                at=started_at,
            )

    def _write_metadata_close(self, started_at: datetime, ended_at: datetime) -> None:
        # QuestDB WAL is append-only for symbol tables; the "update" here is a
        # second row that later queries can join to on run_id (latest wins).
        start_ts = datetime.combine(self.start_date, datetime.min.time(), tzinfo=ET)
        end_ts = datetime.combine(self.end_date, datetime.min.time(), tzinfo=ET)
        with self.db.sender() as s:
            s.row(
                "backtest_metadata",
                symbols={"run_id": self.run_id, "fill_model": self.fill_model},
                columns={
                    "start_date": start_ts,
                    "end_date": end_ts,
                    "cli_args": self.cli_args,
                    "config_hash": "",
                    "git_sha": "",
                    "walkforward_days": int(self.walkforward_days),
                    "universe_size": int(len(self.universe)),
                    "n_cycles": int(self.n_cycles_completed),
                    "notes": f"CLOSE: {self.notes}",
                    "ended_at": ended_at,
                },
                at=ended_at,
            )

    def _cfg_snapshot(self) -> dict:
        # Minimal fingerprint of the sections that actually shape the sim.
        return {
            "sizing": self.cfg.sizing.model_dump(),
            "bayesian": self.cfg.bayesian.model_dump(),
            "regime": self.cfg.regime.model_dump(),
            "universe_size": len(self.universe),
            "history_bars": self.cfg.alpaca.history_bars,
        }
