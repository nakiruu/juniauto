"""JuniAuto entry point.

Runs the daily decision cycle at 15:55 ET (per §2.19). Everything else — feeds,
Bayesian updates, cost model, gateway, execution — hangs off this one scheduler.
"""
from __future__ import annotations

import argparse
import asyncio
import signal
from datetime import datetime
from pathlib import Path

import pandas as pd
import quant_engine as qe
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from prometheus_client import start_http_server

from juniauto.config import JuniAutoConfig, load_config
from juniauto.data import AlpacaFeed, DataAggregator, UniverseBuilder, YahooFeed
from juniauto.db import QuestDBClient
from juniauto.execution import OrderManager, PDTTracker
from juniauto.portfolio import Candidate, compute_target_weights
from juniauto.replay import ReplayHarness
from juniauto.signals import compute_all
from juniauto.utils import configure_logging, get_logger
from juniauto.utils.time_utils import ET, quote_age_sessions, session_of

log = get_logger(__name__)


class JuniAuto:
    def __init__(self, cfg: JuniAutoConfig) -> None:
        self.cfg = cfg
        self.db = QuestDBClient(cfg.database)
        self.alpaca = AlpacaFeed(cfg.alpaca)
        self.yahoo = YahooFeed(
            ttl_days=cfg.yahoo.fundamentals_ttl_days,
            enabled=cfg.yahoo.enabled,
            max_workers=cfg.yahoo.max_workers,
            per_symbol_timeout_seconds=cfg.yahoo.per_symbol_timeout_seconds,
        )
        self.pdt = PDTTracker()
        self.universe = UniverseBuilder(self.alpaca._trading, cfg.universe)  # type: ignore[attr-defined]
        self.aggregator = DataAggregator(cfg, self.alpaca, self.yahoo, self.db)
        self.order_mgr = OrderManager(self.alpaca, self.db, self.pdt)
        self.replay = ReplayHarness(cfg, self.db)
        # C++ engine configs — defaults already match production.yaml, but
        # instantiate here so the object lifetime spans the whole process.
        self.gw_cfg = qe.GatewayConfig()
        self.cost_cfg = qe.CostConfig()
        self.sizing_cfg = qe.SizingConfig()
        self._sched = AsyncIOScheduler(timezone=ET)
        self._stop = asyncio.Event()

    # ---- Lifecycle ----
    async def start(self) -> None:
        log.info(
            "startup",
            version=self.cfg.system.version,
            env=self.cfg.system.environment,
            paper=self.cfg.alpaca.paper,
            trading_enabled=self.cfg.system.trading_enabled,
        )
        if self.cfg.system.trading_enabled and not self.cfg.alpaca.paper:
            log.warning(
                "LIVE_TRADING_ARMED",
                message="trading_enabled=true AND paper=false — real orders will be sent to Alpaca live account",
            )

        # ensure schema exists (idempotent)
        schema = Path(__file__).parent / "db" / "schema.sql"
        try:
            self.db.apply_schema(schema)
            log.info("schema_applied", path=str(schema))
        except Exception as e:
            log.error("schema_apply_failed", error=str(e))
            raise

        if self.cfg.metrics.enabled:
            start_http_server(self.cfg.metrics.port)
            log.info("metrics_server_started", port=self.cfg.metrics.port)

        # boot-time replay smoke check (§3.5 — read-only, safe on paper + prod)
        try:
            r = self.replay.replay_last()
            log.info("boot_replay", ok=r.ok, n_actions=r.n_actions, n_deltas=len(r.deltas))
        except Exception as e:  # noqa: BLE001
            log.warning("boot_replay_failed", error=str(e))

        # daily decision tick at 15:55 ET, Mon-Fri
        hh, mm = self.cfg.model.decision_time_et.split(":")
        self._sched.add_job(
            self._daily_decision_cycle,
            CronTrigger(day_of_week="mon-fri", hour=int(hh), minute=int(mm), timezone=ET),
            id="daily_decision_cycle",
            replace_existing=True,
        )
        # slow-feedback resolution loop hourly (fill outcomes, shadow updates)
        self._sched.add_job(
            self._resolution_loop,
            CronTrigger(minute="5", timezone=ET),
            id="hourly_resolution",
            replace_existing=True,
        )
        self._sched.start()
        log.info("scheduler_started", jobs=[j.id for j in self._sched.get_jobs()])

        # Snapshot immediately at boot so Grafana has data without waiting an hour.
        try:
            self._persist_account_snapshot()
        except Exception as e:  # noqa: BLE001
            log.warning("boot_snapshot_failed", error=str(e))

        await self._stop.wait()
        await self._shutdown()

    async def _shutdown(self) -> None:
        log.info("shutdown_begin")
        self._sched.shutdown(wait=False)
        log.info("shutdown_complete")

    def request_stop(self) -> None:
        self._stop.set()

    # ---- Cycles ----
    async def _daily_decision_cycle(self) -> None:
        """§3.2 seven-step decision cycle.

        1) refresh observations                 (wired: bars/quotes/fundamentals)
        2) source-package selector              (stub)
        3) target gateway → candidates          (stub)
        4) provenance membership → prior action (stub)
        5) execution gate (posterior + costs)   (stub)
        6) order routing                        (stub)
        7) outcome recording                    (stub)
        """
        now = datetime.now(tz=ET)
        log.info("cycle_start", ts=now.isoformat())

        # Guardrail: don't trade if PDT will block us for the whole cycle.
        acct = self.alpaca.get_account()
        dt_count = self.pdt.count_in_window(now)
        log.info(
            "account",
            equity=acct["equity"],
            cash=acct["cash"],
            day_trade_count=dt_count,
            paper=self.cfg.alpaca.paper,
        )

        # --- Step 1: refresh observations (§1.1-§1.3, §3.2 step 1) ---
        symbols = self._resolve_universe()
        log.info("step1_universe", n_symbols=len(symbols), source="config_seed" if self.cfg.universe.symbols else "tape_filter")
        try:
            snap = self.aggregator.snapshot(symbols, now)
            log.info(
                "step1_snapshot",
                bars_symbols=len(snap.bars),
                quotes_symbols=len(snap.quotes),
                fundamentals_symbols=len(snap.fundamentals),
            )
        except Exception as e:  # noqa: BLE001
            log.error("step1_snapshot_failed", error=str(e), error_type=type(e).__name__)
            return  # can't proceed without observations

        # --- Step 2: build feature vectors from the six signal families (§1.4) ---
        try:
            features = compute_all(
                bars=snap.bars_df(),
                fundamentals=snap.fundamentals,
                quotes=snap.quotes,
                as_of_date=now.date(),
                halflife_event_days=self.cfg.freshness_halflife_days["event"],
            )
            log.info(
                "step2_features",
                n_symbols=len(features),
                n_features=len(features.columns),
            )
            self._persist_features(features, now)
        except Exception as e:  # noqa: BLE001
            log.error("step2_features_failed", error=str(e), error_type=type(e).__name__)
            return

        # --- Step 3: source-package selector (§2.20) ---
        # MVP: single "default" package always active. Multi-source
        # G_p,t = decay * G_p,t-1 + realized_evidence with >=95bps threshold
        # is deferred until we have realized outcomes across multiple sources.
        active_package = "default"
        log.info("step3_source", package=active_package)

        # --- Step 4: provenance membership + composite edge (§2.22-2.22a) ---
        # MVP: every candidate gets role=primary (460 bps prior).
        # Bayesian posterior returns 0 on cold start — no realized labels yet.
        # composite_edge = after_cost_edge + membership_bps * friction_multiplier.
        role_enum = qe.Role.Primary
        membership_bps = qe.membership_edge_bps(role_enum, self.gw_cfg)
        friction = self.cfg.model.friction_seed_primary
        predictions = []
        for symbol in features.index:
            after_cost_edge_bps = 0.0  # cold-start ridge; wired to real predict once
                                       # we have realized forward-returns feedback
            sigma_total_bps = 0.0
            composite = qe.composite_edge(after_cost_edge_bps, membership_bps, friction)
            predictions.append({
                "symbol": str(symbol),
                "role": "primary",
                "role_enum": role_enum,
                "mu_edge_bps": after_cost_edge_bps,
                "sigma_total_bps": sigma_total_bps,
                "membership_edge_bps": membership_bps,
                "friction_multiplier": friction,
                "composite_edge_bps": composite,
            })
        log.info(
            "step4_provenance",
            n_candidates=len(predictions),
            composite_edge_bps=composite if predictions else 0.0,
        )
        self._persist_predictions(predictions, now, horizon="1d")

        # --- Step 5: gateway evaluation + cost breakdown (§2.24-2.26) ---
        # For each candidate, build MarketState from the fresh snapshot,
        # evaluate a proposed BUY through the C++ gateway, and persist the
        # full cost breakdown to `gateway_actions`. This commit records
        # decisions only — the next commit wires actual order submission.
        open_orders = self.alpaca.get_open_orders()
        open_symbols = {o["symbol"] for o in open_orders}
        cycle_session = session_of(now)
        gw_actions = []
        for pred in predictions:
            symbol = pred["symbol"]
            action = self._evaluate_gateway(
                pred=pred,
                snap=snap,
                account_equity=acct["equity"],
                open_symbols=open_symbols,
                session=cycle_session,
                now=now,
            )
            gw_actions.append(action)
        n_exec = sum(1 for a in gw_actions if a["executed"])
        log.info(
            "step5_gateway",
            n_evaluated=len(gw_actions),
            n_would_execute=n_exec,
            n_would_reject=len(gw_actions) - n_exec,
        )

        # --- Step 5.5: compute target weights over the executed subset (§2.28-2.30) ---
        current_weights = self._current_weights_by_symbol(acct["equity"])
        gw_actions = self._apply_target_weights(gw_actions, current_weights)
        n_kelly_at_cap = sum(1 for a in gw_actions if a.get("_capped_at_cap"))
        log.info(
            "step5_5_weights",
            invested_pct=round(sum(float(a["target_weight"]) for a in gw_actions) * 100, 2),
            n_at_cap=n_kelly_at_cap,
            scheme_hint="kelly_or_equal_fallback",
        )

        self._persist_gateway_actions(gw_actions, now, horizon="1d")

        # --- Step 6-7: order routing + outcome record (§3.3, §3.2 step 7) ---
        # OrderManager writes to `executions` on submission and updates the PDT
        # tracker; we just count outcomes here.
        if not self.cfg.system.trading_enabled:
            log.info(
                "step6_trading_disabled_dry_run",
                n_would_submit=sum(1 for a in gw_actions if a["rebalance_kind"] == "entry"),
                message="Set system.trading_enabled: true in production.yaml to arm order submission.",
            )
            log.info("cycle_end")
            return

        submitted = 0
        rejected = 0
        skipped_existing = 0
        for a in gw_actions:
            # Commit 3 is BUY-only: entry into names we don't currently hold, sized
            # to the target weight. Add / trim / rotate all land in commit 4.
            if a["rebalance_kind"] != "entry":
                if a["rebalance_kind"] in ("add", "trim", "hold"):
                    skipped_existing += 1
                continue
            if float(a["mid_price"]) <= 0 or float(a["target_weight"]) <= 0:
                continue
            notional = float(a["target_weight"]) * float(acct["equity"])
            qty = round(notional / float(a["mid_price"]), 4)
            if qty <= 0:
                continue
            result = self.order_mgr.route(
                symbol=str(a["symbol"]),
                action_type="BUY",
                side="buy",
                qty=qty,
                model_edge_bps=float(a["net_edge_bps"]),
                decision_ref_price=float(a["mid_price"]),
                limit_price=None,
                horizon="1d",
                now=now,
            )
            if result.executed:
                submitted += 1
            else:
                rejected += 1
                log.info("order_rejected", symbol=result.symbol, reason=result.reject_reason)

        log.info(
            "step6_orders",
            submitted=submitted,
            rejected=rejected,
            skipped_existing=skipped_existing,
        )
        log.info("cycle_end")

    # ---- Step 5.5 helpers ----
    def _current_weights_by_symbol(self, equity: float) -> dict[str, float]:
        """symbol -> current portfolio weight from Alpaca positions."""
        if equity <= 0:
            return {}
        try:
            positions = self.alpaca.get_positions()
        except Exception as e:  # noqa: BLE001
            log.warning("current_weights_positions_fetch_failed", error=str(e))
            return {}
        return {p["symbol"]: float(p["market_value"]) / float(equity) for p in positions}

    def _apply_target_weights(
        self,
        gw_actions: list[dict[str, object]],
        current_weights: dict[str, float],
    ) -> list[dict[str, object]]:
        """Compute target weights across executed candidates, annotate each row
        with target_weight / current_weight / delta_weight / rebalance_kind."""
        executed_syms = [str(a["symbol"]) for a in gw_actions if a["executed"]]
        # Composite edge is the pre-cost Kelly numerator (§2.7-2.8). For MVP with
        # sigma_total = 0 on cold start, the CV-based fallback in compute_target_weights
        # collapses to equal weight — same behaviour as fixed 5% but under one code path.
        # sigma_total_bps is currently 0 across all candidates; sigma_floor (100 bps) applies.
        composite_by_sym: dict[str, float] = {}
        for a in gw_actions:
            if a["executed"]:
                composite_by_sym[str(a["symbol"])] = float(a["gross_edge_bps"]) + float(
                    a["total_cost_bps"]
                )  # reconstruct pre-cost edge; gross = composite - cash_wait so this is close
        # Fall back to net_edge_bps if the reconstruction is degenerate.
        candidates = [
            Candidate(
                symbol=sym,
                conservative_edge_bps=composite_by_sym.get(sym, 138.0),
                sigma_total_bps=0.0,  # will populate once Bayesian trains
            )
            for sym in executed_syms
        ]
        result = compute_target_weights(
            candidates,
            max_name_weight=self.cfg.sizing.max_name_weight,
            cash_floor=self.cfg.sizing.cash_floor,
        )
        log.info(
            "target_weights",
            scheme=result.scheme,
            n_symbols=len(result.weights),
            n_at_cap=result.n_at_name_cap,
            cash_weight=round(result.cash_weight, 4),
        )

        for a in gw_actions:
            sym = str(a["symbol"])
            tw = float(result.weights.get(sym, 0.0))
            cw = float(current_weights.get(sym, 0.0))
            dw = tw - cw
            a["target_weight"] = tw
            a["current_weight"] = cw
            a["delta_weight"] = dw
            a["_capped_at_cap"] = tw >= self.cfg.sizing.max_name_weight - 1e-9
            a["rebalance_kind"] = self._classify_rebalance(a, tw, cw)
        return gw_actions

    @staticmethod
    def _classify_rebalance(action: dict[str, object], tw: float, cw: float) -> str:
        # Rejected candidates keep their reject label under a single kind.
        if not action["executed"]:
            return "reject"
        if cw <= 0.0 and tw > 0.0:
            return "entry"
        if tw <= 0.0 and cw > 0.0:
            return "trim"  # full-close case; SELL logic lands in commit 4
        if tw > cw:
            return "add"   # increase existing position; commit 4 emits the delta-BUY
        if tw < cw:
            return "trim"  # decrease existing position; commit 4 emits the SELL
        return "hold"

    # ---- Step 5 helpers ----
    def _evaluate_gateway(
        self,
        *,
        pred: dict[str, object],
        snap: "MarketSnapshot",  # noqa: F821 — forward ref to avoid circular import
        account_equity: float,
        open_symbols: set[str],
        session: str,
        now: datetime,
    ) -> dict[str, object]:
        symbol = str(pred["symbol"])
        quote = snap.quotes.get(symbol)
        bars_list = snap.bars.get(symbol, [])

        # Reject fast on missing observations — recorded, never crashes the loop.
        if quote is None or quote.mid <= 0:
            return self._reject_gw(pred, now, "no_quote")
        if not bars_list:
            return self._reject_gw(pred, now, "no_bars")
        # Wide-spread guard: IEX-only quotes on names with light IEX activity
        # are frequently implausible (100+ bps spreads on mega-caps during
        # regular hours). Skip up-front rather than push garbage into the
        # cost model, where it yields large false-negative rejections.
        if quote.spread_bps > self.cfg.universe.max_spread_bps:
            log.info(
                "wide_spread_skip",
                symbol=symbol,
                spread_bps=round(quote.spread_bps, 1),
                threshold_bps=self.cfg.universe.max_spread_bps,
                bid=quote.bid,
                ask=quote.ask,
                mid=quote.mid,
            )
            return self._reject_gw(pred, now, "wide_spread")

        last_bar = bars_list[-1]
        state = qe.MarketState()
        state.mid_price = quote.mid
        state.spread_bps = quote.spread_bps
        state.volatility_bps = self._annualized_vol_bps(bars_list)
        state.bar_dollar_volume = float(last_bar.close) * float(last_bar.volume or 0)
        state.adv_dollar = self._adv_dollar(bars_list)
        state.quote_age_sessions = quote_age_sessions(quote.ts, now)
        state.gap_days_to_next_session = 0
        state.session_multiplier = self.cfg.costs.session_multiplier.get(session, 1.0)
        state.adverse_selection_share = self.cfg.costs.adverse_selection_share.get(session, 0.35)

        slippage = qe.SlippageStats()
        slippage.recent_fill_slippage_bps = 0.0  # cold-start

        # MVP sizing: fixed 5% of equity per candidate. Real Kelly sizing in a later pass.
        # Note: reading the cap from the Pydantic config rather than the
        # quant_engine.SizingConfig struct because the pybind11 binding for
        # SizingConfig only exposes `__init__` (no def_readwrite on fields).
        # Adding field bindings requires a C++ rebuild; the two defaults match (0.10).
        target_weight = min(0.05, self.cfg.sizing.max_name_weight)
        notional = target_weight * float(account_equity)

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
            state,
            slippage,
            self.cost_cfg,
            self.gw_cfg,
            qe.ActionType.BUY,
        )
        cost = qe.compute_cost(order, state, slippage, self.cost_cfg, float(pred["composite_edge_bps"]))

        executed = bool(evaluation.executes())
        reject_reason = "" if executed else "net_edge_below_hurdle"

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
            "reject_reason": reject_reason,
            "notional": notional,
            "mid_price": quote.mid,
        }

    @staticmethod
    def _reject_gw(pred: dict[str, object], _now: datetime, reason: str) -> dict[str, object]:
        return {
            "symbol": str(pred["symbol"]),
            "action_type": "HOLD",
            "role": str(pred["role"]),
            "gross_edge_bps": 0.0,
            "entry_cost_bps": 0.0,
            "exit_cost_reserved": 0.0,
            "queue_delay_bps": 0.0,
            "cancel_replace_bps": 0.0,
            "action_memory_bps": 0.0,
            "cash_waiting_value": 0.0,
            "operational_bps": 0.0,
            "total_cost_bps": 0.0,
            "net_edge_bps": 0.0,
            "hurdle_bps": 0.0,
            "friction_multiplier": float(pred["friction_multiplier"]),
            "executed": False,
            "reject_reason": reason,
            "notional": 0.0,
            "mid_price": 0.0,
        }

    @staticmethod
    def _annualized_vol_bps(bars_list: list) -> float:
        """Roll-uncorrected annualized vol from close-to-close log returns.
        Placeholder — Roll correction using quote spreads lands in a later pass."""
        import math
        closes = [float(b.close) for b in bars_list if b.close and b.close > 0]
        if len(closes) < 2:
            return 0.0
        rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        n = len(rets)
        mean = sum(rets) / n
        var = sum((r - mean) ** 2 for r in rets) / max(1, n - 1)
        return 10_000.0 * math.sqrt(max(0.0, var) * 252.0)

    @staticmethod
    def _adv_dollar(bars_list: list, window: int = 20) -> float:
        recent = bars_list[-window:]
        if not recent:
            return 0.0
        vals = [float(b.close) * float(b.volume or 0) for b in recent if b.close and b.volume]
        return sum(vals) / len(vals) if vals else 0.0

    def _persist_gateway_actions(
        self, actions: list[dict[str, object]], ts: datetime, horizon: str
    ) -> None:
        if not actions:
            return
        with self.db.sender() as s:
            for a in actions:
                s.row(
                    "gateway_actions",
                    symbols={
                        "symbol": str(a["symbol"]),
                        "action_type": str(a["action_type"]),
                        "role": str(a["role"]),
                        "horizon": horizon,
                        "reject_reason": str(a["reject_reason"]) or "none",
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

    def _persist_predictions(
        self, predictions: list[dict[str, object]], ts: datetime, horizon: str
    ) -> None:
        if not predictions:
            return
        with self.db.sender() as s:
            for p in predictions:
                s.row(
                    "predictions",
                    symbols={
                        "symbol": str(p["symbol"]),
                        "horizon": horizon,
                        "role": str(p["role"]),
                    },
                    columns={
                        "mu_edge_bps": float(p["mu_edge_bps"]),
                        "sigma_edge_bps": float(p["sigma_total_bps"]),
                        "sigma_total_bps": float(p["sigma_total_bps"]),
                        "p_positive": 0.5,  # neutral cold-start
                        "conservative_edge": float(p["mu_edge_bps"]),  # μ − zq·σ with σ=0
                        "membership_edge": float(p["membership_edge_bps"]),
                        "composite_edge": float(p["composite_edge_bps"]),
                    },
                    at=ts,
                )

    def _persist_features(self, features: pd.DataFrame, ts: datetime) -> None:
        """Write one feature row per symbol to QuestDB via ILP. Skips NaN cells."""
        if features.empty:
            log.warning("features_empty_skip_persist")
            return
        n_written = 0
        with self.db.sender() as s:
            for symbol, row in features.iterrows():
                columns: dict[str, float | int | bool] = {}
                for col, val in row.items():
                    if pd.notna(val):
                        # cast everything to float; ILP handles the wire type
                        columns[str(col)] = float(val)
                # Defaults for the two derived meta-columns not produced by compute_all.
                # (§1.3 freshness_weight; §1.7 data_quality — both use 1.0 as MVP baseline.)
                columns.setdefault("freshness_weight", 1.0)
                columns.setdefault("data_quality", 1.0)
                if not columns:
                    continue
                s.row(
                    "features",
                    symbols={"symbol": str(symbol)},
                    columns=columns,
                    at=ts,
                )
                n_written += 1
        log.info("step2_persisted", n_rows=n_written)

    def _resolve_universe(self) -> list[str]:
        """Return the symbol list for this decision cycle.

        Prefer an explicit config seed list; fall back to the full tape-filtered
        universe builder only if the seed is empty (that path is heavy and
        typically only run in offline research).
        """
        if self.cfg.universe.symbols:
            return list(self.cfg.universe.symbols)
        # Full-universe fallback intentionally deferred — the tape filter
        # needs last_close + ADV + fundamentals for every candidate, which
        # is expensive. Callers who want it can wire it here.
        log.warning("universe_seed_empty_and_full_filter_not_wired")
        return []

    async def _resolution_loop(self) -> None:
        """§3.2 slow-feedback loop.

        Persists account + open positions snapshot every hour so Grafana
        panels stay current. Once step-7 wiring lands, this will also resolve
        stale executions into realized_return_bps and update the shadow
        monitor (§2.41).
        """
        log.info("resolution_start")
        try:
            self._persist_account_snapshot()
        except Exception as e:  # noqa: BLE001 — never let the loop die
            log.error("resolution_snapshot_failed", error=str(e), error_type=type(e).__name__)
        log.info("resolution_end")

    def _persist_account_snapshot(self) -> None:
        """Snapshot equity/cash/PDT + open positions to QuestDB (§3.1)."""
        now = datetime.now(tz=ET)
        acct = self.alpaca.get_account()
        positions = self.alpaca.get_positions()
        dt_count = self.pdt.count_in_window(now)
        unrealized_total = sum(p["unrealized_pl"] for p in positions)

        log.info(
            "snapshot",
            equity=acct["equity"],
            cash=acct["cash"],
            buying_power=acct["buying_power"],
            day_trade_count=dt_count,
            position_count=len(positions),
            unrealized_pl=unrealized_total,
            paper=self.cfg.alpaca.paper,
        )

        pdt_blocked = dt_count >= self.cfg.model.max_day_trades_rolling_5d
        with self.db.sender() as s:
            s.row(
                "account_state",
                columns={
                    "equity": acct["equity"],
                    "cash": acct["cash"],
                    "buying_power": acct["buying_power"],
                    "day_trade_count": dt_count,
                    "position_count": len(positions),
                    "unrealized_pl": unrealized_total,
                    "realized_pl": 0.0,  # Alpaca account API does not expose realized PL directly
                    "pdt_blocked": pdt_blocked,
                },
                at=now,
            )
            for p in positions:
                s.row(
                    "positions",
                    symbols={"symbol": p["symbol"], "side": p["side"]},
                    columns={
                        "qty": p["qty"],
                        "avg_entry_price": p["avg_entry_price"],
                        "market_value": p["market_value"],
                        "unrealized_pl": p["unrealized_pl"],
                    },
                    at=now,
                )


def _install_signal_handlers(app: JuniAuto, loop: asyncio.AbstractEventLoop) -> None:
    def _sig(signame: str) -> None:
        log.info("signal_received", signal=signame)
        app.request_stop()

    for s in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(getattr(signal, s), _sig, s)
        except NotImplementedError:
            # Windows: signal.signal fallback
            signal.signal(getattr(signal, s), lambda *_a: _sig(s))


async def _amain(config_path: str) -> None:
    cfg = load_config(config_path)
    configure_logging(level=cfg.logging.level, json_file=cfg.logging.file if cfg.logging.file else None)
    app = JuniAuto(cfg)
    loop = asyncio.get_running_loop()
    _install_signal_handlers(app, loop)
    await app.start()


def main() -> None:
    p = argparse.ArgumentParser(prog="juniauto")
    p.add_argument("--config", default="/app/config/production.yaml")
    args = p.parse_args()
    asyncio.run(_amain(args.config))


if __name__ == "__main__":
    main()
