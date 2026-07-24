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
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from prometheus_client import start_http_server

from juniauto.config import JuniAutoConfig, load_config
from juniauto.data import AlpacaFeed, DataAggregator, UniverseBuilder, YahooFeed
from juniauto.db import QuestDBClient
from juniauto.execution import OrderManager, PDTTracker
from juniauto.replay import ReplayHarness
from juniauto.signals import compute_all
from juniauto.utils import configure_logging, get_logger
from juniauto.utils.time_utils import ET

log = get_logger(__name__)


class JuniAuto:
    def __init__(self, cfg: JuniAutoConfig) -> None:
        self.cfg = cfg
        self.db = QuestDBClient(cfg.database)
        self.alpaca = AlpacaFeed(cfg.alpaca)
        self.yahoo = YahooFeed(ttl_days=cfg.yahoo.fundamentals_ttl_days)
        self.pdt = PDTTracker()
        self.universe = UniverseBuilder(self.alpaca._trading, cfg.universe)  # type: ignore[attr-defined]
        self.aggregator = DataAggregator(cfg, self.alpaca, self.yahoo, self.db)
        self.order_mgr = OrderManager(self.alpaca, self.db, self.pdt)
        self.replay = ReplayHarness(cfg, self.db)
        self._sched = AsyncIOScheduler(timezone=ET)
        self._stop = asyncio.Event()

    # ---- Lifecycle ----
    async def start(self) -> None:
        log.info("startup", version=self.cfg.system.version, env=self.cfg.system.environment, paper=self.cfg.alpaca.paper)

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

        # Steps 3-7 still stubbed — next wiring commits.
        for step in [
            "3_source_selector",
            "4_target_gateway_and_provenance",
            "5_execution_gate",
            "6_order_routing",
            "7_outcome_record",
        ]:
            log.info("cycle_step_stub", step=step)

        log.info("cycle_end")

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
