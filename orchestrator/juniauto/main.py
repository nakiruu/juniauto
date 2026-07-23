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

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from prometheus_client import start_http_server

from juniauto.config import JuniAutoConfig, load_config
from juniauto.data import AlpacaFeed, DataAggregator, UniverseBuilder, YahooFeed
from juniauto.db import QuestDBClient
from juniauto.execution import OrderManager, PDTTracker
from juniauto.replay import ReplayHarness
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

        1) refresh observations
        2) source-package selector
        3) target gateway → candidates
        4) provenance membership → prior action edge
        5) execution gate (posterior + costs → action)
        6) order routing
        7) outcome recording
        """
        log.info("cycle_start", ts=datetime.now(tz=ET).isoformat())

        # Guardrail: don't trade if PDT will block us for the whole cycle.
        acct = self.alpaca.get_account()
        dt_count = self.pdt.count_in_window()
        log.info("account", equity=acct["equity"], cash=acct["cash"], day_trade_count=dt_count, paper=self.cfg.alpaca.paper)

        # TODO: full pipeline wires in once engine bindings + signal families are online.
        # For now, log the intended step sequence so the scheduler wiring is visible.
        for step in [
            "1_refresh_observations",
            "2_source_selector",
            "3_target_gateway",
            "4_provenance_membership",
            "5_execution_gate",
            "6_order_routing",
            "7_outcome_record",
        ]:
            log.info("cycle_step_stub", step=step)

        log.info("cycle_end")

    async def _resolution_loop(self) -> None:
        """Resolve stale executions → realized_return_bps, update shadow (§2.41)."""
        log.info("resolution_start")
        # TODO
        log.info("resolution_end")


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
