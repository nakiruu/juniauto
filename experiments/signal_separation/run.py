"""CLI for the signal-separation experiment.

Usage:
    docker exec juniauto-engine python -m experiments.signal_separation.run \\
        --start 2022-01-01 --end 2026-07-24 \\
        --run-id exp_sigsep_v1 --fill next_open --benchmarks spy,ew
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from datetime import datetime

from experiments.signal_separation.engine import SignalSeparationBacktestEngine
from juniauto.backtest.benchmarks import benchmark_equal_weight, benchmark_spy
from juniauto.backtest.metrics import compute_and_persist_metrics
from juniauto.config import load_config
from juniauto.db import QuestDBClient
from juniauto.utils import configure_logging, get_logger

log = get_logger(__name__)


def _parse_date(s: str):
    return datetime.strptime(s, "%Y-%m-%d").date()


def _default_config_path() -> str:
    for candidate in ("/app/config/production.yaml", "./config/production.yaml"):
        if os.path.exists(candidate):
            return candidate
    return "./config/production.yaml"


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(
        prog="python -m experiments.signal_separation.run",
        description="Audit-driven fix: μ Kelly numerator + dead-feature drop.",
    )
    p.add_argument("--start", type=_parse_date, required=True)
    p.add_argument("--end", type=_parse_date, required=True)
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--fill", type=str, default="next_open",
                   choices=["next_open", "close", "vwap", "delayed_mid"])
    p.add_argument("--walkforward", type=int, default=21)
    p.add_argument("--initial-cash", type=float, default=10_000.0)
    p.add_argument("--config", type=str, default=_default_config_path())
    p.add_argument("--benchmarks", type=str, default="spy,ew")
    p.add_argument("--universe", type=str, default="")
    p.add_argument("--notes", type=str, default="")
    p.add_argument("--skip-metrics", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    configure_logging(level=cfg.logging.level, json_file=None)
    log.info(
        "sigsep_cli_start",
        run_id=args.run_id, start=str(args.start), end=str(args.end),
        fill=args.fill,
    )

    db_client = QuestDBClient(cfg.database)
    universe = (
        [s.strip() for s in args.universe.split(",") if s.strip()]
        if args.universe else None
    )

    cli_repr = " ".join(shlex.quote(a) for a in argv)
    engine = SignalSeparationBacktestEngine(
        cfg,
        run_id=args.run_id,
        start_date=args.start,
        end_date=args.end,
        fill_model=args.fill,
        walkforward_days=args.walkforward,
        initial_cash=args.initial_cash,
        universe=universe,
        cli_args=cli_repr,
        notes=args.notes or "signal-separation: mu-kelly + reduced-feature bayes",
    )
    engine.run()

    run_universe = list(engine.universe)
    for b in [x.strip() for x in args.benchmarks.split(",") if x.strip()]:
        try:
            if b == "spy":
                benchmark_spy(db_client, run_id=args.run_id,
                              start_date=args.start, end_date=args.end,
                              initial_cash=args.initial_cash)
            elif b == "ew":
                benchmark_equal_weight(db_client, run_id=args.run_id,
                                        start_date=args.start, end_date=args.end,
                                        initial_cash=args.initial_cash,
                                        universe=run_universe)
        except Exception as e:  # noqa: BLE001
            log.error("sigsep_benchmark_failed", name=b, error=str(e))

    if not args.skip_metrics:
        try:
            m = compute_and_persist_metrics(db_client, run_id=args.run_id)
            _print_summary(args.run_id, m)
        except Exception as e:  # noqa: BLE001
            log.error("sigsep_metrics_failed", error=str(e))
            return 3
    return 0


def _print_summary(run_id: str, all_metrics: dict) -> None:
    if not all_metrics:
        print(f"\n[{run_id}] No metrics computed.")
        return
    print(f"\n{'='*80}\nSignal-separation backtest summary: {run_id}\n{'='*80}")
    highlights = [
        "total_return_pct", "cagr_pct", "annualized_vol_pct",
        "sharpe_annualized", "sortino_annualized", "calmar",
        "max_drawdown_pct", "capm_alpha_annualized_bps", "capm_beta",
        "n_fills", "hit_rate",
    ]
    print(f"  {'metric':<32}" + "".join(f"{c:>18}" for c in all_metrics.keys()))
    print("  " + "-" * (32 + 18 * len(all_metrics)))
    for name in highlights:
        row = f"  {name:<32}"
        for ct in all_metrics.keys():
            val = None
            for r in all_metrics[ct]:
                if r.name == name:
                    val = r.value
                    break
            row += f"{val:>18.4f}" if val is not None else f"{'—':>18}"
        print(row)
    print()


if __name__ == "__main__":
    raise SystemExit(main())
