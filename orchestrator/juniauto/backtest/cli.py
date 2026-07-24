"""CLI for the JuniAuto backtest.

Usage:
    python -m juniauto.backtest \\
        --start 2024-01-01 --end 2026-07-01 \\
        --run-id kelly_v3 \\
        --fill next_open \\
        --walkforward 21 \\
        --benchmarks spy,ew,fixed5 \\
        --initial-cash 10000 \\
        --config /app/config/production.yaml

Flow:
    1. Load config (defaults to /app/config/production.yaml in-container,
       or ./config/production.yaml locally).
    2. Apply schema (idempotent) so backtest_* tables exist.
    3. Instantiate BacktestEngine and run.
    4. Compute benchmark equity curves.
    5. Compute + persist metrics for every curve.
    6. Print a compact summary table to stdout.
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from datetime import date, datetime
from pathlib import Path

from juniauto.backtest.benchmarks import (
    benchmark_equal_weight,
    benchmark_fixed5,
    benchmark_spy,
)
from juniauto.backtest.engine import BacktestEngine
from juniauto.backtest.metrics import compute_and_persist_metrics
from juniauto.config import load_config
from juniauto.utils import configure_logging, get_logger

log = get_logger(__name__)


def _default_config_path() -> str:
    for candidate in ("/app/config/production.yaml", "./config/production.yaml"):
        if os.path.exists(candidate):
            return candidate
    return "./config/production.yaml"


def _parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"date must be YYYY-MM-DD, got {s}") from e


def _parse_bench_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m juniauto.backtest",
        description="Event-driven historical backtest for JuniAuto.",
    )
    p.add_argument("--start", type=_parse_date, required=True,
                   help="Backtest window start date (YYYY-MM-DD, inclusive).")
    p.add_argument("--end", type=_parse_date, required=True,
                   help="Backtest window end date (YYYY-MM-DD, inclusive).")
    p.add_argument("--run-id", type=str, required=True,
                   help="Unique label for this run — used as row key across all backtest_* tables.")
    p.add_argument("--fill", type=str, default="next_open",
                   choices=["next_open", "close", "vwap", "delayed_mid"],
                   help="Fill model. next_open is the coordinator-recommended default.")
    p.add_argument("--walkforward", type=int, default=21,
                   help="Bayesian retrain cadence in trading days (default 21 per coordinator).")
    p.add_argument("--initial-cash", type=float, default=10_000.0,
                   help="Starting cash for the main + unconstrained + benchmark curves.")
    p.add_argument("--config", type=str, default=_default_config_path(),
                   help="Path to production.yaml.")
    p.add_argument(
        "--benchmarks", type=_parse_bench_list, default=["spy", "ew", "fixed5"],
        help="Comma-separated benchmarks to compute: spy, ew, fixed5.",
    )
    p.add_argument(
        "--universe", type=str, default="",
        help="Comma-separated symbol list. Empty => use config.universe.symbols.",
    )
    p.add_argument("--notes", type=str, default="", help="Free-form notes stored with metadata.")
    p.add_argument("--skip-benchmarks", action="store_true",
                   help="Skip benchmark equity-curve computation.")
    p.add_argument("--skip-metrics", action="store_true",
                   help="Skip post-run metrics computation.")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(argv)

    cfg = load_config(args.config)
    configure_logging(cfg.logging)
    log.info(
        "backtest_cli_start",
        run_id=args.run_id, start=str(args.start), end=str(args.end),
        fill=args.fill, walkforward=args.walkforward,
        initial_cash=args.initial_cash, config=args.config,
        benchmarks=args.benchmarks,
    )

    # Apply backtest schema idempotently (in case the container hasn't been
    # restarted since a fresh clone).
    from juniauto.db import QuestDBClient
    db_client = QuestDBClient(cfg.database)
    for name in ("schema.sql", "schema_backtest.sql"):
        schema_path = Path(__file__).parent.parent / "db" / name
        try:
            db_client.apply_schema(schema_path)
        except Exception as e:  # noqa: BLE001
            log.error("cli_schema_apply_failed", path=str(schema_path), error=str(e))
            return 2

    universe = (
        [s.strip() for s in args.universe.split(",") if s.strip()]
        if args.universe else None
    )

    cli_repr = " ".join(shlex.quote(a) for a in argv)
    engine = BacktestEngine(
        cfg,
        run_id=args.run_id,
        start_date=args.start,
        end_date=args.end,
        fill_model=args.fill,
        walkforward_days=args.walkforward,
        initial_cash=args.initial_cash,
        universe=universe,
        cli_args=cli_repr,
        notes=args.notes,
    )
    engine.run()

    # Benchmark equity curves — computed AFTER engine so they use the same
    # trading-day set the engine actually iterated.
    if not args.skip_benchmarks:
        run_universe = list(engine.universe)
        for b in args.benchmarks:
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
                elif b == "fixed5":
                    benchmark_fixed5(db_client, run_id=args.run_id,
                                     start_date=args.start, end_date=args.end,
                                     initial_cash=args.initial_cash)
                else:
                    log.warning("unknown_benchmark", name=b)
            except Exception as e:  # noqa: BLE001
                log.error("benchmark_failed", name=b, error=str(e))

    if not args.skip_metrics:
        try:
            all_metrics = compute_and_persist_metrics(db_client, run_id=args.run_id)
            _print_summary(args.run_id, all_metrics)
        except Exception as e:  # noqa: BLE001
            log.error("metrics_computation_failed", error=str(e))
            return 3
    return 0


def _print_summary(run_id: str, all_metrics: dict) -> None:
    if not all_metrics:
        print(f"\n[{run_id}] No metrics computed (no equity curves found).")
        return
    print(f"\n{'='*80}\nBacktest summary: {run_id}\n{'='*80}")
    # Pick a canonical short list of metrics to show per curve.
    highlights = [
        "total_return_pct", "cagr_pct", "annualized_vol_pct",
        "sharpe_annualized", "sortino_annualized", "calmar",
        "max_drawdown_pct", "max_drawdown_days",
        "capm_alpha_annualized_bps", "capm_beta", "capm_r_squared", "capm_alpha_tstat",
        "n_fills", "hit_rate", "n_days_pdt_blocked",
    ]
    headers = ["metric"] + list(all_metrics.keys())
    col_w = max(30, max((len(h) for h in headers), default=15))
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
