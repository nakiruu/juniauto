"""CLI entry point for the multi-horizon experiment.

Usage:
    docker exec juniauto-engine python -m experiments.multi_horizon.run \\
        --start 2022-01-01 --end 2026-07-24 \\
        --run-id exp_mh_v1 --fill next_open --benchmarks spy,ew

Runs the multi-horizon backtest end-to-end:
  1. MultiHorizonBacktestEngine construction (fits 3 ridges from bars+features)
  2. Backtest cycle loop over the date range
  3. Benchmark equity-curve writes (SPY, EW; fixed5 skipped by default)
  4. Metrics computation

Writes to `backtest_*` tables with the given `run_id` — Grafana
overlays this run against `v2_trained` (or any other run_id) without
dashboard changes.
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path

from experiments.multi_horizon.engine import MultiHorizonBacktestEngine
from juniauto.backtest.benchmarks import (
    benchmark_equal_weight,
    benchmark_spy,
)
from juniauto.backtest.metrics import compute_and_persist_metrics
from juniauto.config import load_config
from juniauto.db import QuestDBClient
from juniauto.utils import configure_logging, get_logger

log = get_logger(__name__)


DEFAULT_HORIZONS = (1, 5, 20)
DEFAULT_BLEND_WEIGHTS = (0.15, 0.50, 0.35)   # 1d, 5d, 20d


def _parse_date(s: str):
    from datetime import date
    return datetime.strptime(s, "%Y-%m-%d").date()


def _default_config_path() -> str:
    for candidate in ("/app/config/production.yaml", "./config/production.yaml"):
        if os.path.exists(candidate):
            return candidate
    return "./config/production.yaml"


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(
        prog="python -m experiments.multi_horizon.run",
        description="Multi-horizon Bayesian ridge backtest.",
    )
    p.add_argument("--start", type=_parse_date, required=True)
    p.add_argument("--end", type=_parse_date, required=True)
    p.add_argument("--run-id", type=str, required=True,
                   help="Unique run label. Use `exp_mh_v1` for the first run.")
    p.add_argument("--fill", type=str, default="next_open",
                   choices=["next_open", "close", "vwap", "delayed_mid"])
    p.add_argument("--walkforward", type=int, default=21,
                   help="No-op during backtest (kept for CLI compat).")
    p.add_argument("--initial-cash", type=float, default=10_000.0)
    p.add_argument("--config", type=str, default=_default_config_path())
    p.add_argument("--benchmarks", type=str, default="spy,ew",
                   help="Comma-separated: spy, ew, fixed5")
    p.add_argument("--universe", type=str, default="",
                   help="Comma-separated symbols (default: config.universe.symbols)")
    p.add_argument("--horizons", type=str, default="1,5,20",
                   help="Comma-separated forward-return horizons in trading days.")
    p.add_argument("--blend-weights", type=str, default="0.15,0.50,0.35",
                   help=f"Comma-separated weights, same order as --horizons. Default {DEFAULT_BLEND_WEIGHTS}.")
    p.add_argument("--notes", type=str, default="")
    p.add_argument("--skip-metrics", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    configure_logging(level=cfg.logging.level, json_file=None)

    horizons = tuple(int(x) for x in args.horizons.split(","))
    blend_weights = tuple(float(x) for x in args.blend_weights.split(","))
    if abs(sum(blend_weights) - 1.0) > 1e-6:
        log.error("blend_weights_do_not_sum_to_one", weights=list(blend_weights))
        return 2

    log.info(
        "mh_cli_start",
        run_id=args.run_id, start=str(args.start), end=str(args.end),
        fill=args.fill, horizons=list(horizons), blend_weights=list(blend_weights),
        initial_cash=args.initial_cash, config=args.config,
    )

    db_client = QuestDBClient(cfg.database)
    for name in ("schema.sql", "schema_backtest.sql"):
        schema_path = Path(__file__).parent.parent.parent / "orchestrator" / "juniauto" / "db" / name
        try:
            db_client.apply_schema(schema_path)
        except Exception as e:  # noqa: BLE001
            log.error("mh_cli_schema_apply_failed", path=str(schema_path), error=str(e))
            return 2

    universe = (
        [s.strip() for s in args.universe.split(",") if s.strip()]
        if args.universe else None
    )

    cli_repr = " ".join(shlex.quote(a) for a in argv)
    engine = MultiHorizonBacktestEngine(
        cfg,
        horizons=horizons,
        blend_weights=blend_weights,
        run_id=args.run_id,
        start_date=args.start,
        end_date=args.end,
        fill_model=args.fill,
        walkforward_days=args.walkforward,
        initial_cash=args.initial_cash,
        universe=universe,
        cli_args=cli_repr,
        notes=args.notes or f"multi-horizon h={horizons} w={blend_weights}",
    )
    engine.run()

    # Benchmarks (reuse baseline modules; fixed5 skipped by default here
    # since it's a comparison against the main-run's actions and would be
    # redundant if we're primarily comparing v2_trained vs exp_mh_v1).
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
            else:
                log.warning("mh_unknown_benchmark", name=b)
        except Exception as e:  # noqa: BLE001
            log.error("mh_benchmark_failed", name=b, error=str(e))

    if not args.skip_metrics:
        try:
            m = compute_and_persist_metrics(db_client, run_id=args.run_id)
            _print_summary(args.run_id, m)
        except Exception as e:  # noqa: BLE001
            log.error("mh_metrics_failed", error=str(e))
            return 3
    return 0


def _print_summary(run_id: str, all_metrics: dict) -> None:
    if not all_metrics:
        print(f"\n[{run_id}] No metrics computed.")
        return
    print(f"\n{'='*80}\nMulti-horizon backtest summary: {run_id}\n{'='*80}")
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
