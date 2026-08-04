"""Manually trigger the trailing-stop-management step.

Fires ``TrailingStopManager.manage_cycle`` without waiting for the next
scheduled scan (09:40 phantom, 12:30 phantom, or 15:55 live). Uses the
same fetch+compute path as ``_daily_decision_cycle`` steps 1-4 (bars,
features, predictions) but skips the gateway / order routing / phantom-
write steps entirely — only the stop-management step runs.

Usage from a live container:

    # Fresh DAY-stop submits for every held position (skips entry-day names).
    # Use this when you want to prime the system NOW instead of waiting for
    # the next 15:55 live cycle.
    docker exec juniauto-engine python -m juniauto.tools.trigger_stops --cycle 1555

    # Hysteresis-triggered REPLACE only — REPLACEs names whose recomputed
    # level is materially higher than the standing broker-side stop.
    docker exec juniauto-engine python -m juniauto.tools.trigger_stops --cycle 0945

Safety notes:
    * The main juniauto.main scheduler is still running in parallel. This
      script coexists with it — Alpaca handles concurrent order submits and
      QuestDB ILP writes idempotently. The only real overlap risk is if you
      run this within 60 seconds of a scheduled cycle firing, which would
      produce two rapid submit passes. Harmless in practice.
    * Respects stops.canary_symbols exactly the same way as scheduled cycles.
      Empty list = shadow mode; ["*"] = enable all; else allowlist.
    * Respects stops.entry_day_exempt — will not submit stops for positions
      opened today.
"""
from __future__ import annotations

import argparse
from datetime import datetime

from juniauto.bayesian import BayesianModel
from juniauto.config import load_config
from juniauto.data import AlpacaFeed, DataAggregator, YahooFeed
from juniauto.db import QuestDBClient
from juniauto.execution import PDTTracker, TrailingStopManager
from juniauto.signals import compute_all
from juniauto.utils import configure_logging, get_logger
from juniauto.utils.time_utils import ET

log = get_logger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(
        prog="trigger_stops",
        description="Fire TrailingStopManager.manage_cycle now.",
    )
    p.add_argument(
        "--config", default="/app/config/production.yaml",
        help="Path to production.yaml (default: /app/config/production.yaml)",
    )
    p.add_argument(
        "--cycle", default="1555", choices=["1555", "0945", "1230"],
        help="Cycle type semantics: 1555=fresh DAY submit, "
             "0945/1230=hysteresis-triggered REPLACE. Default: 1555.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    configure_logging(level=cfg.logging.level, json_file=None)
    log.info(
        "trigger_stops_start",
        cycle=args.cycle,
        canary=cfg.stops.canary_symbols,
        enabled=cfg.stops.enabled,
    )
    if not cfg.stops.enabled:
        log.error("trigger_stops_disabled",
                  message="cfg.stops.enabled=false — nothing to do")
        return

    db = QuestDBClient(cfg.database)
    alpaca = AlpacaFeed(cfg.alpaca)
    yahoo = YahooFeed(
        ttl_days=cfg.yahoo.fundamentals_ttl_days,
        enabled=cfg.yahoo.enabled,
        max_workers=cfg.yahoo.max_workers,
        per_symbol_timeout_seconds=cfg.yahoo.per_symbol_timeout_seconds,
    )
    pdt = PDTTracker()
    aggregator = DataAggregator(cfg, alpaca, yahoo, db)
    stop_mgr = TrailingStopManager(cfg.stops, db, alpaca, pdt)

    # 1. Held positions determine the symbol set — no universe fetch needed.
    positions = alpaca.get_positions()
    held_symbols = [p["symbol"] for p in positions]
    if not held_symbols:
        log.info("trigger_stops_no_positions_nothing_to_do")
        return
    log.info("trigger_stops_positions_loaded", n=len(held_symbols))

    # 2. Snapshot bars/quotes/fundamentals for held names only. Same code
    #    path as _daily_decision_cycle step 1.
    now = datetime.now(tz=ET)
    try:
        snap = aggregator.snapshot(held_symbols, now)
    except Exception as e:  # noqa: BLE001
        log.error("trigger_stops_snapshot_failed",
                  error=str(e), error_type=type(e).__name__)
        return
    log.info(
        "trigger_stops_snapshot",
        bars=len(snap.bars),
        quotes=len(snap.quotes),
        fundamentals=len(snap.fundamentals),
    )

    # 3. Features — the manager reads only realized_vol_bps, but compute_all
    #    is cheap and mirrors the live cycle exactly. NaN handling is
    #    identical to main.py:_manage_stops.
    try:
        features = compute_all(
            bars=snap.bars_df(),
            fundamentals=snap.fundamentals,
            quotes=snap.quotes,
            as_of_date=now.date(),
            halflife_event_days=cfg.freshness_halflife_days["event"],
        )
    except Exception as e:  # noqa: BLE001
        log.error("trigger_stops_features_failed", error=str(e))
        return
    features_by_symbol: dict[str, dict[str, float]] = {}
    if not features.empty:
        import pandas as pd  # noqa: WPS433 — deferred to avoid import cost when nothing to compute
        for sym, row in features.iterrows():
            rv = row.get("realized_vol_bps", 0.0)
            features_by_symbol[str(sym)] = {
                "realized_vol_bps": float(rv) if pd.notna(rv) else 0.0,
            }

    # 4. Predictions — optional. Untrained Bayesian returns (0, 0), which
    #    means conservative_edge = 0 -> uses k_loose_vol. That's the correct
    #    behavior for a cold-start model (no adverse-posterior tightening).
    predictions_by_symbol: dict[str, dict[str, float]] = {}
    try:
        bayes = BayesianModel(db, cfg)
        bayes_trained = bayes.is_trained()
    except Exception as e:  # noqa: BLE001
        log.warning("trigger_stops_bayes_init_failed", error=str(e))
        bayes_trained = False
        bayes = None
    if bayes is not None and bayes_trained and not features.empty:
        for sym in features.index:
            try:
                mu, sigma = bayes.predict(features.loc[sym])
            except Exception:  # noqa: BLE001
                mu, sigma = 0.0, 0.0
            predictions_by_symbol[str(sym)] = {
                "mu_edge_bps": float(mu),
                "sigma_total_bps": float(sigma),
            }
    else:
        for sym in held_symbols:
            predictions_by_symbol[sym] = {
                "mu_edge_bps": 0.0, "sigma_total_bps": 0.0,
            }
        log.info(
            "trigger_stops_bayes_cold_start",
            trained=bayes_trained,
            message="predictions default to 0 — level uses chandelier + loose posterior only",
        )

    # 5. Fire the manager. Rate-limit pacing (submit_batch_size /
    #    submit_pause_ms) inside manage_cycle handles Alpaca's 200 req/min.
    summary = stop_mgr.manage_cycle(
        cycle_type=args.cycle,
        positions=positions,
        bars_by_symbol=snap.bars,
        features_by_symbol=features_by_symbol,
        predictions_by_symbol=predictions_by_symbol,
        now=now,
        zq=float(cfg.bayesian.zq),
    )
    log.info("trigger_stops_done", cycle=args.cycle, **summary)


if __name__ == "__main__":
    main()
