"""Bayesian bootstrap — synthesize training data from historical daily bars.

Problem it solves: the Bayesian ridge only starts predicting once
`is_trained()` returns True, which requires >= 30 resolved (features, realized-
return) pairs. Waiting for those to accumulate organically means ~30 trading
days of live-but-uninformative decisions.

Solution: for each historical trading day T in a rolling backfill window,
compute the features vector using bars up to T (no leakage), then use the T+1
close to compute a realized 1-day return. Write both to their respective
QuestDB tables (`features`, `executions`) using action_type='BACKFILL' so
they're distinguishable from real trades but still visible to the ridge
training path in `training.build_training_matrix()`.

Approximations documented in the code:
    - Fundamentals: uses the CURRENT yfinance snapshot for every historical
      date. The §1.3 halflife is ~20 trading days, so within a 60-day backfill
      window this is close enough. Longer backfills would need point-in-time
      fundamentals (yfinance doesn't provide them cleanly).
    - Quotes: no historical quotes; liquidity features fall back to bar-level
      proxies only. Spread_bps will be zero/None, which the signal families
      already tolerate.

Usage (from the host):
    docker exec juniauto-engine python -m juniauto.bayesian.backfill --days 60
    # then retrain picks up the new rows on the next hourly loop, or force it:
    docker exec juniauto-engine python -c "\
from juniauto.config import load_config; from juniauto.db import QuestDBClient; \
from juniauto.bayesian import BayesianModel; \
m = BayesianModel(QuestDBClient(load_config('/app/config/production.yaml').database), \
    load_config('/app/config/production.yaml')); \
print('trained on', m.retrain_from_db(), 'rows')"

The backfill script is idempotent per (symbol, date) — re-running with the
same window will overwrite the same executions rows (QuestDB WAL append +
retrain reads latest per order_id).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from juniauto.bayesian.training import _CANONICAL_FEATURE_COLS, BayesianModel
from juniauto.config import JuniAutoConfig, load_config
from juniauto.data import AlpacaFeed, YahooFeed
from juniauto.data.alpaca_feed import Bar
from juniauto.db import QuestDBClient
from juniauto.signals import compute_all
from juniauto.utils import configure_logging, get_logger

log = get_logger(__name__)
ET = ZoneInfo("America/New_York")


def _bars_slice_df(bars: dict[str, list[Bar]], cutoff: date) -> pd.DataFrame:
    """DataFrame of bars where ts.date() <= cutoff, in the shape compute_all expects."""
    rows: list[dict[str, object]] = []
    for sym, series in bars.items():
        for b in series:
            if b.ts.date() > cutoff:
                continue
            rows.append({
                "symbol": sym,
                "ts": b.ts,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "vwap": b.vwap,
                "trade_count": b.trade_count,
            })
    if not rows:
        return pd.DataFrame(
            columns=["symbol", "ts", "open", "high", "low", "close",
                     "volume", "vwap", "trade_count"]
        )
    return pd.DataFrame(rows).sort_values(["symbol", "ts"]).reset_index(drop=True)


def _close_by_date(bars: dict[str, list[Bar]]) -> dict[str, dict[date, float]]:
    """symbol -> {date: close} lookup for cheap next-day-return math."""
    out: dict[str, dict[date, float]] = {}
    for sym, series in bars.items():
        out[sym] = {b.ts.date(): float(b.close) for b in series if b.close and b.close > 0}
    return out


def _all_trading_dates(bars: dict[str, list[Bar]]) -> list[date]:
    """Union of bar dates across all symbols, sorted ascending."""
    ds: set[date] = set()
    for series in bars.values():
        for b in series:
            ds.add(b.ts.date())
    return sorted(ds)


def backfill_from_bars(
    db: QuestDBClient,
    alpaca: AlpacaFeed,
    yahoo: YahooFeed,
    symbols: list[str],
    n_days: int = 60,
    session_hour_et: int = 13,
) -> int:
    """Return the number of (symbol, date) training rows persisted to QuestDB.

    Fetches a bar window slightly larger than `n_days` (to have room for the
    T+1 forward-return lookup), then iterates the most-recent `n_days` dates
    as decision timestamps.
    """
    if not symbols:
        log.warning("backfill_empty_universe")
        return 0

    log.info("backfill_start", n_symbols=len(symbols), n_days=n_days)

    # Extra 60d buffer covers holidays + forward-return date + rolling window seed.
    bars = alpaca.get_bars(symbols, days=n_days + 60)
    if not any(bars.values()):
        log.error("backfill_no_bars_returned")
        return 0

    # yfinance fundamentals — one snapshot reused across all historical dates.
    # See module docstring for the leakage/approximation trade-off.
    fundamentals = yahoo.get_fundamentals(symbols)

    all_dates = _all_trading_dates(bars)
    closes = _close_by_date(bars)

    if len(all_dates) < 2:
        log.error("backfill_insufficient_history", n_dates=len(all_dates))
        return 0

    # Use only dates with a next-day close available for forward-return.
    usable_dates = all_dates[:-1]
    if len(usable_dates) > n_days:
        usable_dates = usable_dates[-n_days:]

    n_written = 0
    with db.sender() as s:
        for i, date_t in enumerate(usable_dates):
            date_next = all_dates[all_dates.index(date_t) + 1]

            # Point-in-time feature computation: bars only up to date_t.
            bars_df = _bars_slice_df(bars, date_t)
            if bars_df.empty:
                continue

            try:
                features = compute_all(
                    bars=bars_df,
                    fundamentals=fundamentals,
                    quotes=None,
                    as_of_date=date_t,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("backfill_compute_all_failed", date=date_t.isoformat(), error=str(e))
                continue
            if features.empty:
                continue

            # Fix timestamp mid-session so it lands cleanly in the daily partition.
            ts_persist = datetime.combine(
                date_t, datetime.min.time().replace(hour=session_hour_et), tzinfo=ET
            )

            date_written_this_pass = 0
            for sym in features.index:
                close_t = closes.get(sym, {}).get(date_t)
                close_next = closes.get(sym, {}).get(date_next)
                if not close_t or not close_next or close_t <= 0:
                    continue
                realized_return_bps = 10_000.0 * (close_next - close_t) / close_t

                row = features.loc[sym]

                # --- Feature row (features table) ---
                feature_cols: dict[str, float | int] = {}
                for col in _CANONICAL_FEATURE_COLS:
                    val = row.get(col)
                    if val is None or pd.isna(val):
                        continue
                    try:
                        feature_cols[col] = float(val)
                    except (TypeError, ValueError):
                        continue
                if not feature_cols:
                    continue
                # meta weights: full quality (no live-quote uncertainty in backfill)
                feature_cols["freshness_weight"] = 1.0
                feature_cols["data_quality"] = 1.0
                s.row(
                    "features",
                    symbols={"symbol": str(sym)},
                    columns=feature_cols,
                    at=ts_persist,
                )

                # --- Synthetic execution row (executions table) ---
                # order_id is deterministic per (symbol, date) so re-runs overwrite
                # cleanly rather than accumulating duplicates.
                order_id = f"backfill_{sym}_{date_t.isoformat()}"
                s.row(
                    "executions",
                    symbols={
                        "order_id": order_id,
                        "symbol": str(sym),
                        "action_type": "BACKFILL",
                        "side": "buy",
                        "horizon": "1d",
                        "session": "regular",
                    },
                    columns={
                        "qty": 0.0,
                        "fill_price": close_t,
                        "decision_ref_price": close_t,
                        "slippage_bps": 0.0,
                        "spread_bps": 0.0,
                        "market_impact_bps": 0.0,
                        "model_edge_bps": 0.0,
                        "realized_return_bps": realized_return_bps,
                        "day_trade": False,
                    },
                    at=ts_persist,
                )

                date_written_this_pass += 1
                n_written += 1

            if (i + 1) % 10 == 0 or i == len(usable_dates) - 1:
                log.info(
                    "backfill_progress",
                    dates_done=i + 1,
                    dates_total=len(usable_dates),
                    rows_this_date=date_written_this_pass,
                    total_rows=n_written,
                )

    log.info("backfill_complete", n_rows=n_written, n_dates=len(usable_dates))
    return n_written


def _resolve_universe_from_config(cfg: JuniAutoConfig) -> list[str]:
    if cfg.universe.symbols:
        return list(cfg.universe.symbols)
    log.warning("backfill_no_config_universe_fallback_seeds")
    return ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"]


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="juniauto-backfill",
                                description="Bootstrap Bayesian training from historical bars.")
    p.add_argument("--config", default="/app/config/production.yaml", help="path to production.yaml")
    p.add_argument("--days", type=int, default=60, help="number of trading days to backfill (default 60)")
    p.add_argument("--symbols", default="",
                   help="comma-separated symbol override (default: config universe)")
    p.add_argument("--retrain", action="store_true",
                   help="call bayes.retrain_from_db() after backfill (default: off)")
    args = p.parse_args(list(argv) if argv is not None else None)

    cfg = load_config(args.config)
    configure_logging(level=cfg.logging.level, json_file=None)

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = _resolve_universe_from_config(cfg)

    db = QuestDBClient(cfg.database)
    alpaca = AlpacaFeed(cfg.alpaca)
    yahoo = YahooFeed(
        ttl_days=cfg.yahoo.fundamentals_ttl_days,
        enabled=cfg.yahoo.enabled,
        max_workers=cfg.yahoo.max_workers,
        per_symbol_timeout_seconds=cfg.yahoo.per_symbol_timeout_seconds,
    )

    n = backfill_from_bars(db=db, alpaca=alpaca, yahoo=yahoo,
                            symbols=symbols, n_days=args.days)
    log.info("backfill_summary", rows_written=n)

    if args.retrain:
        try:
            model = BayesianModel(db, cfg)
            n_train = model.retrain_from_db()
            log.info("backfill_retrain_complete", n_samples=n_train,
                     is_trained=model.is_trained())
        except Exception as e:  # noqa: BLE001
            log.error("backfill_retrain_failed", error=str(e))
            return 2
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
