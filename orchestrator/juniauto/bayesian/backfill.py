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

Also persists raw bars to the `bars` table so the backtest engine's
snapshot loader can find historical bars for the full window (before this
was added, backtests starting before the ~1yr live bar cache silently
produced empty cycles).

Usage (from the host):
    # Full backfill: bars + features + BACKFILL executions + retrain
    docker exec juniauto-engine python -m juniauto.bayesian.backfill --days 1600 --retrain

    # Bars-only (fast) — seed the bars table for a backtest without
    # regenerating the training set:
    docker exec juniauto-engine python -m juniauto.bayesian.backfill --days 1600 --bars-only

The backfill is idempotent per (symbol, date) — bars use QuestDB's
DEDUP UPSERT KEYS(ts, symbol), and re-writing the same executions rows
uses the same deterministic order_id so training reads see one row per
(symbol, date). order_id is STRING (not SYMBOL) — a large backfill
generates hundreds of thousands of unique IDs, which would overflow any
SYMBOL cap. See schema.sql for the rationale.
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


def _build_full_bars_df(bars: dict[str, list[Bar]]) -> pd.DataFrame:
    """Build the whole bars DataFrame ONCE — call outside the per-date loop.

    Previous impl rebuilt this per-date via a Python list-of-dicts, which
    was O(all_bars) per call × O(n_dates) calls = O(all_bars × n_dates).
    On a 1600-day backfill with 300 symbols that's ~765M tuple constructions
    — literally hours. Building once is ~1s.
    """
    rows: list[dict[str, object]] = []
    for sym, series in bars.items():
        for b in series:
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
                     "volume", "vwap", "trade_count", "_date"]
        )
    df = pd.DataFrame(rows).sort_values(["symbol", "ts"]).reset_index(drop=True)
    # Pre-compute per-bar date column so per-cutoff filtering is a
    # vectorized comparison instead of a Python-side .dt.date per call.
    df["_date"] = df["ts"].apply(lambda t: t.date() if hasattr(t, "date") else t)
    return df


def _bars_slice_df(df_all: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """O(n) vectorized filter of the pre-built bars DataFrame."""
    if df_all.empty:
        return df_all
    mask = df_all["_date"] <= cutoff
    return df_all.loc[mask].drop(columns=["_date"], errors="ignore")


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

    # Persist bars to the `bars` table so the backtest engine's snapshot
    # loader can find historical bars for the full window. Without this
    # step, backtests starting before the live pipeline's rolling ~1yr
    # bar cache would silently produce empty cycles. QuestDB WAL + DEDUP
    # KEYS(ts, symbol) makes this idempotent — re-running just no-ops.
    n_bars_written = _persist_bars(db, bars)
    log.info("backfill_bars_written", n_bars=n_bars_written,
             n_symbols=sum(1 for b in bars.values() if b))

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

    # Build the full bars DataFrame ONCE. Per-date filter is a cheap
    # vectorized mask below — was previously rebuilt from scratch every
    # iteration (O(n²) in bar count × n_dates), which on a 1600-day
    # backfill dominated wall-time.
    bars_df_all = _build_full_bars_df(bars)
    log.info("backfill_bars_df_built", n_rows=len(bars_df_all))

    n_written = 0
    with db.sender() as s:
        for i, date_t in enumerate(usable_dates):
            date_next = all_dates[all_dates.index(date_t) + 1]

            # Point-in-time feature computation: bars only up to date_t.
            bars_df = _bars_slice_df(bars_df_all, date_t)
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
                # cleanly rather than accumulating duplicates. order_id is STRING
                # (schema.sql) — must go in columns={} not symbols={} or the
                # SYMBOL cap will overflow at ~65k unique ids.
                order_id = f"backfill_{sym}_{date_t.isoformat()}"
                s.row(
                    "executions",
                    symbols={
                        "symbol": str(sym),
                        "action_type": "BACKFILL",
                        "side": "buy",
                        "horizon": "1d",
                        "session": "regular",
                    },
                    columns={
                        "order_id": order_id,
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


def _persist_bars(db: QuestDBClient, bars: dict[str, list[Bar]]) -> int:
    """Write every bar in `bars` to the QuestDB `bars` table via ILP.

    Idempotent: schema.sql declares `bars` as `WAL DEDUP UPSERT KEYS(ts, symbol)`,
    so re-writing the same (ts, symbol) rows leaves a single row in place.
    Session is hardcoded to 'regular' because Alpaca daily bars are always
    regular-session by definition.
    """
    n_written = 0
    with db.sender() as s:
        for sym, series in bars.items():
            for b in series:
                s.row(
                    "bars",
                    symbols={"symbol": sym, "session": "regular"},
                    columns={
                        "open": float(b.open),
                        "high": float(b.high),
                        "low": float(b.low),
                        "close": float(b.close),
                        "volume": int(b.volume or 0),
                        "vwap": float(b.vwap) if b.vwap is not None else 0.0,
                        "trade_count": int(b.trade_count) if b.trade_count is not None else 0,
                    },
                    at=b.ts,
                )
                n_written += 1
    return n_written


def backfill_bars_only(
    db: QuestDBClient,
    alpaca: AlpacaFeed,
    symbols: list[str],
    n_days: int,
) -> int:
    """Fetch and persist raw bars only — no feature computation, no
    training-row synthesis. Use this to seed the `bars` table quickly
    when you don't need to regenerate the Bayesian training set.
    """
    if not symbols:
        log.warning("backfill_bars_only_empty_universe")
        return 0
    log.info("backfill_bars_only_start", n_symbols=len(symbols), n_days=n_days)
    bars = alpaca.get_bars(symbols, days=n_days + 60)
    n = _persist_bars(db, bars)
    log.info("backfill_bars_only_complete", n_bars=n)
    return n


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
    p.add_argument("--bars-only", action="store_true",
                   help="only fetch + persist bars to QuestDB — skip feature "
                        "computation and Bayesian training-row synthesis. Fast; "
                        "use when you just need to seed the bars table for a "
                        "backtest.")
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

    if args.bars_only:
        n = backfill_bars_only(db=db, alpaca=alpaca, symbols=symbols, n_days=args.days)
        log.info("backfill_summary", mode="bars_only", bars_written=n)
        return 0 if n > 0 else 1

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
