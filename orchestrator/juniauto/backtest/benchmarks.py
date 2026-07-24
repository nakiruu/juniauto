"""Benchmark equity curves for backtest comparison.

Each function computes a daily equity curve over the same trading-day
window as the main run and writes it to backtest_equity_curve under a
distinct `curve_type` label so Grafana can overlay them. All benchmarks
start from the same `initial_cash` as the main curve for apples-to-
apples percentage comparison.

Ranked (per coordinator design review):
    1. SPY buy-and-hold          -> curve_type='benchmark_spy'
    2. Equal-weight universe     -> curve_type='benchmark_ew'
       (monthly rebalanced across the same seed universe)
    3. Fixed-5% per name         -> curve_type='benchmark_fixed5'
       (sizes the main run's gateway-executed candidates at 5% each)
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from juniauto.db import QuestDBClient
from juniauto.utils import get_logger
from juniauto.utils.time_utils import ET

log = get_logger(__name__)


CURVE_BENCH_SPY = "benchmark_spy"
CURVE_BENCH_EW = "benchmark_ew"
CURVE_BENCH_FIXED5 = "benchmark_fixed5"


def benchmark_spy(
    db: QuestDBClient,
    *,
    run_id: str,
    start_date: date,
    end_date: date,
    initial_cash: float,
) -> pd.DataFrame:
    """Fully-invested SPY buy-and-hold. Buys at first-day close, marks daily
    at close. Cash after buy is the fractional remainder (typically small)."""
    closes = _load_daily_closes(db, "SPY", start_date, end_date)
    if closes.empty:
        log.warning("bench_spy_no_bars", start=str(start_date), end=str(end_date))
        return pd.DataFrame()
    first_px = float(closes.iloc[0])
    shares = int(initial_cash // first_px)
    cash_remainder = initial_cash - shares * first_px
    curve = shares * closes + cash_remainder
    return _write_curve(
        db, run_id=run_id, curve_type=CURVE_BENCH_SPY,
        ts_index=closes.index, equity=curve, cash=cash_remainder, initial_cash=initial_cash,
    )


def benchmark_equal_weight(
    db: QuestDBClient,
    *,
    run_id: str,
    start_date: date,
    end_date: date,
    initial_cash: float,
    universe: list[str],
    rebalance_days: int = 21,
) -> pd.DataFrame:
    """Equal-weight over the seed universe with monthly rebalance.

    On each rebalance date, distributes current equity equally across
    every symbol that has a valid close bar on that date. Symbols with
    missing bars are skipped (their weight goes to cash).
    """
    # Load a wide close panel: rows=date, cols=symbol.
    panel = _load_close_panel(db, universe, start_date, end_date)
    if panel.empty:
        return pd.DataFrame()

    dates = panel.index.tolist()
    # Determine rebalance date indices (every `rebalance_days` cycles).
    rebal_idxs = list(range(0, len(dates), max(1, rebalance_days)))
    if rebal_idxs and rebal_idxs[-1] != len(dates) - 1:
        pass  # last rebalance stays; no re-rebalance at final bar

    # Shares held per symbol; changes on rebalance days.
    shares: dict[str, float] = {}
    cash = initial_cash
    equity_series = []

    r_i = 0  # next rebalance index cursor
    for i, d in enumerate(dates):
        if r_i < len(rebal_idxs) and i == rebal_idxs[r_i]:
            # Rebalance: sell everything at today's close, redistribute
            mv = sum(float(panel.iloc[i][s]) * qty for s, qty in shares.items() if not pd.isna(panel.iloc[i][s]))
            total_equity = cash + mv
            # Redistribute over symbols with valid close today
            valid_syms = [s for s in universe if not pd.isna(panel.iloc[i].get(s, float("nan")))]
            if valid_syms:
                per_name = total_equity / len(valid_syms)
                shares = {}
                for s in valid_syms:
                    px = float(panel.iloc[i][s])
                    if px > 0:
                        shares[s] = per_name / px
                cash = 0.0
            r_i += 1

        # Mark equity at today's close
        mv = 0.0
        for s, qty in shares.items():
            px = panel.iloc[i].get(s, float("nan"))
            if not pd.isna(px):
                mv += float(px) * qty
        equity_series.append(cash + mv)

    idx = pd.DatetimeIndex(dates)
    return _write_curve(
        db, run_id=run_id, curve_type=CURVE_BENCH_EW,
        ts_index=idx, equity=pd.Series(equity_series, index=idx),
        cash=0.0, initial_cash=initial_cash,
    )


def benchmark_fixed5(
    db: QuestDBClient,
    *,
    run_id: str,
    start_date: date,
    end_date: date,
    initial_cash: float,
) -> pd.DataFrame:
    """Replay the main-curve's gateway-executed candidates but size everyone
    at a flat 5% (capped by max_name_weight upstream). Reads the actual
    backtest_gateway_actions rows for this run to preserve the same
    signal/gateway decisions, then computes returns from close-to-close
    bar prices for the held basket.

    This is the "did Kelly beat naive equal-weight-on-signal" test —
    counterpart to the live shadow_ev_delta_bps metric.
    """
    # Pull one row per (ts, symbol) where the main-curve action executed.
    rows = db.query(
        """
        SELECT ts::date AS d, symbol
          FROM backtest_gateway_actions
         WHERE run_id = %s AND curve_type = 'main' AND executed = TRUE
         ORDER BY d ASC, symbol ASC
        """,
        (run_id,),
    )
    if not rows:
        log.info("bench_fixed5_no_actions", run_id=run_id)
        return pd.DataFrame()

    # Group by day into the target basket for that day.
    basket_by_day: dict[date, list[str]] = {}
    for d, sym in rows:
        basket_by_day.setdefault(d, []).append(sym)

    days = sorted(basket_by_day.keys())
    if not days:
        return pd.DataFrame()

    # Load close panel for every symbol seen.
    all_syms = sorted({s for b in basket_by_day.values() for s in b})
    panel = _load_close_panel(db, all_syms, start_date, end_date)
    if panel.empty:
        return pd.DataFrame()

    # Simple carry-forward: on day d, hold equal-weight (5% each, cash to 100%)
    # basket; on d+1 rebalance to the next day's basket. Between rebalances
    # equity moves with the held prices.
    equity = initial_cash
    shares: dict[str, float] = {}
    cash = initial_cash
    curve_dates = list(panel.index)
    equity_series = []

    for d_ts in curve_dates:
        d = d_ts.date() if isinstance(d_ts, datetime) else d_ts
        # Mark to today's close
        mv = 0.0
        for s, qty in shares.items():
            px = panel.loc[d_ts].get(s, float("nan"))
            if not pd.isna(px):
                mv += float(px) * qty
        equity = cash + mv

        # If this is a signal day, rebalance to today's basket at close.
        if d in basket_by_day:
            basket = basket_by_day[d]
            # Take today's closes as fill prices (approximation — same as
            # 'close' fill_model for the main curve at 5%/name).
            valid = [(s, float(panel.loc[d_ts][s])) for s in basket
                     if not pd.isna(panel.loc[d_ts].get(s, float("nan")))]
            if valid:
                per_name_notional = min(0.05, 1.0 / len(valid)) * equity
                shares = {s: per_name_notional / px for s, px in valid if px > 0}
                spent = sum(qty * px for s, qty in shares.items() for _s, px in valid if _s == s)
                cash = max(0.0, equity - spent)
        equity_series.append(equity)

    idx = pd.DatetimeIndex(curve_dates)
    return _write_curve(
        db, run_id=run_id, curve_type=CURVE_BENCH_FIXED5,
        ts_index=idx, equity=pd.Series(equity_series, index=idx),
        cash=cash, initial_cash=initial_cash,
    )


# ================================================================
# Internals
# ================================================================
def _load_daily_closes(
    db: QuestDBClient, symbol: str, start_date: date, end_date: date
) -> pd.Series:
    rows = db.query(
        """
        SELECT ts::date AS d, close
          FROM bars
         WHERE symbol = %s
           AND ts::date >= %s::date
           AND ts::date <= %s::date
         ORDER BY ts ASC
        """,
        (symbol, start_date, end_date),
    )
    if not rows:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex([d for d, _ in rows])
    return pd.Series([float(c) for _, c in rows], index=idx, name=symbol)


def _load_close_panel(
    db: QuestDBClient, symbols: list[str], start_date: date, end_date: date
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    quoted = ",".join(f"'{s}'" for s in symbols)
    rows = db.query(
        f"""
        SELECT ts::date AS d, symbol, close
          FROM bars
         WHERE symbol IN ({quoted})
           AND ts::date >= %s::date
           AND ts::date <= %s::date
         ORDER BY d ASC
        """,
        (start_date, end_date),
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["d", "symbol", "close"])
    df["d"] = pd.to_datetime(df["d"])
    return df.pivot_table(index="d", columns="symbol", values="close", aggfunc="last")


def _write_curve(
    db: QuestDBClient,
    *,
    run_id: str,
    curve_type: str,
    ts_index,
    equity,
    cash: float,
    initial_cash: float,
) -> pd.DataFrame:
    """Persist an equity curve as backtest_equity_curve rows. Also returns
    the DataFrame for direct use by the metrics stage."""
    equity = pd.Series(equity)
    equity.index = pd.DatetimeIndex(ts_index)
    daily_ret_bps = 10_000.0 * equity.pct_change().fillna(0.0)
    cum_ret_bps = 10_000.0 * (equity / initial_cash - 1.0)
    with db.sender() as s:
        for ts, eq in equity.items():
            s.row(
                "backtest_equity_curve",
                symbols={"run_id": run_id, "curve_type": curve_type},
                columns={
                    "equity": float(eq),
                    "cash": float(cash),
                    "invested": float(eq - cash),
                    "position_count": 1 if curve_type == CURVE_BENCH_SPY else 0,
                    "pdt_day_trade_count": 0,
                    "pdt_blocked": False,
                    "daily_return_bps": float(daily_ret_bps.loc[ts]),
                    "cum_return_bps": float(cum_ret_bps.loc[ts]),
                },
                at=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            )
    out = pd.DataFrame({
        "equity": equity,
        "daily_return_bps": daily_ret_bps,
        "cum_return_bps": cum_ret_bps,
    })
    return out
