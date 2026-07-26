"""Post-run analytics for backtest equity curves.

Reads per-curve equity from `backtest_equity_curve`, computes the full
metric inventory from the coordinator design review, writes results as
long/tall rows into `backtest_metrics(run_id, curve_type, metric_name,
metric_value, metric_unit)`.

Metric categories:
    - Return: total, CAGR, annualized_vol, sharpe, sortino, calmar
    - Alpha/beta: CAPM regression vs SPY (alpha_bps, beta, r_squared, alpha_tstat)
    - Drawdown: max_dd_pct, max_dd_days, avg_recovery_days, current_dd_pct
    - Tail: var_95_bps, cvar_95_bps, downside_deviation_bps
    - Operational: turnover_annualized, hit_rate, avg_win_bps, avg_loss_bps,
                   n_trades, n_days_pdt_blocked, cost_drag_bps_annual
    - Regime: sharpe_by_stress_quintile_{Q1..Q5}

All metrics operate on the equity CURVE (daily marks), NOT on individual
trade returns — that's the industry norm for portfolio-level reporting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

import numpy as np
import pandas as pd

from juniauto.db import QuestDBClient
from juniauto.utils import get_logger
from juniauto.utils.time_utils import ET

log = get_logger(__name__)


RF_ANNUAL = 0.045                     # 4.5% short-rate default; matches FED_FUNDS_RATE env
TRADING_DAYS_PER_YEAR = 252
QUINTILE_LABELS = ["Q1_calm", "Q2", "Q3", "Q4", "Q5_stress"]


@dataclass
class MetricRow:
    name: str
    value: float
    unit: str


# ================================================================
# Public API
# ================================================================
def compute_and_persist_metrics(
    db: QuestDBClient,
    *,
    run_id: str,
    curve_types: Iterable[str] = ("main", "unconstrained", "benchmark_spy",
                                   "benchmark_ew", "benchmark_fixed5"),
) -> dict[str, list[MetricRow]]:
    """Compute the full metric inventory for each curve_type present in
    backtest_equity_curve for `run_id`. Returns a dict[curve_type -> [MetricRow]].
    Rows are also persisted to backtest_metrics via ILP.
    """
    # Load all equity curves for this run into a DataFrame:
    #   index=ts, columns=(curve_type, equity), (curve_type, daily_return_bps)
    curves = _load_equity_curves(db, run_id)
    if not curves:
        log.warning("no_equity_curves_for_run", run_id=run_id)
        return {}

    spy_curve = curves.get("benchmark_spy")
    if spy_curve is None:
        log.warning("no_spy_benchmark_capm_will_skip", run_id=run_id)

    # Cost drag needs the executions table; regime bucketing needs backtest_market_regime.
    fills_by_curve = _load_fills(db, run_id)
    regime_series = _load_regime(db, run_id)

    all_rows: dict[str, list[MetricRow]] = {}
    now_ts = datetime.now(tz=ET)
    for curve_type in curve_types:
        c = curves.get(curve_type)
        if c is None or c.empty:
            continue
        rows: list[MetricRow] = []
        rows.extend(_return_risk_metrics(c["equity"]))
        rows.extend(_drawdown_metrics(c["equity"]))
        rows.extend(_tail_metrics(c["daily_return_bps"] / 10_000.0))
        if spy_curve is not None and curve_type != "benchmark_spy":
            rows.extend(_capm_metrics(c["daily_return_bps"] / 10_000.0,
                                       spy_curve["daily_return_bps"] / 10_000.0))
        # Ops metrics only apply to the algo curves (not benchmarks).
        if curve_type in ("main", "unconstrained"):
            rows.extend(_operational_metrics(c, fills_by_curve.get(curve_type, pd.DataFrame())))
        # Regime-conditioned Sharpe: only meaningful if regime observations exist.
        if regime_series is not None and not regime_series.empty:
            rows.extend(_regime_quintile_sharpe(c, regime_series))

        _persist_metric_rows(db, run_id=run_id, curve_type=curve_type, rows=rows, ts=now_ts)
        all_rows[curve_type] = rows
        log.info(
            "metrics_computed",
            run_id=run_id, curve_type=curve_type,
            n_metrics=len(rows),
            sample={r.name: round(r.value, 4) for r in rows[:5]},
        )

    return all_rows


# ================================================================
# Loading helpers
# ================================================================
def _load_equity_curves(db: QuestDBClient, run_id: str) -> dict[str, pd.DataFrame]:
    rows = db.query(
        """
        SELECT ts, curve_type, equity, daily_return_bps, cum_return_bps
          FROM backtest_equity_curve
         WHERE run_id = %s
         ORDER BY ts ASC
        """,
        (run_id,),
    )
    if not rows:
        return {}
    df = pd.DataFrame(rows, columns=["ts", "curve_type", "equity", "daily_return_bps", "cum_return_bps"])
    df["ts"] = pd.to_datetime(df["ts"])
    out: dict[str, pd.DataFrame] = {}
    for ct, sub in df.groupby("curve_type"):
        sub = sub.set_index("ts").sort_index()
        # Dedup on timestamp — earlier failed runs of the same run_id leave
        # rows behind since QuestDB WAL is append-only. Keep the LAST write
        # per (ts, curve_type) so the most recent successful run wins.
        sub = sub[~sub.index.duplicated(keep="last")]
        # Recompute daily returns from equity to catch cases where the writer
        # left them at 0.0 (e.g. main curve during backtest — engine writes 0).
        sub["daily_return_bps"] = 10_000.0 * sub["equity"].pct_change().fillna(0.0)
        out[str(ct)] = sub
    return out


def _load_fills(db: QuestDBClient, run_id: str) -> dict[str, pd.DataFrame]:
    rows = db.query(
        """
        SELECT ts, curve_type, symbol, side, qty, fill_price, slippage_bps
          FROM backtest_executions
         WHERE run_id = %s
         ORDER BY ts ASC
        """,
        (run_id,),
    )
    if not rows:
        return {}
    df = pd.DataFrame(rows, columns=["ts", "curve_type", "symbol", "side", "qty", "fill_price", "slippage_bps"])
    df["ts"] = pd.to_datetime(df["ts"])
    return {str(ct): sub.reset_index(drop=True) for ct, sub in df.groupby("curve_type")}


def _load_regime(db: QuestDBClient, run_id: str) -> pd.Series | None:
    rows = db.query(
        """
        SELECT ts, stress_ema
          FROM backtest_market_regime
         WHERE run_id = %s
         ORDER BY ts ASC
        """,
        (run_id,),
    )
    if not rows:
        return None
    ser = pd.Series(
        [float(v) if v is not None else float("nan") for _, v in rows],
        index=pd.DatetimeIndex([t for t, _ in rows]),
        name="stress_ema",
    ).dropna()
    if ser.empty:
        return None
    # Dedup on timestamp — same rationale as _load_equity_curves.
    ser = ser[~ser.index.duplicated(keep="last")]
    return ser


# ================================================================
# Return + risk
# ================================================================
def _return_risk_metrics(equity: pd.Series) -> list[MetricRow]:
    equity = equity.dropna()
    if len(equity) < 2:
        return []
    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    n_days = len(equity)
    n_years = n_days / TRADING_DAYS_PER_YEAR
    cagr = (1.0 + total_ret) ** (1.0 / max(n_years, 1e-9)) - 1.0
    daily_ret = equity.pct_change().dropna()
    ann_vol = float(daily_ret.std(ddof=1)) * math.sqrt(TRADING_DAYS_PER_YEAR)
    excess = daily_ret - (RF_ANNUAL / TRADING_DAYS_PER_YEAR)
    sharpe = float(excess.mean() / excess.std(ddof=1)) * math.sqrt(TRADING_DAYS_PER_YEAR) if excess.std(ddof=1) > 0 else 0.0
    downside = daily_ret[daily_ret < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = (
        float(excess.mean() / downside_std) * math.sqrt(TRADING_DAYS_PER_YEAR)
        if downside_std > 0 else 0.0
    )
    max_dd = _max_drawdown(equity)
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    return [
        MetricRow("total_return_pct", total_ret * 100.0, "pct"),
        MetricRow("cagr_pct", cagr * 100.0, "pct"),
        MetricRow("annualized_vol_pct", ann_vol * 100.0, "pct"),
        MetricRow("sharpe_annualized", sharpe, "ratio"),
        MetricRow("sortino_annualized", sortino, "ratio"),
        MetricRow("calmar", calmar, "ratio"),
        MetricRow("n_trading_days", float(n_days), "count"),
    ]


# ================================================================
# Drawdown
# ================================================================
def _drawdown_metrics(equity: pd.Series) -> list[MetricRow]:
    equity = equity.dropna()
    if len(equity) < 2:
        return []
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    max_dd = float(dd.min())
    trough_idx = dd.idxmin()
    # Duration = days from previous peak to trough
    prior_peak = equity.loc[:trough_idx].idxmax()
    max_dd_days = float((trough_idx - prior_peak).days) if prior_peak is not None else 0.0

    # Current drawdown vs latest peak
    current_dd = float(dd.iloc[-1])

    # Average recovery days: identify each drawdown episode (peak → recovery)
    recoveries = _recovery_days(equity)
    avg_recovery = float(np.mean(recoveries)) if recoveries else 0.0

    return [
        MetricRow("max_drawdown_pct", max_dd * 100.0, "pct"),
        MetricRow("max_drawdown_days", max_dd_days, "days"),
        MetricRow("current_drawdown_pct", current_dd * 100.0, "pct"),
        MetricRow("avg_recovery_days", avg_recovery, "days"),
        MetricRow("ulcer_index", _ulcer_index(equity), "pct"),
    ]


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    return float((equity / running_max - 1.0).min())


def _ulcer_index(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd_pct = (equity / running_max - 1.0) * 100.0
    return float(math.sqrt((dd_pct ** 2).mean()))


def _recovery_days(equity: pd.Series) -> list[float]:
    running_max = equity.cummax()
    below = equity < running_max
    if not below.any():
        return []
    recoveries = []
    in_dd = False
    dd_start = None
    for ts, is_below in below.items():
        if is_below and not in_dd:
            in_dd = True
            dd_start = ts
        elif not is_below and in_dd:
            recoveries.append(float((ts - dd_start).days))
            in_dd = False
            dd_start = None
    return recoveries


# ================================================================
# Tail
# ================================================================
def _tail_metrics(daily_ret: pd.Series) -> list[MetricRow]:
    daily_ret = daily_ret.dropna()
    if len(daily_ret) < 20:
        return []
    var_95 = float(np.percentile(daily_ret, 5))
    tail = daily_ret[daily_ret <= var_95]
    cvar_95 = float(tail.mean()) if len(tail) > 0 else var_95
    downside = daily_ret[daily_ret < 0]
    downside_dev = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    return [
        MetricRow("var_95_daily_bps", var_95 * 10_000.0, "bps"),
        MetricRow("cvar_95_daily_bps", cvar_95 * 10_000.0, "bps"),
        MetricRow("downside_deviation_daily_bps", downside_dev * 10_000.0, "bps"),
    ]


# ================================================================
# CAPM alpha/beta
# ================================================================
def _capm_metrics(portfolio_ret: pd.Series, spy_ret: pd.Series) -> list[MetricRow]:
    # Align on common dates
    df = pd.concat([portfolio_ret.rename("p"), spy_ret.rename("m")], axis=1).dropna()
    if len(df) < 30:
        return []
    rf_daily = RF_ANNUAL / TRADING_DAYS_PER_YEAR
    y = (df["p"] - rf_daily).to_numpy()
    x = (df["m"] - rf_daily).to_numpy()
    n = len(y)
    x_mean = x.mean()
    y_mean = y.mean()
    ss_xx = float(((x - x_mean) ** 2).sum())
    if ss_xx <= 0:
        return []
    beta = float(((x - x_mean) * (y - y_mean)).sum() / ss_xx)
    alpha = float(y_mean - beta * x_mean)
    y_hat = alpha + beta * x
    residuals = y - y_hat
    ss_res = float((residuals ** 2).sum())
    ss_tot = float(((y - y_mean) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # Standard error of alpha:
    #   SE(alpha) = sqrt(sigma_hat^2 * (1/n + x_mean^2 / ss_xx))
    sigma_hat_sq = ss_res / max(1, n - 2)
    se_alpha = math.sqrt(sigma_hat_sq * (1.0 / n + x_mean ** 2 / ss_xx))
    t_alpha = alpha / se_alpha if se_alpha > 0 else 0.0
    # Annualize alpha to bps/year
    alpha_annual_bps = alpha * TRADING_DAYS_PER_YEAR * 10_000.0
    info_ratio = (
        (alpha / math.sqrt(sigma_hat_sq)) * math.sqrt(TRADING_DAYS_PER_YEAR)
        if sigma_hat_sq > 0 else 0.0
    )
    return [
        MetricRow("capm_alpha_annualized_bps", alpha_annual_bps, "bps"),
        MetricRow("capm_beta", beta, "ratio"),
        MetricRow("capm_r_squared", r_squared, "ratio"),
        MetricRow("capm_alpha_tstat", t_alpha, "ratio"),
        MetricRow("information_ratio_annualized", info_ratio, "ratio"),
    ]


# ================================================================
# Operational
# ================================================================
def _operational_metrics(curve: pd.DataFrame, fills: pd.DataFrame) -> list[MetricRow]:
    equity = curve["equity"].dropna()
    if equity.empty:
        return []
    n_days = len(equity)
    n_years = n_days / TRADING_DAYS_PER_YEAR
    # PDT-blocked days
    if "pdt_blocked" in curve.columns:
        n_pdt_blocked = int(curve["pdt_blocked"].sum())
    else:
        n_pdt_blocked = 0
    rows: list[MetricRow] = [
        MetricRow("n_days_pdt_blocked", float(n_pdt_blocked), "count"),
    ]
    if fills.empty:
        rows.append(MetricRow("n_fills", 0.0, "count"))
        return rows
    n_fills = int(len(fills))
    n_buys = int((fills["side"] == "buy").sum())
    n_sells = int((fills["side"] == "sell").sum())
    turnover_ann = n_buys * 0.05 / max(n_years, 1e-9)  # 5% notional per buy proxy
    hit_rate = _hit_rate_from_fills(fills)
    avg_slippage = float(fills["slippage_bps"].mean())
    rows.extend([
        MetricRow("n_fills", float(n_fills), "count"),
        MetricRow("n_buys", float(n_buys), "count"),
        MetricRow("n_sells", float(n_sells), "count"),
        MetricRow("turnover_annualized_gross", turnover_ann, "ratio"),
        MetricRow("avg_slippage_bps", avg_slippage, "bps"),
        MetricRow("hit_rate", hit_rate, "ratio"),
    ])
    return rows


def _hit_rate_from_fills(fills: pd.DataFrame) -> float:
    """Approximate hit rate: fraction of buy-then-sell round trips where
    sell price > buy price. Since SimBroker doesn't tag round trips
    explicitly, we FIFO-match per symbol."""
    if fills.empty:
        return 0.0
    wins = 0
    trips = 0
    by_sym: dict[str, list[tuple[float, float]]] = {}  # symbol -> [(qty, price), ...]
    for _, row in fills.sort_values("ts").iterrows():
        sym = str(row["symbol"])
        side = str(row["side"])
        qty = float(row["qty"])
        px = float(row["fill_price"])
        if side == "buy":
            by_sym.setdefault(sym, []).append((qty, px))
        else:  # sell
            queue = by_sym.get(sym, [])
            remaining = qty
            while remaining > 1e-9 and queue:
                q0, p0 = queue[0]
                match = min(q0, remaining)
                trips += 1
                if px > p0:
                    wins += 1
                remaining -= match
                if match >= q0 - 1e-9:
                    queue.pop(0)
                else:
                    queue[0] = (q0 - match, p0)
    return wins / trips if trips > 0 else 0.0


# ================================================================
# Regime-conditioned Sharpe
# ================================================================
def _regime_quintile_sharpe(curve: pd.DataFrame, regime_series: pd.Series) -> list[MetricRow]:
    daily_ret = curve["equity"].pct_change().dropna()
    df = pd.concat([daily_ret.rename("r"), regime_series.rename("s")], axis=1).dropna()
    if len(df) < 20:
        return []
    # Quintile-bucket by stress_ema
    try:
        df["bucket"] = pd.qcut(df["s"], q=5, labels=QUINTILE_LABELS, duplicates="drop")
    except ValueError:
        return []
    rows: list[MetricRow] = []
    for label in QUINTILE_LABELS:
        sub = df[df["bucket"] == label]
        if len(sub) < 5:
            continue
        excess = sub["r"] - RF_ANNUAL / TRADING_DAYS_PER_YEAR
        std = float(excess.std(ddof=1))
        sharpe = float(excess.mean() / std) * math.sqrt(TRADING_DAYS_PER_YEAR) if std > 0 else 0.0
        rows.append(MetricRow(f"sharpe_by_regime_{label}", sharpe, "ratio"))
        rows.append(MetricRow(f"n_days_regime_{label}", float(len(sub)), "count"))
    return rows


# ================================================================
# Persistence
# ================================================================
def _persist_metric_rows(
    db: QuestDBClient,
    *,
    run_id: str,
    curve_type: str,
    rows: list[MetricRow],
    ts: datetime,
) -> None:
    if not rows:
        return
    with db.sender() as s:
        for r in rows:
            s.row(
                "backtest_metrics",
                symbols={
                    "run_id": run_id, "curve_type": curve_type,
                    "metric_name": r.name, "metric_unit": r.unit,
                },
                columns={"metric_value": float(r.value)},
                at=ts,
            )
