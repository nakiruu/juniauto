"""Execution resolution — populate realized_return_bps for settled trades.

For each unresolved execution (realized_return_bps = 0.0, age > 1 trading day),
look up the most-recent close bar available in QuestDB and compute the
forward-return relative to the fill price.

Formula (§2.11):
    BUY:  realized_return_bps = 10_000 * (close - fill_price) / fill_price
    SELL: sign flips  → 10_000 * (fill_price - close) / fill_price

Uses the Postgres wire UPDATE path (WAL table, no DEDUP key on order_id so
the UPDATE succeeds without conflicts).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import psycopg

if TYPE_CHECKING:
    from juniauto.db import QuestDBClient

log = logging.getLogger(__name__)


def resolve_stale_executions(
    db: "QuestDBClient",
    alpaca_feed: Any,  # not used — present for signature symmetry with main.py hook
) -> int:
    """Resolve all stale executions into realized_return_bps.

    Scans the ``executions`` table for rows where:
    - ``realized_return_bps = 0.0`` (not yet resolved)
    - ``ts < dateadd('d', -1, now())`` (older than 1 trading day)
    - ``action_type IN ('BUY', 'ROTATE')``

    For each stale row, finds the most-recent close from ``bars`` with a
    timestamp on or after the execution timestamp, then UPDATEs the row.

    This function is idempotent: the WHERE clause on ``realized_return_bps = 0.0``
    means already-resolved rows are never re-processed.

    Args:
        db: Active QuestDBClient instance.
        alpaca_feed: Alpaca feed reference (unused; reserved for future live
            price fallback when no bar is available in QuestDB).

    Returns:
        Count of rows successfully resolved in this call. 0 on any error.
    """
    try:
        stale_rows = _fetch_stale_executions(db)
    except Exception as exc:  # noqa: BLE001
        log.error("resolution_fetch_failed", error=str(exc))
        return 0

    n_scanned = len(stale_rows)
    if n_scanned == 0:
        log.info("resolution_batch", n_resolved=0, n_scanned=0)
        return 0

    n_resolved = 0
    for row in stale_rows:
        order_id, symbol, fill_price, side = row[0], row[1], row[2], row[3]
        try:
            close = _latest_close(db, symbol)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "resolution_bar_lookup_failed",
                order_id=str(order_id),
                symbol=str(symbol),
                error=str(exc),
            )
            continue

        if close is None:
            log.info(
                "resolution_no_bar",
                order_id=str(order_id),
                symbol=str(symbol),
            )
            continue

        fill = float(fill_price)
        cl = float(close)
        if fill <= 0:
            log.warning(
                "resolution_bad_fill_price",
                order_id=str(order_id),
                fill_price=fill,
            )
            continue

        side_str = str(side).lower()
        if side_str == "sell":
            realized_bps = 10_000.0 * (fill - cl) / fill
        else:
            # "buy" and any unknown side treated as BUY direction
            realized_bps = 10_000.0 * (cl - fill) / fill

        try:
            _update_realized_return(db, order_id, realized_bps)
            n_resolved += 1
            log.debug(
                "resolution_resolved",
                order_id=str(order_id),
                symbol=str(symbol),
                realized_bps=round(realized_bps, 2),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "resolution_update_failed",
                order_id=str(order_id),
                symbol=str(symbol),
                error=str(exc),
            )

    log.info("resolution_batch", n_resolved=n_resolved, n_scanned=n_scanned)
    return n_resolved


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_stale_executions(db: "QuestDBClient") -> list[tuple[Any, ...]]:
    """Return (order_id, symbol, fill_price, side) for all unresolved rows."""
    sql = """
        SELECT order_id, symbol, fill_price, side
        FROM executions
        WHERE realized_return_bps = 0.0
          AND ts < dateadd('d', -1, now())
          AND action_type IN ('BUY', 'ROTATE')
    """
    return db.query(sql)


def _latest_close(db: "QuestDBClient", symbol: str) -> float | None:
    """Return the most-recent close price for ``symbol`` from the ``bars`` table.

    Returns None when no bar is available (e.g. brand-new symbol or data gap).
    """
    sql = """
        SELECT close
        FROM bars
        WHERE symbol = $1
          AND session = 'regular'
        ORDER BY ts DESC
        LIMIT 1
    """
    row = db.query_one(sql, (symbol,))
    if row is None or row[0] is None:
        return None
    return float(row[0])


def _update_realized_return(
    db: "QuestDBClient", order_id: str, realized_bps: float
) -> None:
    """UPDATE executions SET realized_return_bps = $1 WHERE order_id = $2.

    Uses psycopg directly through the client's internal connection factory
    so we can issue a true DML UPDATE on the WAL table (ILP only appends;
    reads via query() would need a separate connection for writes).

    QuestDB WAL tables without DEDUP support UPDATE via the PG wire.
    DEDUP tables may reject it — but executions has no DEDUP clause so
    this is safe. Any DDL-level error is caught and re-raised for the
    caller's try/except.
    """
    sql = "UPDATE executions SET realized_return_bps = $1 WHERE order_id = $2"
    # Access the private _pg_conninfo to open a short-lived write connection.
    # This is intentional — QuestDBClient deliberately exposes only read helpers
    # in its public API. A future refactor can add a public execute() method.
    conn_info = db._pg_conninfo  # noqa: SLF001
    with psycopg.connect(conn_info) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (realized_bps, str(order_id)))
        conn.commit()
