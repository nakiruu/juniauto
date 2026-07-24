"""Resolve phantom-cycle rows with realized 1-trading-day forward returns.

Phantom cycles record what the decision pipeline WOULD have picked at a
non-15:55 decision time (e.g., 09:40 ET open, 12:30 ET midday) without
placing orders. Their realized-return backfill is different from real
executions: there's no fill price, so we compute return from bars alone:

    realized_return_bps = 10_000 * (close_next - close_at_cycle) / close_at_cycle

where close_at_cycle is looked up from the bars table on the phantom's ts
date, and close_next is the first bar close strictly after that date.

Kept separate from bayesian.resolution (which resolves real executions and
their fill prices) so phantom data never leaks into Bayesian training —
using our own predictions as training labels would be self-fulfilling.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import psycopg

from juniauto.utils import get_logger

if TYPE_CHECKING:
    from juniauto.db import QuestDBClient

log = get_logger(__name__)


def resolve_stale_phantoms(db: "QuestDBClient", batch_limit: int = 5000) -> int:
    """Backfill realized_return_bps for phantom rows older than 1 trading day
    where the field is still null. Returns count of rows resolved this pass.
    """
    try:
        # Rows to resolve: > 1 calendar day old (approximates 1 trading day; the
        # bars lookup will simply miss on weekends and try again next hour).
        rows = db.query(
            """
            SELECT symbol, ts, target_weight
              FROM phantom_gateway_actions
             WHERE realized_return_bps IS NULL
               AND target_weight > 0.0
               AND ts < dateadd('d', -1, now())
             LIMIT %s
            """,
            (batch_limit,),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("phantom_resolve_scan_failed", error=str(exc))
        return 0

    if not rows:
        return 0

    n_resolved = 0
    n_missing = 0
    n_errored = 0

    # Direct psycopg for UPDATEs; QuestDBClient exposes query() but no execute helper.
    conninfo = db._pg_conninfo  # noqa: SLF001 — private but stable across the module

    with psycopg.connect(conninfo) as conn:
        for sym, ts, _target_weight in rows:
            try:
                # close at cycle date
                close_at_cycle = db.query_one(
                    """
                    SELECT close FROM bars
                     WHERE symbol = %s
                       AND ts::date = %s::date
                     ORDER BY ts DESC
                     LIMIT 1
                    """,
                    (sym, ts),
                )
                # first bar close strictly after cycle date
                close_next_row = db.query_one(
                    """
                    SELECT close FROM bars
                     WHERE symbol = %s
                       AND ts::date > %s::date
                     ORDER BY ts ASC
                     LIMIT 1
                    """,
                    (sym, ts),
                )
                if not close_at_cycle or not close_next_row:
                    n_missing += 1
                    continue
                c0 = float(close_at_cycle[0])
                c1 = float(close_next_row[0])
                if c0 <= 0.0:
                    n_missing += 1
                    continue
                realized_bps = 10_000.0 * (c1 - c0) / c0
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE phantom_gateway_actions
                           SET realized_return_bps = %s,
                               resolved_at = now()
                         WHERE symbol = %s AND ts = %s
                        """,
                        (realized_bps, sym, ts),
                    )
                conn.commit()
                n_resolved += 1
            except Exception as row_exc:  # noqa: BLE001
                n_errored += 1
                log.warning("phantom_resolve_row_failed", symbol=sym, ts=str(ts), error=str(row_exc))
                try:
                    conn.rollback()
                except Exception:  # noqa: BLE001
                    pass

    log.info(
        "phantom_resolution_batch",
        scanned=len(rows),
        resolved=n_resolved,
        missing_bars=n_missing,
        errored=n_errored,
    )
    return n_resolved
