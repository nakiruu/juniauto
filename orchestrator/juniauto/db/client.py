"""QuestDB client — Postgres wire for reads, ILP TCP for hot-path writes."""
from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import psycopg
from questdb.ingress import Sender, TimestampNanos

from juniauto.config import DatabaseConfig


# Strip SQL comments before splitting on ';' — a comment like
# "-- All tables partition by day; SYMBOL columns are ..." contains a semicolon
# that a naive split would treat as a statement terminator.
_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


class QuestDBClient:
    """Dual-transport client.

    Reads use the Postgres wire (port 8812) via psycopg — SQL, prepared statements, joins.
    Writes use ILP TCP (port 9009) via the official `questdb` sender — batched, non-blocking.
    """

    def __init__(self, cfg: DatabaseConfig) -> None:
        self._cfg = cfg
        self._pg_conninfo = (
            f"host={cfg.host} port={cfg.port} user={cfg.user} "
            f"password={cfg.password} dbname={cfg.name} sslmode=disable"
        )
        # ILP shares the same host as the Postgres wire; different port.
        self._ilp_conf = f"tcp::addr={cfg.host}:9009;auto_flush_rows=1000;auto_flush_interval=1000;"

    # ---- Schema ----
    def apply_schema(self, schema_path: Path | str) -> None:
        raw = Path(schema_path).read_text(encoding="utf-8")
        # Order matters: block comments first (may span lines), then line comments.
        cleaned = _BLOCK_COMMENT.sub("", raw)
        cleaned = _LINE_COMMENT.sub("", cleaned)
        statements = [s.strip() for s in cleaned.split(";") if s.strip()]
        with self._pg_conn() as conn, conn.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)  # type: ignore[arg-type]
            conn.commit()

    # ---- Reads ----
    def query(self, sql: str, params: Sequence[Any] | None = None) -> list[tuple[Any, ...]]:
        with self._pg_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params or ())
            return list(cur.fetchall())

    def query_one(self, sql: str, params: Sequence[Any] | None = None) -> tuple[Any, ...] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # ---- Writes (ILP) ----
    @contextmanager
    def sender(self) -> Iterator[Sender]:
        """Batched ILP writer. Auto-flushes every 1000 rows or 1s, whichever first."""
        with Sender.from_conf(self._ilp_conf) as s:
            yield s

    @staticmethod
    def now_ns() -> TimestampNanos:
        return TimestampNanos.now()

    # ---- Internals ----
    @contextmanager
    def _pg_conn(self) -> Iterator[psycopg.Connection[tuple[Any, ...]]]:
        conn = psycopg.connect(self._pg_conninfo)
        try:
            yield conn
        finally:
            conn.close()
