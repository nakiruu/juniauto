"""Point-in-time snapshot assembler for backtest.

Given a target timestamp (the backtest "now") and a symbol list, returns
a MarketSnapshot shaped IDENTICALLY to what DataAggregator.snapshot()
returns during live trading. The engine can then call the shipped
`signals.compute_all(bars=snap.bars_df(), ...)` without any wrapping.

Data-source policy (coordinator design review Q2 default):
    - Bars come from QuestDB `bars` table, filtered to
      `ts >= now - N_days AND ts < now` (strict <, so today's bar is
      NOT included at decision time — matches live-cycle semantics
      where the 15:55 tick sees only bars that have printed before it).
    - Fundamentals come from the live YahooFeed cache directly — this
      is the "current-snapshot" approximation. A future --strict-pit
      flag would gate on cache_created_at <= now.
    - Quotes are set to None / empty because we don't have historical
      IEX quotes at cycle-cadence resolution; the C++ gateway's
      wide_spread guard is bypassed accordingly (see engine).

Transport: preload_all_bars uses QuestDB's REST /exp (CSV export)
endpoint, NOT the PG wire. The PG wire has been unreliable under load
(drops connections mid-response, malformed error frames, "fd already
closed" internal errors) while REST is rock solid — every diagnostic
curl during debugging succeeded. Since backtest preload is a single
one-shot bulk read, REST is the right choice.
"""
from __future__ import annotations

import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from io import StringIO

import pandas as pd

from juniauto.data.aggregator import MarketSnapshot
from juniauto.data.alpaca_feed import Bar
from juniauto.data.yahoo_feed import Fundamentals, YahooFeed
from juniauto.db import QuestDBClient
from juniauto.utils import get_logger
from juniauto.utils.time_utils import ET

log = get_logger(__name__)


class HistoricalSnapshotLoader:
    """Build MarketSnapshots from QuestDB `bars` + YahooFeed cache.

    Backtest lives-or-dies path: we preload ALL bars into memory at engine
    init via `preload_all_bars(...)` and filter per-cycle from that in-memory
    DataFrame. QuestDB's PG wire is unreliable under sustained per-cycle
    range queries (drops connections mid-response, malformed error frames,
    "fd already closed" internal errors), so ONE big query + in-memory
    filtering is dramatically more reliable AND faster than N smaller
    queries. For a 4.6y × 300-symbol backtest that's 1 query instead of
    ~1144 — and the resulting DataFrame is ~50 MB, negligible in RAM.
    """

    def __init__(
        self,
        db: QuestDBClient,
        yahoo: YahooFeed | None,
        history_bars: int = 252,
    ) -> None:
        self._db = db
        self._yahoo = yahoo
        self._history_bars = int(history_bars)
        # Warm-cache of (symbol, date) -> Bar for O(1) fill lookups from SimBroker.
        # Populated lazily as `snapshot` runs so subsequent settle() calls hit
        # the cache instead of round-tripping to QuestDB.
        self._bar_cache: dict[tuple[str, date], Bar] = {}
        # In-memory bar store — populated by preload_all_bars(). While None,
        # falls back to per-snapshot DB queries (only used if the engine
        # forgets to preload — for safety, not for correctness).
        self._all_bars_df: pd.DataFrame | None = None
        self._bars_by_symbol: dict[str, list[Bar]] | None = None

    # ---- Preload ----
    def preload_all_bars(
        self,
        symbols: list[str],
        earliest: datetime,
        latest: datetime,
    ) -> int:
        """Load ALL bars in [earliest, latest] for `symbols` into memory in
        ONE request via QuestDB's REST /exp (CSV export) endpoint.

        Rationale: QuestDB's PG wire drops even single large queries
        mid-response with "server closed the connection unexpectedly" —
        NOT a QuestDB crash (health is fine, memory is fine, storage is
        fine), a psycopg-side parse failure on QuestDB's response frames.
        The REST /exp endpoint returns clean CSV without going through
        the PG wire and has been 100% reliable across all diagnostics.
        For a 4.6y × 300-symbol window (~450k rows) the CSV is ~40 MB
        and downloads in ~2-3 seconds.

        Returns the number of bar rows loaded. If 0, the backtest will
        silently produce empty cycles — check the log for the specific
        REST-response error.
        """
        # Use ISO timestamps in the SQL string; QuestDB's REST endpoint
        # doesn't support parameter substitution, but timestamp string
        # literals work fine (and are unambiguous).
        sql = (
            "SELECT symbol, ts, open, high, low, close, volume, "
            "COALESCE(vwap, 0.0) AS vwap, COALESCE(trade_count, 0) AS trade_count "
            "FROM bars "
            f"WHERE ts >= '{earliest.isoformat()}' AND ts <= '{latest.isoformat()}' "
            "ORDER BY symbol ASC, ts ASC"
        )
        try:
            df = self._rest_query_df(sql)
        except Exception as e:  # noqa: BLE001
            log.error("preload_bars_rest_failed", error=str(e), error_type=type(e).__name__)
            self._all_bars_df = pd.DataFrame()
            self._bars_by_symbol = {}
            return 0

        if df.empty:
            log.warning("preload_bars_empty", earliest=str(earliest), latest=str(latest))
            self._all_bars_df = pd.DataFrame()
            self._bars_by_symbol = {}
            return 0

        # Filter to requested universe.
        requested = set(symbols)
        df = df[df["symbol"].isin(requested)].reset_index(drop=True)
        if df.empty:
            log.warning("preload_bars_none_in_universe",
                        n_universe=len(requested), n_bars_before_filter=len(df))
            self._all_bars_df = pd.DataFrame()
            self._bars_by_symbol = {}
            return 0

        # Parse timestamps once, add helper _date column for cheap filtering.
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(ET)
        df["_date"] = df["ts"].dt.date

        # Build symbol -> [Bar] dict (matches aggregator's shape).
        by_sym: dict[str, list[Bar]] = {}
        for row in df.itertuples(index=False):
            ts_dt = row.ts.to_pydatetime()
            b = Bar(
                symbol=str(row.symbol), ts=ts_dt,
                open=float(row.open), high=float(row.high), low=float(row.low), close=float(row.close),
                volume=int(row.volume) if pd.notna(row.volume) else 0,
                vwap=float(row.vwap) if pd.notna(row.vwap) and float(row.vwap) > 0 else None,
                trade_count=int(row.trade_count) if pd.notna(row.trade_count) else None,
            )
            by_sym.setdefault(str(row.symbol), []).append(b)
            self._bar_cache[(str(row.symbol), ts_dt.date())] = b

        self._bars_by_symbol = by_sym
        self._all_bars_df = df
        n = len(df)
        log.info(
            "preload_bars_complete",
            n_rows=n, n_symbols=len(by_sym),
            earliest=str(earliest.date()),
            latest=str(latest.date()),
        )
        return n

    # ---- REST transport ----
    def _rest_query_df(self, sql: str, timeout: float = 120.0) -> pd.DataFrame:
        """Hit QuestDB's REST /exp (CSV export) endpoint and return a
        DataFrame. Bypasses psycopg + PG wire entirely.
        `_db._cfg.host` is the container-network hostname (e.g. `questdb`);
        REST port is always 9000 (container-internal, per docker-compose).
        """
        cfg = self._db._cfg  # noqa: SLF001 — deliberate; DB client hides transport
        url = f"http://{cfg.host}:9000/exp?" + urllib.parse.urlencode({"query": sql})
        req = urllib.request.Request(url, headers={"User-Agent": "juniauto-backtest/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8")
        return pd.read_csv(StringIO(payload))

    # ---- Public API (aggregator-shaped) ----
    def snapshot(self, symbols: list[str], now: datetime) -> MarketSnapshot:
        """Return a snapshot with bars up to (but not including) `now`.

        `now` is timezone-aware ET. The lookback is
        `now - history_bars * 1.5 days` to survive weekends + holidays
        the same way the live AlpacaFeed.get_bars does.
        """
        window_start = now - timedelta(days=int(self._history_bars * 1.5))
        # Prefer the in-memory preload if it's populated. Falls back to a
        # per-snapshot DB query only if the engine forgot to preload.
        if self._bars_by_symbol is not None:
            bars = self._slice_preloaded(symbols, window_start, now)
        else:
            bars = self._load_bars(symbols, window_start, now)
        fundamentals = self._load_fundamentals(symbols)
        snap = MarketSnapshot(
            ts=now,
            bars=bars,
            quotes={},   # deliberately empty; see docstring
            fundamentals=fundamentals,
        )
        # Warm the bar cache with EVERY bar we just fetched — the SimBroker
        # will need same-day-close and next-day-open lookups, and hitting
        # QuestDB per fill would 10-100× the backtest wall-time.
        for sym, series in bars.items():
            for b in series:
                self._bar_cache[(sym, b.ts.date() if isinstance(b.ts, datetime) else b.ts)] = b
        return snap

    def next_bar(self, symbol: str, target_date: date) -> Bar | None:
        """Return the bar for `(symbol, target_date)` — cache-first, then
        DB fallback. Used by SimBroker as its bars_provider."""
        hit = self._bar_cache.get((symbol, target_date))
        if hit is not None:
            return hit
        # Cache miss: fetch a small window (target_date to +7d) and cache.
        rows = self._db.query(
            """
            SELECT ts, open, high, low, close, volume, vwap, COALESCE(trade_count, 0)
              FROM bars
             WHERE symbol = %s
               AND ts::date >= %s::date
               AND ts::date <= %s::date
             ORDER BY ts ASC
            """,
            (symbol, target_date, target_date + timedelta(days=7)),
        )
        for ts, o, h, l, c, v, vw, tc in rows:
            ts_dt = ts if isinstance(ts, datetime) else datetime.combine(ts, datetime.min.time(), tzinfo=ET)
            bar = Bar(
                symbol=symbol,
                ts=ts_dt,
                open=float(o), high=float(h), low=float(l), close=float(c),
                volume=int(v or 0),
                vwap=float(vw) if vw is not None else None,
                trade_count=int(tc) if tc is not None else None,
            )
            self._bar_cache[(symbol, ts_dt.date())] = bar
        return self._bar_cache.get((symbol, target_date))

    def clear_bar_cache(self) -> None:
        self._bar_cache.clear()

    def _slice_preloaded(
        self, symbols: list[str], window_start: datetime, now: datetime
    ) -> dict[str, list[Bar]]:
        """O(n) slice of the preloaded in-memory bar store."""
        if not self._bars_by_symbol:
            return {s: [] for s in symbols}
        requested = set(symbols)
        out: dict[str, list[Bar]] = {s: [] for s in symbols}
        for sym, series in self._bars_by_symbol.items():
            if sym not in requested:
                continue
            # Bars are stored sorted by ts per symbol from preload_all_bars.
            out[sym] = [b for b in series if window_start <= b.ts < now]
        return out

    # ---- Internals ----
    def _load_bars(
        self,
        symbols: list[str],
        window_start: datetime,
        now: datetime,
    ) -> dict[str, list[Bar]]:
        if not symbols:
            return {}
        # NO `symbol IN (...)` clause. With ~300 symbols the interpolated
        # IN list produces a 5-6KB SQL statement that QuestDB's PG wire
        # sometimes returns a malformed error response for, which psycopg
        # then reports as "server closed the connection unexpectedly" for
        # every cycle. The bars table only contains symbols the live
        # pipeline / backfill wrote — themselves derived from
        # config.universe.symbols — so filtering by time window is
        # sufficient and correct; we apply an in-memory set filter after
        # for defensive isolation.
        sql = """
            SELECT symbol, ts, open, high, low, close, volume, vwap, COALESCE(trade_count, 0)
              FROM bars
             WHERE ts >= %s
               AND ts < %s
             ORDER BY symbol ASC, ts ASC
        """
        rows = self._db.query(sql, (window_start, now))
        requested = set(symbols)
        out: dict[str, list[Bar]] = {s: [] for s in symbols}
        for sym, ts, o, h, l, c, v, vw, tc in rows:
            if sym not in requested:
                continue
            ts_dt = ts if isinstance(ts, datetime) else datetime.combine(ts, datetime.min.time(), tzinfo=ET)
            out.setdefault(sym, []).append(Bar(
                symbol=sym,
                ts=ts_dt,
                open=float(o), high=float(h), low=float(l), close=float(c),
                volume=int(v or 0),
                vwap=float(vw) if vw is not None else None,
                trade_count=int(tc) if tc is not None else None,
            ))
        return out

    def _load_fundamentals(self, symbols: list[str]) -> dict[str, Fundamentals]:
        """Current-snapshot fundamentals (coordinator Q2 default). If the
        YahooFeed is disabled (config.yahoo.enabled=false) or unavailable,
        return empty — the Bayesian and cost model handle missing
        fundamentals gracefully."""
        if self._yahoo is None:
            return {}
        try:
            return self._yahoo.get_fundamentals(symbols)
        except Exception as e:  # noqa: BLE001
            log.warning("historical_fundamentals_failed", error=str(e))
            return {}


def _quote_sym(s: str) -> str:
    # Symbols may contain "." (e.g., BRK.B). Wrap in single quotes with backslash
    # escaping — QuestDB supports standard SQL string escaping.
    esc = s.replace("'", "''")
    return f"'{esc}'"
