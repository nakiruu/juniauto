"""yfinance supplemental feed for fundamentals.

Never used for prices/quotes (Alpaca IEX is the price feed). yfinance is only for:
    - trailing PE, PEG, EPS, revenue growth, ROE, margins, debt/equity (§1.4.2)
    - analyst revision direction, institutional ownership
    - earnings calendar (next report date; input to gap_days_to_next_trading_session)

Behaviour under rate limits:
    - Yahoo aggressively 429s bulk requests. This module:
        1. skips known ETFs (they don't expose fundamentals worth having anyway),
        2. read-through disk-caches each symbol for TTL_DAYS,
        3. rate-limits per-symbol calls with a 250 ms delay,
        4. retries per-symbol with exponential backoff (2s, 4s, 8s) on any exception
           or empty-info response, and
        5. falls back to any prior on-disk cache (even if TTL-stale) rather than
           returning None for a symbol that ran out of retries.
    - A single symbol's failure never kills the batch — downstream data_quality
      weights already handle NaN fundamentals per §1.7.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yfinance as yf
from curl_cffi import requests as cffi_requests
from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from juniauto.utils import get_logger


# Substrings that indicate a transient rate-limit / connectivity issue worth
# retrying. Anything else (parse bugs, missing symbols, auth failures) fails
# fast — retrying a code bug wastes ~14 seconds per symbol on backoff.
_RETRYABLE_HINTS: tuple[str, ...] = (
    "429",
    "too many",
    "rate limit",
    "timed out",
    "timeout",
    "connection reset",
    "connection refused",
    "temporarily unavailable",
    "empty info",  # our own soft-429 sentinel
)


def _is_retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(hint in msg for hint in _RETRYABLE_HINTS)

log = get_logger(__name__)

CACHE_DIR = Path("/app/cache/yahoo")

# yfinance returns almost no useful fundamentals for these; skip to save API calls.
# Sector / broad-market ETFs typically have trailingPE/marketCap only, and even
# those are unreliable. The signal families already tolerate missing fundamentals.
_KNOWN_ETFS: frozenset[str] = frozenset({
    # broad market
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "VOOG", "VOOV", "VUG", "VTV",
    # sector SPDRs
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC",
    # thematic / other common
    "ARKK", "TQQQ", "SQQQ", "SOXX", "SMH", "GLD", "SLV", "TLT", "IEF", "HYG",
    "LQD", "AGG", "BND", "VXUS", "EFA", "EEM", "IEMG", "IVV",
})


@dataclass(frozen=True, slots=True)
class Fundamentals:
    symbol: str
    fetched_at: str
    market_cap: float | None
    trailing_pe: float | None
    forward_pe: float | None
    peg_ratio: float | None
    price_to_book: float | None
    return_on_equity: float | None
    profit_margins: float | None
    gross_margins: float | None
    revenue_growth: float | None
    earnings_growth: float | None
    debt_to_equity: float | None
    quick_ratio: float | None
    beta: float | None
    dividend_yield: float | None
    next_earnings_date: str | None


def _empty_fundamentals(symbol: str) -> Fundamentals:
    """Placeholder used for ETFs and hard-failed fetches. All numeric fields None
    → downstream §1.7 data_quality weight naturally drops the row's confidence."""
    return Fundamentals(
        symbol=symbol,
        fetched_at=datetime.utcnow().isoformat(),
        market_cap=None,
        trailing_pe=None,
        forward_pe=None,
        peg_ratio=None,
        price_to_book=None,
        return_on_equity=None,
        profit_margins=None,
        gross_margins=None,
        revenue_growth=None,
        earnings_growth=None,
        debt_to_equity=None,
        quick_ratio=None,
        beta=None,
        dividend_yield=None,
        next_earnings_date=None,
    )


class _EmptyInfoError(RuntimeError):
    """yfinance returned an empty .info dict — usually a silent 429 or lookup miss."""


class YahooFeed:
    """Read-through disk cache + parallel fetches with hard timeout + browser
    impersonation.

    - `enabled=False`: short-circuits to empty Fundamentals for every symbol.
      The pipeline never blocks on yfinance. Use this when yfinance is flaky
      in your environment (it usually is).
    - `enabled=True`: parallel fetches with `max_workers` threads. Each future
      is hard-capped at `per_symbol_timeout_seconds` so no single symbol can
      hang the batch. On timeout or any non-retryable exception, falls back
      to stale on-disk cache, then to empty Fundamentals.

    Uses a persistent curl_cffi session mimicking Chrome's TLS + HTTP/2
    fingerprint. curl_cffi Session is thread-safe as of 0.6+, so sharing
    across workers is safe.
    """

    def __init__(
        self,
        ttl_days: int = 20,
        enabled: bool = False,
        max_workers: int = 4,
        per_symbol_timeout_seconds: float = 8.0,
    ) -> None:
        self._ttl = timedelta(days=ttl_days)
        self._enabled = enabled
        self._max_workers = max(1, int(max_workers))
        self._timeout_s = float(per_symbol_timeout_seconds)
        self._session = cffi_requests.Session(impersonate="chrome", timeout=int(self._timeout_s))
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            log.warning("yahoo_cache_mkdir_failed", error=str(e))

    def get_fundamentals(self, symbols: list[str]) -> dict[str, Fundamentals]:
        if not self._enabled:
            log.info("yahoo_disabled_skip_all", n=len(symbols))
            return {s: _empty_fundamentals(s) for s in symbols}

        out: dict[str, Fundamentals] = {}
        cold: list[str] = []
        n_etf = 0
        for sym in symbols:
            if sym.upper() in _KNOWN_ETFS:
                out[sym] = _empty_fundamentals(sym)
                n_etf += 1
                continue
            cached = self._load_cache(sym, allow_stale=False)
            if cached is not None:
                out[sym] = cached
            else:
                cold.append(sym)

        if n_etf:
            log.info("yahoo_skip_etfs", n=n_etf)

        if not cold:
            log.info("yahoo_cache_hit_all", n=len(out))
            return out

        log.info(
            "yahoo_fetch_start",
            n_cold=len(cold),
            workers=self._max_workers,
            timeout_s=self._timeout_s,
        )
        # Submit all fetches to the pool; collect with per-future timeout so a
        # single hung request cannot stall the batch.
        with ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="yahoo") as pool:
            future_to_sym = {pool.submit(self._fetch_one, sym): sym for sym in cold}
            for fut in as_completed(future_to_sym, timeout=self._timeout_s * len(cold) + 5):
                sym = future_to_sym[fut]
                try:
                    f = fut.result(timeout=self._timeout_s)
                    out[sym] = f
                    self._save_cache(sym, f)
                except (FuturesTimeout, RetryError, Exception) as e:  # noqa: BLE001
                    stale = self._load_cache(sym, allow_stale=True)
                    if stale is not None:
                        log.warning("yahoo_fetch_failed_use_stale", symbol=sym, error=str(e))
                        out[sym] = stale
                    else:
                        log.warning("yahoo_fetch_failed_no_cache", symbol=sym, error=str(e))
                        out[sym] = _empty_fundamentals(sym)
        return out

    # ---- Single-symbol fetch with retry ----
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=8),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    def _fetch_one(self, sym: str) -> Fundamentals:
        ticker = yf.Ticker(sym, session=self._session)
        try:
            info = ticker.info or {}
        except Exception as e:
            raise _EmptyInfoError(f"info raise: {e}") from e
        # Yahoo often returns an ~empty dict on soft-429 without raising.
        # Treat missing marketCap AND missing trailingPE as an empty response.
        if not info or (info.get("marketCap") is None and info.get("trailingPE") is None):
            raise _EmptyInfoError(f"empty info for {sym}")
        calendar_next = self._extract_next_earnings(ticker)
        return Fundamentals(
            symbol=sym,
            fetched_at=datetime.utcnow().isoformat(),
            market_cap=self._num(info.get("marketCap")),
            trailing_pe=self._num(info.get("trailingPE")),
            forward_pe=self._num(info.get("forwardPE")),
            peg_ratio=self._num(info.get("pegRatio")),
            price_to_book=self._num(info.get("priceToBook")),
            return_on_equity=self._num(info.get("returnOnEquity")),
            profit_margins=self._num(info.get("profitMargins")),
            gross_margins=self._num(info.get("grossMargins")),
            revenue_growth=self._num(info.get("revenueGrowth")),
            earnings_growth=self._num(info.get("earningsGrowth")),
            debt_to_equity=self._num(info.get("debtToEquity")),
            quick_ratio=self._num(info.get("quickRatio")),
            beta=self._num(info.get("beta")),
            dividend_yield=self._num(info.get("dividendYield")),
            next_earnings_date=calendar_next.isoformat() if calendar_next else None,
        )

    # ---- Cache ----
    def _load_cache(self, sym: str, *, allow_stale: bool) -> Fundamentals | None:
        p = CACHE_DIR / f"{sym}.json"
        if not p.exists():
            return None
        try:
            raw = json.loads(p.read_text())
            fetched = datetime.fromisoformat(raw["fetched_at"])
            if not allow_stale and datetime.utcnow() - fetched > self._ttl:
                return None
            return Fundamentals(**raw)
        except Exception:
            return None

    def _save_cache(self, sym: str, f: Fundamentals) -> None:
        p = CACHE_DIR / f"{sym}.json"
        try:
            p.write_text(json.dumps(asdict(f)))
        except (PermissionError, OSError) as e:
            log.warning("yahoo_cache_write_failed", symbol=sym, error=str(e))

    # ---- Helpers ----
    @staticmethod
    def _num(x: Any) -> float | None:
        try:
            v = float(x)
            return v if v == v else None  # filter NaN
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_next_earnings(ticker: yf.Ticker) -> date | None:
        try:
            cal = ticker.calendar
            if cal is None:
                return None
            if hasattr(cal, "loc"):
                val = cal.loc["Earnings Date"].iloc[0] if "Earnings Date" in cal.index else None  # type: ignore[union-attr]
            else:
                val = cal.get("Earnings Date")
                if isinstance(val, list):
                    val = val[0] if val else None
            if val is None:
                return None
            if hasattr(val, "date"):
                return val.date()
            return date.fromisoformat(str(val)[:10])
        except Exception:
            return None


# Silence a possibly-unused-import warning while keeping `replace` handy for
# potential future partial-cache-refresh work.
_ = replace
