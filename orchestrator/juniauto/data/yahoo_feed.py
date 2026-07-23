"""yfinance supplemental feed for fundamentals.

Never used for prices/quotes (Alpaca IEX is the price feed). yfinance is only for:
    - trailing PE, PEG, EPS, revenue growth, ROE, margins, debt/equity (§1.4.2)
    - analyst revision direction, institutional ownership
    - earnings calendar (next report date; input to gap_days_to_next_trading_session)

Cached on disk with a TTL matched to the fundamental halflife (§1.3: ~20 trading days).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yfinance as yf


CACHE_DIR = Path("/app/cache/yahoo")


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


class YahooFeed:
    """Read-through disk cache; single-call fanout is fine because yfinance
    already batches under the hood via `yf.Tickers`.
    """

    def __init__(self, ttl_days: int = 20) -> None:
        self._ttl = timedelta(days=ttl_days)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def get_fundamentals(self, symbols: list[str]) -> dict[str, Fundamentals]:
        out: dict[str, Fundamentals] = {}
        cold: list[str] = []
        for sym in symbols:
            cached = self._load_cache(sym)
            if cached is not None:
                out[sym] = cached
            else:
                cold.append(sym)
        if not cold:
            return out

        # yfinance batches metadata via .info per ticker; we accept the N calls.
        tickers = yf.Tickers(" ".join(cold))
        for sym in cold:
            try:
                info = tickers.tickers[sym].info or {}
            except Exception:
                info = {}
            calendar_next = self._extract_next_earnings(tickers.tickers[sym])
            f = Fundamentals(
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
            out[sym] = f
            self._save_cache(sym, f)
        return out

    # ---- Internals ----
    def _load_cache(self, sym: str) -> Fundamentals | None:
        p = CACHE_DIR / f"{sym}.json"
        if not p.exists():
            return None
        try:
            raw = json.loads(p.read_text())
            fetched = datetime.fromisoformat(raw["fetched_at"])
            if datetime.utcnow() - fetched > self._ttl:
                return None
            return Fundamentals(**raw)
        except Exception:
            return None

    def _save_cache(self, sym: str, f: Fundamentals) -> None:
        p = CACHE_DIR / f"{sym}.json"
        p.write_text(json.dumps(asdict(f)))

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
            # yfinance calendar shape shifts across versions; probe both DataFrame + dict
            if hasattr(cal, "loc"):
                val = cal.loc["Earnings Date"].iloc[0] if "Earnings Date" in cal.index else None  # type: ignore[union-attr]
            else:
                val = cal.get("Earnings Date")
                if isinstance(val, list):
                    val = val[0] if val else None
            if val is None:
                return None
            if hasattr(val, "date"):
                return val.date()  # pandas Timestamp
            return date.fromisoformat(str(val)[:10])
        except Exception:
            return None
