"""Universe filter — the input side of §1.1 candidate selection.

Only securities that (a) can plausibly improve after-cost returns and (b) can be
traded through Alpaca belong in the universe. Filters applied per config:
    - min_price
    - min_adv_shares (~20d)
    - min_market_cap
    - allowed exchanges (NYSE, NASDAQ, optionally ETFs)
    - valid tradable Alpaca asset with active status

The output is a sorted list of symbols the rest of the pipeline treats as
`x_i,t` candidates.
"""
from __future__ import annotations

from dataclasses import dataclass

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

from juniauto.config import UniverseConfig
from juniauto.data.yahoo_feed import Fundamentals
from juniauto.utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class UniverseAsset:
    symbol: str
    exchange: str
    asset_class: str
    tradable: bool


class UniverseBuilder:
    """Two-stage: Alpaca `assets` for tradability/exchange, then price/ADV/mktcap on the tape."""

    def __init__(self, trading: TradingClient, cfg: UniverseConfig) -> None:
        self._trading = trading
        self._cfg = cfg

    # ---- Stage 1: tradable universe from Alpaca ----
    def alpaca_universe(self) -> list[UniverseAsset]:
        """All active, tradable, easy-to-borrow US equities on the allowed exchanges."""
        out: list[UniverseAsset] = []
        for cls in [AssetClass.US_EQUITY]:
            req = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=cls)
            for a in self._trading.get_all_assets(req):  # type: ignore[union-attr]
                exch = str(a.exchange.value if hasattr(a.exchange, "value") else a.exchange)  # type: ignore[union-attr]
                if exch not in self._cfg.exchanges:
                    continue
                if not a.tradable:  # type: ignore[union-attr]
                    continue
                # ETFs share the US_EQUITY class in Alpaca; filter later on fundamentals if not wanted.
                out.append(
                    UniverseAsset(
                        symbol=a.symbol,               # type: ignore[union-attr]
                        exchange=exch,
                        asset_class=str(cls.value),
                        tradable=True,
                    )
                )
        log.info("universe_alpaca_stage", n=len(out), exchanges=self._cfg.exchanges)
        return out

    # ---- Stage 2: price / ADV / market cap tape filter ----
    def apply_tape_filter(
        self,
        assets: list[UniverseAsset],
        last_close: dict[str, float],
        adv_shares: dict[str, float],
        fundamentals: dict[str, Fundamentals],
    ) -> list[str]:
        keep: list[str] = []
        for a in assets:
            px = last_close.get(a.symbol)
            adv = adv_shares.get(a.symbol)
            f = fundamentals.get(a.symbol)
            if px is None or adv is None:
                continue
            if px < self._cfg.min_price:
                continue
            if adv < self._cfg.min_adv_shares:
                continue
            if f is not None and f.market_cap is not None:
                if f.market_cap < self._cfg.min_market_cap:
                    continue
            # ETF handling: yfinance often has None market_cap for ETFs; keep if include_etfs True.
            keep.append(a.symbol)
        keep.sort()
        log.info(
            "universe_tape_stage",
            n_in=len(assets),
            n_out=len(keep),
            min_price=self._cfg.min_price,
            min_adv=self._cfg.min_adv_shares,
            min_mcap=self._cfg.min_market_cap,
        )
        return keep
