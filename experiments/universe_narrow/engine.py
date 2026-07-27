"""UniverseNarrowBacktestEngine — subclass of BacktestEngine.

Overrides ONLY `_compute_predictions` to filter features by the
quarter's active universe before the parent runs. Everything else —
snapshot loader, gateway evaluation, sizing, order routing,
persistence, metrics — reused unchanged.
"""
from __future__ import annotations

from datetime import date

from experiments.universe_narrow.selector import QuarterlyUniverseSelector
from juniauto.backtest.engine import BacktestEngine
from juniauto.config import JuniAutoConfig
from juniauto.utils import get_logger

log = get_logger(__name__)


class UniverseNarrowBacktestEngine(BacktestEngine):
    """Backtest engine that filters candidates to a quarterly top-N universe.

    Universe selection is precomputed at __init__ (one-shot cost, ~1-2s)
    and O(1) per cycle. Base engine's preload_all_bars still loads the
    FULL 299-symbol bars so signal computation has cross-sectional
    context, but only top-N pass through _compute_predictions each cycle.
    """

    def __init__(
        self,
        cfg: JuniAutoConfig,
        *,
        top_n: int = 50,
        **kwargs,
    ) -> None:
        super().__init__(cfg, **kwargs)
        self.selector = QuarterlyUniverseSelector(
            self.db,
            start_date=self.start_date,
            end_date=self.end_date,
            top_n=top_n,
        )
        log.info("un_engine_ready", top_n=top_n, summary=self.selector.summary())

    def _compute_predictions(self, features):
        """Filter features to the current quarter's active universe, then
        delegate to the parent's prediction path. If active set is empty
        (before first quarter boundary or all bars filtered out), returns
        no predictions and the cycle exits gracefully with no orders.
        """
        # Extract cycle date from the engine's clock
        cycle_date: date = self.clock.now().date()
        active = self.selector.active_symbols(cycle_date)
        if not active:
            log.info("un_cycle_no_active_universe", ts=str(cycle_date))
            return []

        # Filter the features DataFrame to active-universe symbols only.
        # `features.index` is the symbol axis after compute_all.
        filtered = features.loc[features.index.isin(active)]
        if filtered.empty:
            log.info(
                "un_cycle_features_empty_after_filter",
                ts=str(cycle_date),
                n_active=len(active),
                n_features_before=len(features),
            )
            return []

        # Delegate — every downstream step (predictions, gateway,
        # sizing, routing) receives the filtered set only.
        return super()._compute_predictions(filtered)
