"""Semantic and contextual signal family computer.

This module is a structured placeholder. Per spec §1.4.4, missing or stale
context data reduces confidence rather than contributing invented edge:

    "Missing / generic / stale context reduces confidence rather than
     inventing edge."
    — docs/knowledge-base/part1-data-preparation.md §1.4.4

ARCHITECTURE NOTE:
    This is where sector/theme classification will slot in when a semantic
    data source (e.g. a news-embedding similarity service, sector ETF momentum,
    or LLM-generated context alignment score) is integrated. The interface is
    intentionally narrow: a context_map dict[str, float] keyed by symbol with
    a pre-scored alignment value, and a sector_map dict[str, float] for
    sector-level context.

    Until those sources are available, all symbols return:
        context_alignment = 0.0   (neutral, not NaN — stale context is
                                   explicitly "no information", not "missing data")
        sector_context    = 0.0

    The distinction from NaN is intentional: NaN propagates the missing-data
    weight penalty (§1.7); zero is the spec-mandated neutral prior for context
    (§1.4.4 "missing context → 0, not invented edge").

Output columns match `features` table §1.4.4:
    context_alignment, sector_context.
See docs/knowledge-base/part1-data-preparation.md §1.4.4.
"""
from __future__ import annotations

import pandas as pd

from juniauto.utils import get_logger

log = get_logger(__name__)


class SemanticSignals:
    """Compute semantic/contextual signal family.

    Current implementation returns zero for all symbols as the semantic data
    pipeline is not yet integrated. Zero is the spec-mandated neutral value
    for missing context (§1.4.4), distinct from NaN (which would trigger
    a data-quality penalty via the observation weight w_n in §1.7).

    Future slot:
        - context_alignment: cosine similarity of company news embedding to
          current market theme vector, freshness-weighted.
        - sector_context: sector ETF momentum or sector-relative strength,
          mapped per symbol via GICS or custom classification.
    """

    def compute(
        self,
        symbols: list[str],
        context_map: dict[str, float] | None = None,
        sector_map: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Compute semantic features for each symbol.

        Args:
            symbols: All candidate symbols for this decision cycle.
            context_map: Optional pre-scored context alignment values per symbol.
                If None or a symbol is absent, uses 0.0 (spec §1.4.4 default).
            sector_map: Optional pre-scored sector context values per symbol.
                If None or a symbol is absent, uses 0.0 (spec §1.4.4 default).

        Returns:
            DataFrame indexed by symbol with columns:
            context_alignment, sector_context.
            All values are 0.0 when context data is unavailable (§1.4.4).
        """
        if not symbols:
            return pd.DataFrame()

        ctx = context_map or {}
        sec = sector_map or {}

        rows = [
            {
                "symbol": sym,
                # spec §1.4.4: missing context → 0, not invented edge
                "context_alignment": float(ctx.get(sym, 0.0)),
                "sector_context": float(sec.get(sym, 0.0)),
            }
            for sym in symbols
        ]

        out = pd.DataFrame(rows).set_index("symbol")
        # No cross-sectional normalization applied here: zero values from
        # missing context must remain zero, not be MAD-scaled.
        # When real scores are provided via context_map/sector_map, the caller
        # should pre-normalize them or add a normalization pass here.

        log.debug(
            "semantic_signals_computed",
            n_symbols=len(out),
            n_with_context=sum(1 for s in symbols if s in ctx),
        )
        return out
