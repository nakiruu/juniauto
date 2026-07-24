"""Bayesian training and resolution module for JuniAuto.

Public surface:
    BayesianModel             -- wrapper around qe.GroupedRidge for online training + prediction.
    resolve_stale_executions  -- populates realized_return_bps for settled executions.
    resolve_stale_phantoms    -- populates realized_return_bps for phantom-cycle rows.
    backfill_from_bars        -- bootstrap synthetic training data from historical daily bars.
"""
from juniauto.bayesian.backfill import backfill_from_bars
from juniauto.bayesian.phantom_resolution import resolve_stale_phantoms
from juniauto.bayesian.resolution import resolve_stale_executions
from juniauto.bayesian.training import BayesianModel

__all__ = [
    "BayesianModel",
    "resolve_stale_executions",
    "resolve_stale_phantoms",
    "backfill_from_bars",
]
