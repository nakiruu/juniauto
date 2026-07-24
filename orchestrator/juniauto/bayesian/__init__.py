"""Bayesian training and resolution module for JuniAuto.

Public surface:
    BayesianModel          -- wrapper around qe.GroupedRidge for online training + prediction.
    resolve_stale_executions -- populates realized_return_bps for settled executions.
"""
from juniauto.bayesian.resolution import resolve_stale_executions
from juniauto.bayesian.training import BayesianModel

__all__ = ["BayesianModel", "resolve_stale_executions"]
