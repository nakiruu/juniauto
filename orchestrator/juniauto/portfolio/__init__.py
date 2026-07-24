"""Portfolio-level weighting and rebalance logic.

Implements the plan in docs/knowledge-base/part4-gateway-execution.md §2.28-2.30:
Kelly-fractional per-name sizing capped at max_name_weight, with the aggregate
constraint that Σw ≤ (1 − cash_floor). Cold-start (all edges indistinguishable)
falls back to equal weight across the eligible set — coherent, cap-bounded,
spec-compliant.
"""
from juniauto.portfolio.weighting import (
    Candidate,
    WeightingResult,
    compute_target_weights,
    edges_cv,
    fixed_equal_weights,
    select_top_k,
)

__all__ = [
    "Candidate",
    "WeightingResult",
    "compute_target_weights",
    "fixed_equal_weights",
    "edges_cv",
    "select_top_k",
]
