"""Target-weight construction — Kelly-fractional with cold-start fallback.

Contract:
    compute_target_weights(candidates, ...) -> WeightingResult

    Given a list of Candidates with (symbol, conservative_edge_bps, sigma_total_bps),
    produce a `dict[symbol -> weight]` such that:
        - Σw ≤ (1 − cash_floor)   (aggregate cash reserve honoured)
        - 0 ≤ w_i ≤ max_name_weight (per-name hard cap)
        - w_i ∝ max(0, edge_i) / max(σ_i², σ_floor²)   (Kelly-scaled)
    with an equal-weight fallback when all edges are indistinguishable within
    `cold_start_cv_threshold`.

    Also exposes `fixed_equal_weights(...)` — the naive 5%/name baseline the
    shadow monitor compares against.

Design notes:
    - Long-only: negative edges get zero weight, not a short.
    - `σ_floor_bps` prevents division-by-tiny-vol when the Bayesian is uninformative
      (cold-start σ_total = 0 → Kelly = ∞ without a floor).
    - Water-filling cap enforcement: if a raw Kelly weight exceeds the per-name cap,
      set it to the cap and re-normalize the remaining names against the reduced
      budget. Iterates until no cap violations remain (bounded to 100 passes; in
      practice converges in ≤ N iterations).
    - Pure function of inputs → deterministic → replay-safe (§3.5).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class Candidate:
    symbol: str
    conservative_edge_bps: float
    sigma_total_bps: float


@dataclass(frozen=True, slots=True)
class WeightingResult:
    weights: dict[str, float]     # symbol -> target portfolio weight
    scheme: str                    # "kelly" | "equal_weight_fallback" | "all_cash"
    cash_weight: float             # 1 − Σweights
    n_at_name_cap: int             # how many names were capped at max_name_weight


# Sentinels chosen to avoid divide-by-tiny-vol on cold start. 100 bps ≈ 1%
# annualized vol as a plausible lower bound for the cheapest ETF; single-name
# stocks are typically ≥ 1500 bps, so this floor only ever binds during the
# uninformative-Bayesian regime.
DEFAULT_SIGMA_FLOOR_BPS = 100.0
DEFAULT_COLD_START_CV_THRESHOLD = 1e-6


def compute_target_weights(
    candidates: Iterable[Candidate],
    *,
    max_name_weight: float,
    cash_floor: float,
    sigma_floor_bps: float = DEFAULT_SIGMA_FLOOR_BPS,
    cold_start_cv_threshold: float = DEFAULT_COLD_START_CV_THRESHOLD,
) -> WeightingResult:
    cands = [c for c in candidates]  # materialize (may be a generator)
    if not cands:
        return WeightingResult(weights={}, scheme="all_cash", cash_weight=1.0, n_at_name_cap=0)

    if not (0.0 < max_name_weight <= 1.0):
        raise ValueError(f"max_name_weight must be in (0, 1], got {max_name_weight}")
    if not (0.0 <= cash_floor < 1.0):
        raise ValueError(f"cash_floor must be in [0, 1), got {cash_floor}")

    invested_budget = 1.0 - cash_floor  # e.g. 0.95

    edges = np.array([c.conservative_edge_bps for c in cands], dtype=float)
    sigmas = np.array([c.sigma_total_bps for c in cands], dtype=float)

    # ---- Cold-start detection ----
    # If the coefficient of variation across edges is below threshold, treat
    # the ranking as uninformative and equal-weight the eligible set.
    if _is_cold_start(edges, cold_start_cv_threshold):
        per_name = min(max_name_weight, invested_budget / len(cands))
        weights = {c.symbol: per_name for c in cands}
        cash_weight = 1.0 - per_name * len(cands)
        return WeightingResult(
            weights=weights,
            scheme="equal_weight_fallback",
            cash_weight=cash_weight,
            n_at_name_cap=(len(cands) if per_name == max_name_weight else 0),
        )

    # ---- Kelly-scaled raw weights ----
    # Score is proportional to μ_conservative / σ² with σ² floored. Negative
    # edges get zero (long-only). Absolute magnitude doesn't matter; we
    # normalize to the invested budget below.
    positive_edges = np.maximum(edges, 0.0)
    variances = np.maximum(sigmas * sigmas, sigma_floor_bps * sigma_floor_bps)
    raw_scores = positive_edges / variances

    if raw_scores.sum() <= 0.0:
        # No positive-edge candidates. Sit in cash rather than force allocation.
        return WeightingResult(weights={}, scheme="all_cash", cash_weight=1.0, n_at_name_cap=0)

    weights_arr, n_capped = _water_fill(
        raw_scores=raw_scores,
        budget=invested_budget,
        name_cap=max_name_weight,
    )

    weights = {c.symbol: float(weights_arr[i]) for i, c in enumerate(cands) if weights_arr[i] > 0.0}
    cash_weight = 1.0 - float(weights_arr.sum())
    return WeightingResult(
        weights=weights,
        scheme="kelly",
        cash_weight=cash_weight,
        n_at_name_cap=n_capped,
    )


def fixed_equal_weights(
    candidates: Iterable[Candidate],
    *,
    per_name_weight: float,
    max_name_weight: float,
    cash_floor: float,
) -> WeightingResult:
    """The MVP baseline strategy — fixed weight per name, floor-clamped.

    Used by the shadow monitor to compare against the Kelly scheme's realized
    performance. Same output shape as compute_target_weights() so downstream
    diffing is uniform.
    """
    cands = [c for c in candidates]
    if not cands:
        return WeightingResult(weights={}, scheme="all_cash", cash_weight=1.0, n_at_name_cap=0)

    w = min(per_name_weight, max_name_weight)
    invested_budget = 1.0 - cash_floor
    if w * len(cands) > invested_budget:
        w = invested_budget / len(cands)

    weights = {c.symbol: w for c in cands}
    cash_weight = 1.0 - w * len(cands)
    n_capped = sum(1 for _ in cands) if w == max_name_weight else 0
    return WeightingResult(
        weights=weights,
        scheme="fixed_equal_baseline",
        cash_weight=cash_weight,
        n_at_name_cap=n_capped,
    )


# ---------- Internals ----------
def _is_cold_start(edges: np.ndarray, threshold: float) -> bool:
    if len(edges) <= 1:
        return True
    mean = float(np.abs(edges).mean())
    std = float(edges.std())
    if mean == 0.0:
        return std < threshold  # all zero → cold
    return (std / mean) < threshold


def _water_fill(
    raw_scores: np.ndarray, budget: float, name_cap: float, max_iter: int = 100
) -> tuple[np.ndarray, int]:
    """Proportional allocation with iterative per-name cap enforcement.

    Returns (weights, n_capped). Weights sum to ≤ budget, each ≤ name_cap.
    """
    n = len(raw_scores)
    weights = np.zeros(n, dtype=float)
    capped = np.zeros(n, dtype=bool)
    scores = raw_scores.astype(float, copy=True)

    for _ in range(max_iter):
        active = ~capped
        active_score_sum = float(scores[active].sum())
        remaining_budget = budget - float(name_cap * capped.sum())
        if active_score_sum <= 0.0 or remaining_budget <= 0.0:
            break
        weights[active] = scores[active] * (remaining_budget / active_score_sum)
        # Any weight exceeding the cap gets pinned to the cap; repeat.
        newly_capped = active & (weights > name_cap)
        if not newly_capped.any():
            break
        weights[newly_capped] = name_cap
        capped |= newly_capped

    # Assign fixed-cap slots (already at name_cap).
    weights[capped] = name_cap
    return weights, int(capped.sum())
