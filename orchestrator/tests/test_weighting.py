"""Unit tests for portfolio/weighting.py.

Covers the properties the plan promises:
    - Cold-start uniform edges → equal-weight fallback
    - Kelly scaling under differentiated edges
    - Per-name cap enforcement (water-filling)
    - Cash-floor invariant Σw ≤ 1 − cash_floor
    - Negative / zero edges get zero weight (long-only)
    - fixed_equal_weights baseline correctness
"""
from __future__ import annotations

import math

import pytest

from juniauto.portfolio import Candidate, compute_target_weights, fixed_equal_weights


# ---- fixtures ----
def _c(symbol: str, edge: float, sigma: float) -> Candidate:
    return Candidate(symbol=symbol, conservative_edge_bps=edge, sigma_total_bps=sigma)


# ---- cold-start fallback ----
def test_empty_candidates_all_cash() -> None:
    r = compute_target_weights([], max_name_weight=0.10, cash_floor=0.05)
    assert r.scheme == "all_cash"
    assert r.weights == {}
    assert r.cash_weight == 1.0


def test_cold_start_uniform_edges_equal_weights() -> None:
    """Today's live case: composite_edge is 138 bps flat across every candidate."""
    cands = [_c(f"S{i}", 138.0, 0.0) for i in range(20)]
    r = compute_target_weights(cands, max_name_weight=0.10, cash_floor=0.05)
    assert r.scheme == "equal_weight_fallback"
    # 0.95 invested budget / 20 candidates = 0.0475 each (under the 0.10 cap)
    assert all(math.isclose(w, 0.0475, abs_tol=1e-9) for w in r.weights.values())
    assert math.isclose(sum(r.weights.values()), 0.95, abs_tol=1e-9)
    assert math.isclose(r.cash_weight, 0.05, abs_tol=1e-9)


def test_cold_start_few_candidates_hits_name_cap() -> None:
    """5 candidates × equal weight → would be 19% each, capped at 10%."""
    cands = [_c(f"S{i}", 100.0, 0.0) for i in range(5)]
    r = compute_target_weights(cands, max_name_weight=0.10, cash_floor=0.05)
    assert r.scheme == "equal_weight_fallback"
    assert all(math.isclose(w, 0.10, abs_tol=1e-9) for w in r.weights.values())
    assert r.n_at_name_cap == 5
    # 5 * 0.10 = 0.50 invested, 0.50 cash (cash_floor exceeded — that's fine)
    assert math.isclose(r.cash_weight, 0.50, abs_tol=1e-9)


# ---- Kelly scaling ----
def test_higher_edge_gets_more_weight() -> None:
    cands = [
        _c("A", 200.0, 2000.0),  # edge 200, vol 2000 -> score 200/2000^2 = 5e-5
        _c("B", 100.0, 2000.0),  # score 2.5e-5
    ]
    r = compute_target_weights(cands, max_name_weight=0.10, cash_floor=0.05)
    assert r.scheme == "kelly"
    assert r.weights["A"] > r.weights["B"]
    # Ratio of weights == ratio of scores (before any cap)
    assert math.isclose(r.weights["A"] / r.weights["B"], 2.0, rel_tol=1e-6)


def test_lower_vol_gets_more_weight_same_edge() -> None:
    cands = [
        _c("STABLE", 100.0, 1000.0),   # score 100/1e6 = 1e-4
        _c("VOLATILE", 100.0, 2000.0), # score 100/4e6 = 2.5e-5
    ]
    r = compute_target_weights(cands, max_name_weight=0.10, cash_floor=0.05)
    assert r.weights["STABLE"] > r.weights["VOLATILE"]


def test_negative_edges_get_zero_weight() -> None:
    cands = [
        _c("GOOD", 200.0, 1000.0),
        _c("BAD", -100.0, 1000.0),
        _c("MEH", 0.0, 1000.0),
    ]
    r = compute_target_weights(cands, max_name_weight=0.10, cash_floor=0.05)
    assert "BAD" not in r.weights
    assert "MEH" not in r.weights
    assert "GOOD" in r.weights


def test_all_negative_edges_all_cash() -> None:
    cands = [
        _c("A", -100.0, 1000.0),
        _c("B", -50.0, 1000.0),
    ]
    r = compute_target_weights(cands, max_name_weight=0.10, cash_floor=0.05)
    assert r.scheme == "all_cash"
    assert r.weights == {}


# ---- Cap enforcement (water-filling) ----
def test_water_fill_caps_dominant_name() -> None:
    """One name has 100x edge → would take ~all budget; must be capped at 10%,
    residual redistributed to the weaker names."""
    cands = [
        _c("DOMINANT", 10_000.0, 1000.0),  # score 1e4/1e6 = 0.01
        _c("A", 100.0, 1000.0),             # score 1e-4
        _c("B", 100.0, 1000.0),             # score 1e-4
        _c("C", 100.0, 1000.0),             # score 1e-4
    ]
    r = compute_target_weights(cands, max_name_weight=0.10, cash_floor=0.05)
    assert r.weights["DOMINANT"] == pytest.approx(0.10, abs=1e-9)
    # Remaining 0.85 budget split evenly across A, B, C (equal scores)
    for sym in ("A", "B", "C"):
        assert r.weights[sym] == pytest.approx(0.85 / 3, abs=1e-9)
    assert r.n_at_name_cap == 1
    assert sum(r.weights.values()) == pytest.approx(0.95, abs=1e-9)


def test_multiple_names_hit_cap() -> None:
    cands = [
        _c("A", 10_000.0, 1000.0),
        _c("B", 8_000.0, 1000.0),
        _c("C", 100.0, 1000.0),
        _c("D", 100.0, 1000.0),
    ]
    r = compute_target_weights(cands, max_name_weight=0.10, cash_floor=0.05)
    assert r.weights["A"] == pytest.approx(0.10, abs=1e-9)
    assert r.weights["B"] == pytest.approx(0.10, abs=1e-9)
    # C, D share remaining 0.75
    assert r.weights["C"] == pytest.approx(0.75 / 2, abs=1e-9)
    assert r.weights["D"] == pytest.approx(0.75 / 2, abs=1e-9)


# ---- Invariants ----
@pytest.mark.parametrize("cash_floor", [0.0, 0.05, 0.20, 0.50])
def test_sum_never_exceeds_invested_budget(cash_floor: float) -> None:
    cands = [_c(f"S{i}", 200.0 * (i + 1), 1000.0) for i in range(15)]
    r = compute_target_weights(cands, max_name_weight=0.10, cash_floor=cash_floor)
    invested_budget = 1.0 - cash_floor
    assert sum(r.weights.values()) <= invested_budget + 1e-9


def test_no_weight_exceeds_name_cap() -> None:
    cands = [_c(f"S{i}", 500.0 * i, 1000.0) for i in range(1, 6)]
    r = compute_target_weights(cands, max_name_weight=0.08, cash_floor=0.05)
    for w in r.weights.values():
        assert w <= 0.08 + 1e-9


# ---- Baseline ----
def test_fixed_equal_baseline_basic() -> None:
    cands = [_c(f"S{i}", 138.0, 0.0) for i in range(10)]
    r = fixed_equal_weights(cands, per_name_weight=0.05, max_name_weight=0.10, cash_floor=0.05)
    assert r.scheme == "fixed_equal_baseline"
    assert all(w == 0.05 for w in r.weights.values())
    assert r.cash_weight == pytest.approx(0.50, abs=1e-9)


def test_fixed_equal_baseline_scales_down_when_over_budget() -> None:
    # 30 * 0.05 = 1.50 > 0.95 invested → scale to 0.95 / 30 each
    cands = [_c(f"S{i}", 100.0, 0.0) for i in range(30)]
    r = fixed_equal_weights(cands, per_name_weight=0.05, max_name_weight=0.10, cash_floor=0.05)
    expected = 0.95 / 30
    assert all(w == pytest.approx(expected, abs=1e-9) for w in r.weights.values())


# ---- Validation ----
def test_invalid_max_name_weight_raises() -> None:
    with pytest.raises(ValueError):
        compute_target_weights([_c("A", 100.0, 1000.0)], max_name_weight=0.0, cash_floor=0.05)
    with pytest.raises(ValueError):
        compute_target_weights([_c("A", 100.0, 1000.0)], max_name_weight=1.1, cash_floor=0.05)


def test_invalid_cash_floor_raises() -> None:
    with pytest.raises(ValueError):
        compute_target_weights([_c("A", 100.0, 1000.0)], max_name_weight=0.10, cash_floor=-0.01)
    with pytest.raises(ValueError):
        compute_target_weights([_c("A", 100.0, 1000.0)], max_name_weight=0.10, cash_floor=1.0)
