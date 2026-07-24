"""Unit tests for select_top_k + edges_cv (universe commit 3)."""
from __future__ import annotations

import pytest

from juniauto.portfolio import Candidate, edges_cv, select_top_k


def _c(sym: str, edge: float, sigma: float = 1000.0) -> Candidate:
    return Candidate(symbol=sym, conservative_edge_bps=edge, sigma_total_bps=sigma)


# ---- edges_cv ----
def test_edges_cv_empty() -> None:
    assert edges_cv([]) == 0.0


def test_edges_cv_single() -> None:
    assert edges_cv([_c("A", 100)]) == 0.0


def test_edges_cv_uniform_returns_zero() -> None:
    assert edges_cv([_c("A", 100), _c("B", 100), _c("C", 100)]) == 0.0


def test_edges_cv_all_zero_returns_zero() -> None:
    assert edges_cv([_c("A", 0), _c("B", 0), _c("C", 0)]) == 0.0


def test_edges_cv_differentiated() -> None:
    # std([50, 100, 150]) ≈ 40.82, |mean| = 100, CV ≈ 0.408
    cv = edges_cv([_c("A", 50), _c("B", 100), _c("C", 150)])
    assert cv == pytest.approx(0.408, abs=0.01)


# ---- select_top_k ----
def test_top_k_empty_returns_empty() -> None:
    assert select_top_k([], k=5) == []


def test_top_k_zero_returns_empty() -> None:
    assert select_top_k([_c("A", 100)], k=0) == []


def test_top_k_larger_than_candidates_returns_all() -> None:
    cands = [_c("A", 100), _c("B", 50)]
    got = select_top_k(cands, k=5)
    assert len(got) == 2


def test_top_k_ranks_by_edge_when_no_incumbents() -> None:
    cands = [_c("D", 40), _c("A", 100), _c("C", 60), _c("B", 80)]
    got = select_top_k(cands, k=2)
    assert [c.symbol for c in got] == ["A", "B"]


def test_top_k_incumbent_preserved_by_hysteresis() -> None:
    """Incumbent 'HELD' with edge 90 beats new 'NEW' with edge 100 because
    hysteresis adds 20 bps to incumbent (90+20=110 > 100)."""
    cands = [_c("HELD", 90), _c("NEW", 100)]
    got = select_top_k(cands, k=1, incumbents={"HELD"}, hysteresis_edge_bps=20.0)
    assert [c.symbol for c in got] == ["HELD"]


def test_top_k_incumbent_displaced_when_beaten_by_margin() -> None:
    """Incumbent boost of 20 not enough — new candidate edge exceeds by >20."""
    cands = [_c("HELD", 90), _c("NEW", 120)]
    got = select_top_k(cands, k=1, incumbents={"HELD"}, hysteresis_edge_bps=20.0)
    assert [c.symbol for c in got] == ["NEW"]


def test_top_k_multiple_incumbents_at_boundary() -> None:
    """K=3 with 2 incumbents. Hysteresis preserves both if they're near the cut."""
    cands = [
        _c("HELD_A", 80),  # boosted to 100
        _c("HELD_B", 70),  # boosted to 90
        _c("NEW_X", 95),   # no boost
        _c("NEW_Y", 85),   # no boost
        _c("NEW_Z", 50),   # no boost
    ]
    got = select_top_k(cands, k=3, incumbents={"HELD_A", "HELD_B"}, hysteresis_edge_bps=20.0)
    syms = {c.symbol for c in got}
    # Boosted ranking: HELD_A=100, NEW_X=95, HELD_B=90, NEW_Y=85, NEW_Z=50
    assert syms == {"HELD_A", "NEW_X", "HELD_B"}


def test_top_k_hysteresis_zero_disables_incumbent_bonus() -> None:
    cands = [_c("HELD", 90), _c("NEW", 100)]
    got = select_top_k(cands, k=1, incumbents={"HELD"}, hysteresis_edge_bps=0.0)
    assert [c.symbol for c in got] == ["NEW"]
