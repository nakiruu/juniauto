"""Unit tests for TrailingStopManager — level math, hysteresis, ratchet property.

Covers the pure functions (compute_atr, compute_level, should_replace) and
skips the IO-heavy manage_cycle / reconcile paths (those are exercised end-
to-end via integration tests against paper Alpaca once the canary is populated).
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from juniauto.config import StopsConfig
from juniauto.execution import (
    compute_atr,
    compute_level,
    should_replace,
    snap_to_broker_tick,
)


# Lightweight Bar stub for compute_atr — matches the fields the function reads.
@dataclass(frozen=True)
class _Bar:
    high: float
    low: float
    close: float


def _default_cfg(**overrides: float) -> StopsConfig:
    cfg = StopsConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# --------------------------------------------------------------------------
# compute_atr
# --------------------------------------------------------------------------

def test_compute_atr_two_bars_uses_second_tr() -> None:
    # bar 0: high 105, low 95, close 100  (prev_close undefined — skipped)
    # bar 1: high 108, low 100, close 105
    #   TR = max(108-100, |108-100|, |100-100|) = 8
    bars = [_Bar(105, 95, 100), _Bar(108, 100, 105)]
    assert compute_atr(bars, lookback=20) == pytest.approx(8.0)


def test_compute_atr_uses_gap_via_prev_close() -> None:
    # bar 0 close = 100; bar 1 gaps up: high 115, low 108, close 112
    #   TR = max(115-108, |115-100|, |108-100|) = 15 (dominated by gap)
    bars = [_Bar(105, 95, 100), _Bar(115, 108, 112)]
    assert compute_atr(bars, lookback=20) == pytest.approx(15.0)


def test_compute_atr_averages_recent_tr_within_lookback() -> None:
    # Three usable TR values: 8, 10, 6. Mean = 8.0
    bars = [
        _Bar(105, 95, 100),
        _Bar(108, 100, 105),   # TR 8
        _Bar(115, 105, 108),   # TR max(10, |115-105|, |105-105|) = 10
        _Bar(112, 106, 110),   # TR max(6, |112-108|, |106-108|) = 6
    ]
    assert compute_atr(bars, lookback=20) == pytest.approx(8.0)


def test_compute_atr_returns_zero_when_insufficient_bars() -> None:
    assert compute_atr([], lookback=20) == 0.0
    assert compute_atr([_Bar(100, 90, 95)], lookback=20) == 0.0


# --------------------------------------------------------------------------
# compute_level — chandelier component alone
# --------------------------------------------------------------------------

def test_level_first_computation_no_prior() -> None:
    """Fresh entry, no prior state. Level = max(chandelier, posterior)."""
    cfg = _default_cfg()  # k_chandelier=3.0, k_loose_vol=3.0
    level = compute_level(
        cfg=cfg,
        symbol="TEST",
        qty=10.0,
        entry_price=100.0,
        current_close=105.0,
        prior_stop_price=None,
        prior_high_water_mark=None,
        atr_20=2.0,
        daily_vol_pct=0.015,        # 1.5% daily vol
        posterior_edge_bps=5.0,     # positive edge
        posterior_sigma_bps=3.0,    # positive - 1*3 = +2 conservative
    )
    # HWM = max(100, 105) = 105
    # chandelier = 105 - 3*2 = 99
    # conservative_edge = 5 - 1*3 = 2 >= 0 -> k_loose_vol=3
    # posterior = 105 * (1 - 3*0.015) = 105 * 0.955 = 100.275
    # stop = max(99, 100.275) = 100.275
    assert level.high_water_mark == pytest.approx(105.0)
    assert level.chandelier_component == pytest.approx(99.0)
    assert level.posterior_component == pytest.approx(100.275)
    assert level.stop_price == pytest.approx(100.275)
    assert level.method == "posterior"
    assert level.conservative_edge_bps == pytest.approx(2.0)


def test_level_switches_to_tight_when_conservative_edge_negative() -> None:
    """When conservative_edge < 0 the posterior component uses k_tight_vol
    (1.0 default) instead of k_loose_vol (3.0). Tighter stop."""
    cfg = _default_cfg()
    level = compute_level(
        cfg=cfg,
        symbol="TEST",
        qty=10.0,
        entry_price=100.0,
        current_close=105.0,
        prior_stop_price=None,
        prior_high_water_mark=None,
        atr_20=2.0,
        daily_vol_pct=0.015,
        posterior_edge_bps=1.0,     # positive
        posterior_sigma_bps=5.0,    # 1 - 5 = -4 conservative (adverse)
    )
    # conservative_edge = 1 - 5 = -4 < 0 -> k_tight_vol=1
    # posterior = 105 * (1 - 1*0.015) = 105 * 0.985 = 103.425
    # chandelier = 99
    # stop = max(99, 103.425) = 103.425 — much tighter than loose case
    assert level.posterior_component == pytest.approx(103.425)
    assert level.stop_price == pytest.approx(103.425)
    assert level.conservative_edge_bps == pytest.approx(-4.0)


def test_level_ratchets_up_never_down() -> None:
    """Prior stop above both new components -> keep prior. Trailing property."""
    cfg = _default_cfg()
    level = compute_level(
        cfg=cfg,
        symbol="TEST",
        qty=10.0,
        entry_price=100.0,
        current_close=95.0,          # price dropped
        prior_stop_price=97.0,       # prior stop was already tight
        prior_high_water_mark=110.0,
        atr_20=2.0,
        daily_vol_pct=0.015,
        posterior_edge_bps=5.0,
        posterior_sigma_bps=3.0,
    )
    # HWM = max(100, 95, 110) = 110
    # chandelier = 110 - 3*2 = 104. current close = 95, so 104 > 95 → level would exit anyway
    # posterior = 95 * (1 - 3*0.015) = 90.725 (loose)
    # stop = max(104, 90.725, 97) = 104
    # But wait — chandelier at 104 > current_close 95 → stop already crossed
    # This is fine for the ratchet test: prior 97 vs chandelier 104 vs posterior 90.725 → 104
    assert level.stop_price == pytest.approx(104.0)
    assert level.method == "chandelier"
    assert level.high_water_mark == pytest.approx(110.0)


def test_level_prior_wins_when_new_components_lower() -> None:
    """When both new components fall below the prior stop, keep prior."""
    cfg = _default_cfg()
    level = compute_level(
        cfg=cfg,
        symbol="TEST",
        qty=10.0,
        entry_price=100.0,
        current_close=95.0,
        prior_stop_price=99.0,
        prior_high_water_mark=100.0,   # not a new peak
        atr_20=3.0,                    # wider ATR pulls chandelier down
        daily_vol_pct=0.02,
        posterior_edge_bps=5.0,
        posterior_sigma_bps=3.0,
    )
    # HWM = max(100, 95, 100) = 100
    # chandelier = 100 - 3*3 = 91
    # posterior = 95 * (1 - 3*0.02) = 89.3
    # stop = max(91, 89.3, 99) = 99 (prior wins)
    assert level.stop_price == pytest.approx(99.0)
    assert level.method == "flat_prior"


def test_level_chandelier_wins_when_higher_than_posterior() -> None:
    cfg = _default_cfg()
    level = compute_level(
        cfg=cfg,
        symbol="TEST", qty=10.0,
        entry_price=100.0, current_close=110.0,
        prior_stop_price=None, prior_high_water_mark=115.0,
        atr_20=1.0,                    # tight ATR -> chandelier stays high
        daily_vol_pct=0.03,             # wide vol -> posterior gets loose
        posterior_edge_bps=10.0, posterior_sigma_bps=1.0,
    )
    # HWM = 115. chandelier = 115 - 3 = 112
    # posterior = 110 * (1 - 3*0.03) = 110 * 0.91 = 100.1
    # stop = max(112, 100.1) = 112
    assert level.stop_price == pytest.approx(112.0)
    assert level.method == "chandelier"


def test_level_uses_custom_k_multipliers() -> None:
    """Confirm cfg overrides propagate."""
    cfg = _default_cfg(k_chandelier=2.0, k_loose_vol=2.0)
    level = compute_level(
        cfg=cfg,
        symbol="TEST", qty=10.0,
        entry_price=100.0, current_close=105.0,
        prior_stop_price=None, prior_high_water_mark=None,
        atr_20=2.0, daily_vol_pct=0.01,
        posterior_edge_bps=5.0, posterior_sigma_bps=3.0,
    )
    # chandelier = 105 - 2*2 = 101
    # posterior = 105 * (1 - 2*0.01) = 102.9
    assert level.chandelier_component == pytest.approx(101.0)
    assert level.posterior_component == pytest.approx(102.9)
    assert level.stop_price == pytest.approx(102.9)


# --------------------------------------------------------------------------
# should_replace hysteresis
# --------------------------------------------------------------------------

def test_should_replace_no_current_broker_stop() -> None:
    """No standing stop -> caller submits fresh, don't gate on hysteresis."""
    cfg = _default_cfg()
    assert should_replace(cfg=cfg, new_stop=100.0, current_broker_stop=0.0, atr_20=2.0) is False


def test_should_replace_never_loosens() -> None:
    """new_stop < current — never loosen a trailing stop."""
    cfg = _default_cfg()
    assert should_replace(cfg=cfg, new_stop=98.0, current_broker_stop=100.0, atr_20=2.0) is False


def test_should_replace_equal_stops_no_action() -> None:
    """Level unchanged -> no REPLACE."""
    cfg = _default_cfg()
    assert should_replace(cfg=cfg, new_stop=100.0, current_broker_stop=100.0, atr_20=2.0) is False


def test_should_replace_small_delta_below_threshold_no_action() -> None:
    """Delta below both pct and ATR thresholds -> no REPLACE."""
    cfg = _default_cfg(min_replace_delta_pct=0.005, min_replace_delta_atr_frac=0.05)
    # 100.2 vs 100.0: delta 0.2. pct threshold: 0.005*100=0.5. ATR frac: 0.05*2=0.1
    # max(0.5, 0.1) = 0.5. delta 0.2 < 0.5 -> no REPLACE
    assert should_replace(cfg=cfg, new_stop=100.2, current_broker_stop=100.0, atr_20=2.0) is False


def test_should_replace_delta_above_pct_threshold_fires() -> None:
    cfg = _default_cfg(min_replace_delta_pct=0.005, min_replace_delta_atr_frac=0.05)
    # 100.8 vs 100.0: delta 0.8. threshold max(0.5, 0.1) = 0.5 -> REPLACE
    assert should_replace(cfg=cfg, new_stop=100.8, current_broker_stop=100.0, atr_20=2.0) is True


def test_should_replace_atr_threshold_dominates_when_larger() -> None:
    """Wide ATR raises the threshold above the pct one."""
    cfg = _default_cfg(min_replace_delta_pct=0.005, min_replace_delta_atr_frac=0.05)
    # ATR = 20 -> atr_frac threshold = 0.05*20 = 1.0.  pct threshold = 0.005*100 = 0.5.
    # delta 0.8 < 1.0 (ATR wins) -> no REPLACE even though 0.8 > 0.5 pct threshold
    assert should_replace(cfg=cfg, new_stop=100.8, current_broker_stop=100.0, atr_20=20.0) is False


# --------------------------------------------------------------------------
# Combined scenario — the ROK-style case that motivated the design
# --------------------------------------------------------------------------

def test_scenario_rok_style_posterior_tightens_when_bayesian_turns_adverse() -> None:
    """Simulate the ROK -7% scenario: name that peaked yesterday drops sharply
    today and the Bayesian conservative_edge flips negative. Stop should
    tighten aggressively via the posterior component."""
    cfg = _default_cfg()

    # Day T-1 close: 100. HWM = 100.
    yday = compute_level(
        cfg=cfg, symbol="ROK", qty=5.0,
        entry_price=95.0, current_close=100.0,
        prior_stop_price=None, prior_high_water_mark=None,
        atr_20=1.5, daily_vol_pct=0.015,
        posterior_edge_bps=8.0, posterior_sigma_bps=4.0,   # conservative +4 (bullish)
    )
    # Chandelier: 100 - 3*1.5 = 95.5. Posterior (loose): 100*(1-3*0.015)=95.5. Tie: hybrid
    assert yday.stop_price == pytest.approx(95.5, rel=1e-6)

    # Day T close: 93 (-7%). Bayesian retrained overnight; edge now barely
    # positive but sigma widened (realized vol spiked). conservative_edge < 0.
    today = compute_level(
        cfg=cfg, symbol="ROK", qty=5.0,
        entry_price=95.0, current_close=93.0,
        prior_stop_price=yday.stop_price, prior_high_water_mark=yday.high_water_mark,
        atr_20=2.0,                          # ATR widened after the down move
        daily_vol_pct=0.025,                 # daily vol also up
        posterior_edge_bps=1.0, posterior_sigma_bps=8.0,   # conservative -7 (adverse)
    )
    # HWM: 100 (unchanged, price fell)
    # chandelier: 100 - 3*2 = 94
    # posterior (TIGHT because conservative<0): 93*(1-1*0.025) = 90.675
    # prior: 95.5. stop = max(94, 90.675, 95.5) = 95.5 (prior wins — already above spot!)
    # Broker order would have fired near open when price crossed below 95.5.
    assert today.stop_price == pytest.approx(95.5)
    assert today.method == "flat_prior"
    assert today.conservative_edge_bps == pytest.approx(-7.0)


# --------------------------------------------------------------------------
# snap_to_broker_tick — Alpaca sub-penny + below-market rejection fixes
# --------------------------------------------------------------------------

def test_snap_floors_sub_penny_stops_for_dollar_plus_stocks() -> None:
    """Alpaca rejects stops with sub-penny increments for stocks >= $1."""
    # 310.2954817527754 -> floor to 310.29
    adjusted, note = snap_to_broker_tick(310.2954817527754, current_price=320.0)
    assert adjusted == pytest.approx(310.29)
    assert note == "penny_floor"


def test_snap_half_penny_floors_down() -> None:
    """97.645 must become 97.64 (floor), not 97.65 (round)."""
    adjusted, note = snap_to_broker_tick(97.645, current_price=98.0)
    assert adjusted == pytest.approx(97.64)
    assert note == "penny_floor"


def test_snap_whole_penny_no_adjustment() -> None:
    """Clean whole-penny stop below market -> no note."""
    adjusted, note = snap_to_broker_tick(100.00, current_price=105.00)
    assert adjusted == pytest.approx(100.00)
    assert note == ""


def test_snap_caps_stop_above_market() -> None:
    """AMGN-style: computed 387.12 vs current 387.02.
    Must be capped 25 bps below current."""
    adjusted, note = snap_to_broker_tick(387.12, current_price=387.02)
    # max_allowed = 387.02 * (1 - 25/10000) = 386.052...
    # floor to penny: 386.05
    assert adjusted == pytest.approx(386.05)
    assert note == "capped_below_market"
    assert adjusted < 387.02  # strictly below market


def test_snap_caps_stop_equal_to_market() -> None:
    """MBB-style: stop == current market. Must be pulled below."""
    adjusted, note = snap_to_broker_tick(93.27, current_price=93.27)
    # max_allowed = 93.27 * 0.9975 = 93.03...
    assert adjusted == pytest.approx(93.03)
    assert note == "capped_below_market"


def test_snap_sub_dollar_uses_finer_tick() -> None:
    """Stocks < $1 use $0.0001 tick, not $0.01."""
    adjusted, note = snap_to_broker_tick(0.8543299, current_price=0.90)
    assert adjusted == pytest.approx(0.8543)
    assert note == "penny_floor"


def test_snap_returns_zero_when_market_too_close() -> None:
    """Degenerate input -> caller must skip submission."""
    adjusted, note = snap_to_broker_tick(0.0, current_price=100.0)
    assert adjusted == 0.0
    assert note == "too_close_to_market"

    adjusted, note = snap_to_broker_tick(50.0, current_price=0.0)
    assert adjusted == 0.0
    assert note == "too_close_to_market"


def test_snap_never_returns_price_at_or_above_market() -> None:
    """Invariant: for side='sell', returned stop < current_price always."""
    for computed, current in [
        (100.0, 100.0), (100.5, 100.0), (99.99, 100.0),
        (10.005, 10.00), (1000.0, 999.5),
    ]:
        adjusted, _ = snap_to_broker_tick(computed, current_price=current)
        if adjusted > 0:
            assert adjusted < current, f"stop {adjusted} not below market {current}"


def test_scenario_pltr_style_stop_ratchets_up_on_strong_run() -> None:
    """PLTR +25%: over successive cycles, HWM climbs, chandelier climbs with
    it, stop_price never regresses."""
    cfg = _default_cfg()

    day1 = compute_level(
        cfg=cfg, symbol="PLTR", qty=20.0,
        entry_price=50.0, current_close=52.0,
        prior_stop_price=None, prior_high_water_mark=None,
        atr_20=1.0, daily_vol_pct=0.02,
        posterior_edge_bps=10.0, posterior_sigma_bps=5.0,
    )
    # HWM 52, chandelier 52-3=49, posterior 52*(1-0.06)=48.88, stop=49
    assert day1.stop_price == pytest.approx(49.0)

    day2 = compute_level(
        cfg=cfg, symbol="PLTR", qty=20.0,
        entry_price=50.0, current_close=58.0,
        prior_stop_price=day1.stop_price, prior_high_water_mark=day1.high_water_mark,
        atr_20=1.2, daily_vol_pct=0.02,
        posterior_edge_bps=12.0, posterior_sigma_bps=5.0,
    )
    # HWM = 58, chandelier=58-3*1.2=54.4, posterior=58*0.94=54.52
    # stop = max(54.4, 54.52, 49) = 54.52
    assert day2.stop_price == pytest.approx(54.52, rel=1e-6)
    assert day2.stop_price > day1.stop_price  # ratcheted up

    day3 = compute_level(
        cfg=cfg, symbol="PLTR", qty=20.0,
        entry_price=50.0, current_close=62.5,     # +25% from entry
        prior_stop_price=day2.stop_price, prior_high_water_mark=day2.high_water_mark,
        atr_20=1.5, daily_vol_pct=0.02,
        posterior_edge_bps=10.0, posterior_sigma_bps=5.0,
    )
    # HWM=62.5. chandelier=62.5-4.5=58. posterior=62.5*0.94=58.75. stop=58.75
    assert day3.stop_price == pytest.approx(58.75, rel=1e-6)
    assert day3.stop_price > day2.stop_price
