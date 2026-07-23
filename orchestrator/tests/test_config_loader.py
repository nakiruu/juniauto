"""Smoke tests for config loading and env interpolation."""
from __future__ import annotations

from pathlib import Path

import pytest

from juniauto.config import load_config

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "production.yaml"


def _set_db_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "8812")
    monkeypatch.setenv("DB_USER", "admin")
    monkeypatch.setenv("DB_PASSWORD", "quest")
    monkeypatch.setenv("DB_NAME", "qdb")
    monkeypatch.setenv("LOG_LEVEL", "INFO")


def test_load_production_config_paper(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _set_db_env(monkeypatch)
    monkeypatch.setenv("ALPACA_PAPER", "true")
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "PKTEST123")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "sk_paper_secret")
    # Live keys intentionally unset — should not be needed on paper.

    cfg = load_config(CONFIG_PATH)
    # Alpaca paper resolution
    assert cfg.alpaca.paper is True
    assert cfg.alpaca.api_key == "PKTEST123"
    assert cfg.alpaca.secret_key == "sk_paper_secret"
    assert cfg.alpaca.base_url == "https://paper-api.alpaca.markets"
    assert cfg.alpaca.data_url == "https://data.alpaca.markets"
    # spec-anchored constants — do not tune without a spec reference
    assert cfg.model.retained_baseline_floor_bps == 200
    assert cfg.model.primary_role_signal_bps == 460
    assert cfg.model.secondary_role_signal_bps == 348
    assert cfg.model.friction_seed_primary == 0.30
    assert cfg.model.effective_execution_horizon_minutes == 390
    assert cfg.model.minimum_holding_period_days == 1
    assert cfg.model.max_day_trades_rolling_5d == 3
    assert cfg.bayesian.zq == 1.0
    assert cfg.bayesian.rho == 1.0
    assert cfg.bayesian.ridge_lambda == 5
    assert cfg.bayesian.prior_strength_kappa == 20
    assert cfg.shadow.prior_strength_kappa0 == 7
    assert cfg.shadow.min_clean_rows == 30
    assert cfg.shadow.min_positive_share == 0.55
    assert cfg.shadow.required_consecutive_passes == 2
    # spec quirk #2: buy exit product = 0.5525
    assert (cfg.costs.buy_exit_haircut * cfg.costs.buy_exit_future_factor) == 0.5525
    # spec quirk #4: SQRT_IMPACT_COEFF appears once
    assert cfg.costs.sqrt_impact_coeff == 25


def test_load_production_config_live(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _set_db_env(monkeypatch)
    monkeypatch.setenv("ALPACA_PAPER", "false")
    monkeypatch.setenv("ALPACA_LIVE_API_KEY", "AKLIVE9999")
    monkeypatch.setenv("ALPACA_LIVE_SECRET_KEY", "sk_live_secret")

    cfg = load_config(CONFIG_PATH)
    assert cfg.alpaca.paper is False
    assert cfg.alpaca.api_key == "AKLIVE9999"
    assert cfg.alpaca.secret_key == "sk_live_secret"
    assert cfg.alpaca.base_url == "https://api.alpaca.markets"


def test_missing_selected_pair_raises(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _set_db_env(monkeypatch)
    monkeypatch.setenv("ALPACA_PAPER", "true")
    # Only live keys set — the paper pair (which is selected) is missing.
    monkeypatch.delenv("ALPACA_PAPER_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET_KEY", raising=False)
    monkeypatch.setenv("ALPACA_LIVE_API_KEY", "AKLIVE9999")
    monkeypatch.setenv("ALPACA_LIVE_SECRET_KEY", "sk_live_secret")

    with pytest.raises(ValueError, match="ALPACA_PAPER_API_KEY / ALPACA_PAPER_SECRET_KEY"):
        load_config(CONFIG_PATH)
