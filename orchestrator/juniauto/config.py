"""Configuration loader — YAML + env var interpolation, Pydantic-validated.

Every numeric constant here traces back to PRINCIPLESLONG.md via §-references in
`config/production.yaml`. Do not add fields without a spec anchor.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


# ---------- Env interpolation ----------
_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def _interp(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _interp(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_interp(v) for v in node]
    if isinstance(node, str):
        def repl(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            val = os.environ.get(name)
            if val is None:
                if default is None:
                    raise KeyError(f"Env var {name} is unset and has no default")
                val = default
            return val
        return _ENV_PATTERN.sub(repl, node)
    return node


# ---------- Sections ----------
class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # applied model params (§2.19). System-level identity (name/version/timezone)
    # lives in SystemConfig above — do NOT duplicate here, YAML doesn't carry them
    # in this section and pydantic will reject the whole load.
    target_cadence: str
    decision_time_et: str
    source_selection_mode: str
    retained_baseline_floor_bps: int
    primary_role_signal_bps: int
    secondary_role_signal_bps: int
    friction_seed_primary: float
    friction_seed_secondary: float
    friction_seed_retained: float
    exit_reserve: float
    effective_execution_horizon_minutes: int
    rotation_funded_sells: bool
    action_memory_enforcement: str
    automatic_surface_switching: bool
    minimum_holding_period_days: int
    max_day_trades_rolling_5d: int
    minimum_hurdle_bps: int
    source_evidence_threshold_bps: int
    source_evidence_decay: float


class BayesianConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    zq: float
    rho: float
    rho_h_flat: float
    rho_h_ladder: dict[str, float]
    ridge_lambda: float
    prior_strength_kappa: float


class LiquidityCostConfig(BaseModel):
    min_bps: float
    slope: float
    cap_bps: float
    missing_volume_floor_bps: float


class StaleQuoteConfig(BaseModel):
    band1_threshold_sessions: float
    band1_cap_bps: float
    band2_threshold_sessions: float
    band2_cap_bps: float
    band2_slope: float


class GapRiskConfig(BaseModel):
    slope: float
    cap_bps: float


class QueueDelayConfig(BaseModel):
    min_bps: float
    slope_size: float
    slope_session: float
    cap_bps: float


class CancelReplaceConfig(BaseModel):
    api_budget_bps: float
    lost_queue_priority_bps: float
    cap_bps: float


class ActionMemoryConfig(BaseModel):
    fallback_round_trip_bps: float
    horizon_seconds: int


class SlippageConfig(BaseModel):
    per_fill_decay: float
    cold_start_decay: float
    max_fills: int
    cold_start_universe_fills: int
    floor_bps: float
    cap_bps: float


class OperationalCostConfig(BaseModel):
    base_bps: float
    cap_bps: float


class CostsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spread_bps_cap: float
    volatility_bps_cap: float
    sqrt_impact_coeff: float
    liquidity: LiquidityCostConfig
    stale_quote: StaleQuoteConfig
    gap_risk: GapRiskConfig
    queue_delay: QueueDelayConfig
    cancel_replace: CancelReplaceConfig
    action_memory: ActionMemoryConfig
    adverse_selection_share: dict[str, float]
    session_multiplier: dict[str, float]
    slippage: SlippageConfig
    buy_exit_haircut: float
    buy_exit_future_factor: float
    operational: OperationalCostConfig


class SizingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_name_weight: float
    cash_floor: float
    aggregate_comfortable_weight: float
    gamma_risk: float
    lambda_confirm: float


class ShadowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prior_delta: float
    prior_strength_kappa0: float
    min_clean_rows: int
    min_positive_share: float
    required_consecutive_passes: int
    min_n_eff: dict[str, int]
    rho_label: dict[str, float]


class UniverseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_price: float
    min_adv_shares: int
    min_market_cap: int
    exchanges: list[str]
    include_etfs: bool


class AlpacaConfig(BaseModel):
    """Alpaca credentials + endpoints.

    Keys are resolved from environment by `_resolve_alpaca()` based on the
    `ALPACA_PAPER` flag: paper=true reads ALPACA_PAPER_{API,SECRET}_KEY,
    paper=false reads ALPACA_LIVE_{API,SECRET}_KEY. Base URL is derived
    automatically (paper-api vs api) so the caller cannot mismatch keys and URL.
    """
    model_config = ConfigDict(extra="forbid")
    feed: str
    bar_timeframes: list[str]
    history_bars: int
    paper: bool = True
    api_key: str = ""
    secret_key: str = ""
    base_url: str = ""
    data_url: str = "https://data.alpaca.markets"


class YahooConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fundamentals_ttl_days: int


class DatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str
    port: int
    user: str
    password: str
    name: str


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: str = "INFO"
    format: str = "json"
    file: str = "/app/logs/juniauto.log"
    rotation: str = "daily"


class MetricsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool
    port: int
    interval_seconds: int


class SystemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    version: str
    environment: str
    timezone: str


class JuniAutoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    system: SystemConfig
    model: ModelConfig
    bayesian: BayesianConfig
    costs: CostsConfig
    sizing: SizingConfig
    shadow: ShadowConfig
    freshness_halflife_days: dict[str, int]
    universe: UniverseConfig
    alpaca: AlpacaConfig
    yahoo: YahooConfig
    database: DatabaseConfig
    logging: LoggingConfig
    metrics: MetricsConfig


def _resolve_alpaca(raw: dict[str, Any]) -> dict[str, Any]:
    """Pick the paper vs live key pair from environment. Raises if the selected
    pair is missing so we fail fast at boot instead of returning cryptic 401s.
    """
    paper = os.environ.get("ALPACA_PAPER", "true").lower() == "true"
    raw["paper"] = paper
    if paper:
        raw["api_key"] = os.environ.get("ALPACA_PAPER_API_KEY", "")
        raw["secret_key"] = os.environ.get("ALPACA_PAPER_SECRET_KEY", "")
        raw["base_url"] = "https://paper-api.alpaca.markets"
        which = "PAPER"
    else:
        raw["api_key"] = os.environ.get("ALPACA_LIVE_API_KEY", "")
        raw["secret_key"] = os.environ.get("ALPACA_LIVE_SECRET_KEY", "")
        raw["base_url"] = "https://api.alpaca.markets"
        which = "LIVE"
    raw["data_url"] = "https://data.alpaca.markets"
    if not raw["api_key"] or not raw["secret_key"]:
        raise ValueError(
            f"ALPACA_PAPER={'true' if paper else 'false'} but "
            f"ALPACA_{which}_API_KEY / ALPACA_{which}_SECRET_KEY is missing from environment. "
            "Fill both in .env (see .env.example)."
        )
    return raw


def load_config(path: str | Path) -> JuniAutoConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    interpolated = _interp(raw)
    # port is a string after interpolation of "${DB_PORT:-8812}"; coerce.
    if isinstance(interpolated.get("database", {}).get("port"), str):
        interpolated["database"]["port"] = int(interpolated["database"]["port"])
    # Alpaca creds come from env, not YAML; resolve the correct paper/live pair.
    interpolated["alpaca"] = _resolve_alpaca(interpolated.get("alpaca", {}))
    return JuniAutoConfig.model_validate(interpolated)
