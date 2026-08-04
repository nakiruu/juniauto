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
from pydantic import BaseModel, ConfigDict, Field  # noqa: F401


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
    # Reweight step (§2.16-2.17). Any delta_weight smaller than this band
    # collapses to HOLD — prevents cycle-to-cycle churn from tiny numerical
    # drift in the target weights.
    rebalance_dead_band: float = 0.01
    # SELL/ROTATE hurdle in bps. When wired end-to-end this is the minimum
    # EV improvement required to justify closing a live position. For MVP
    # (Bayesian uninformative) it acts as an informational threshold only;
    # the dead-band + PDT gate do the actual filtering.
    rotation_hurdle_bps: float = 15.0
    # ---- Top-K holdings cap (multi-agent-coordinator design commit 3/4) ----
    # Hard cap on number of concurrently-held names once the Bayesian is
    # informative. Gated by top_k_activation_cv_threshold so cold-start
    # falls back to current uncapped equal-weight scheme.
    max_holdings: int = 8
    # Coefficient of variation (std/|mean|) that composite_edge must exceed
    # across executed candidates to activate top-K. Below threshold the
    # ranking is too uniform to be informative — fall back to uncapped Kelly.
    # Empirical: a ridge-regularized Bayesian on 7000 backfill samples
    # produces CV ≈ 0.08. Coordinator's original 0.25 was overcautious and
    # kept top-K silently disabled even under a trained model, leaving the
    # portfolio to accrue leverage from stacking many small positions. 0.05
    # activates on typical trained CVs without firing on cold-start (which
    # has CV=0 by construction).
    top_k_activation_cv_threshold: float = 0.05
    # Edge-delta hysteresis for the top-K selector: incumbents (currently-held
    # names) get this many bps added to their conservative_edge when ranking.
    # A new candidate must beat an incumbent by this margin to displace it,
    # preventing rank-noise churn at the K-th slot boundary.
    hysteresis_edge_delta_bps: float = 20.0


class RegimeConfig(BaseModel):
    """Market-regime stress signal (observation-only, per coordinator review).

    Blends three components into a scalar stress ∈ [0, 1]:
        (a) SPY drawdown from 63-day high, normalized to [0, 1]
        (b) SPY 21-day realized-vol percentile in trailing 252d
        (c) mean pairwise 21d return correlation among the top-N by dollar volume

    EMA-smooth with halflife=`regime_stress_ema_halflife_days` to prevent
    single-day whipsaws from cascading into a large gamma swing. The implied
    gamma_multiplier is *logged* to `market_regime` + Prometheus but NOT
    applied to Kelly sizing while `apply_to_kelly=false`.

    Promotion criterion: after ~60 sessions of shadow logging, compare
    counterfactual weight vectors (γ × stress_ema) against live weights and
    verify turnover cost < expected drawdown reduction.
    """
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    apply_to_kelly: bool = False                        # OBSERVATION-ONLY DURING SHADOW PHASE

    # γ_risk multiplier range. base=1.0 matches sizing.gamma_risk (§2.1).
    # max=3.0 is a soft upper bound — literature (Moreira & Muir 2017)
    # suggests continuous vol-target scalars in the 1-3× range for equity.
    gamma_risk_base: float = 1.0
    gamma_risk_max: float = 3.0

    # Component window params.
    drawdown_lookback_days: int = 63                    # ~3 months trading; captures cyclical bear onset
    vol_lookback_days: int = 21                         # ~1 month realized vol
    vol_pct_lookback_days: int = 252                    # 1y percentile ranking window
    corr_top_n: int = 50                                # top-N names by 20d dollar volume
    corr_lookback_days: int = 21                        # same window as vol for symmetry

    # Component blend weights. Must sum to 1.0; validated in load_config.
    weight_drawdown: float = 0.4
    weight_vol_percentile: float = 0.4
    weight_correlation: float = 0.2

    # EMA halflife for stress smoothing. 5 trading days ≈ 1 week.
    # α = 1 - 0.5**(1/halflife) → halflife=5 → α ≈ 0.129 per cycle.
    regime_stress_ema_halflife_days: float = 5.0

    # Reference symbol for drawdown + vol.
    reference_symbol: str = "SPY"


class ShadowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prior_delta: float
    prior_strength_kappa0: float
    min_clean_rows: int
    min_positive_share: float
    required_consecutive_passes: int
    min_n_eff: dict[str, int]
    rho_label: dict[str, float]


class StopsConfig(BaseModel):
    """Trailing stop management (added 2026-08-04).

    Broker-side DAY stop-market orders on all held positions, resubmitted
    every 15:55 ET (Alpaca DAY expiry at 16:00). Level is the tighter of a
    Chandelier floor and a posterior-conditional accelerator, ratcheting up
    only. 09:45 phantom cycle can REPLACE when the level moves past the
    hysteresis threshold. Re-entry after a stop-out gated by a decaying
    EV-hurdle bump added to minimum_hurdle_bps.

    Rollout: `canary_symbols` acts as the phase-4 gate. Empty list = shadow
    mode (compute levels, persist to active_stops, do NOT submit to Alpaca).
    Populate with a small allowlist to start submitting real stops on those
    names only. Set to ["*"] to enable for all held positions.
    """
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True                      # master kill switch
    # Canary allowlist — empty = shadow only (compute + persist, no broker submits).
    # ["*"] = enable for all held positions. Otherwise = only these symbols.
    canary_symbols: list[str] = []
    # Level formula — Chandelier floor (§2.31 drift, industry-standard 3xATR).
    k_chandelier: float = 3.0
    atr_lookback_days: int = 20
    # Level formula — posterior-conditional accelerator (§2.6 conservative_edge).
    k_loose_vol: float = 3.0   # conservative_edge >= 0: use loose stop
    k_tight_vol: float = 1.0   # conservative_edge < 0: tighten aggressively
    # Hysteresis at 09:45 phantom (15:55 always submits fresh DAY stops).
    # REPLACE fires only when new_stop > current AND delta exceeds max(pct, atr_frac).
    min_replace_delta_pct: float = 0.005      # 50 bps of price
    min_replace_delta_atr_frac: float = 0.05  # OR 5% of ATR
    # Re-entry EV-hurdle bump (§2.26 minimum_required_edge_bps addition).
    bump_initial_bps: float = 50.0
    bump_halflife_sessions: float = 3.0
    bump_deactivate_threshold_bps: float = 1.0  # decay below this = penalty.active=false
    # Operational — Alpaca rate limiting.
    submit_batch_size: int = 20
    submit_pause_ms: int = 200
    # PDT interaction — never submit a stop on entry day (would fire as day trade).
    entry_day_exempt: bool = True


class UniverseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_price: float
    min_adv_shares: int
    min_market_cap: int
    exchanges: list[str]
    include_etfs: bool
    # Optional explicit seed list. When set, the decision cycle uses these
    # symbols directly and skips the full universe builder. Useful for MVP
    # and for pinning a stable universe. Empty list = fall back to the
    # tape-filtered universe builder.
    symbols: list[str] = []
    # Reject symbols whose IEX latest quote has a spread wider than this
    # threshold, up-front with reject_reason=wide_spread. Prevents garbage
    # IEX quotes on less-active names from producing huge false-negative
    # rejections in the cost model.
    max_spread_bps: float = 100.0
    # ---- Pre-filter thresholds (observation-only for MVP) ----
    # Log a warning per symbol whose observed ADV (close × volume, 20d mean)
    # is below `min_adv_usd`. Currently informational — no removal from the
    # candidate set. Once we've watched a few weeks of data, promote to hard
    # enforcement by dropping the symbol before the gateway.
    min_adv_usd: float = 50_000_000.0
    # Typical historical spread threshold. Same observation-only semantics.
    # Runtime wide-spread guard (max_spread_bps above) does the actual per-cycle
    # rejection; this field is for offline curation of the seed list.
    max_typical_spread_bps: float = 30.0


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
    # Master kill switch. When False, YahooFeed short-circuits every call to
    # empty Fundamentals — the pipeline runs without fundamentals and Bayesian
    # cold-start composite edge is unaffected. Flip to True once yfinance is
    # stable in your environment.
    enabled: bool = False
    # Concurrency + hard per-symbol timeout for the enabled path.
    max_workers: int = 4
    per_symbol_timeout_seconds: float = 8.0


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
    # Safety gate: even when the container is up and the pipeline is fully
    # wired, no order is submitted to Alpaca unless this is explicitly True.
    # Default False — flip only after inspecting a few cycles of gateway_actions.
    trading_enabled: bool = False


class JuniAutoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    system: SystemConfig
    model: ModelConfig
    bayesian: BayesianConfig
    costs: CostsConfig
    sizing: SizingConfig
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    stops: StopsConfig = Field(default_factory=StopsConfig)
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
    # trading_enabled is env-var driven ("${TRADING_ENABLED:-false}"); coerce string→bool.
    sys_cfg = interpolated.get("system", {})
    if isinstance(sys_cfg.get("trading_enabled"), str):
        sys_cfg["trading_enabled"] = sys_cfg["trading_enabled"].strip().lower() in (
            "true", "1", "yes", "on",
        )
    # Alpaca creds come from env, not YAML; resolve the correct paper/live pair.
    interpolated["alpaca"] = _resolve_alpaca(interpolated.get("alpaca", {}))
    # Validate regime blend weights sum to 1.0 (±1e-6) if section provided.
    reg = interpolated.get("regime")
    if isinstance(reg, dict):
        w = (
            float(reg.get("weight_drawdown", 0.4))
            + float(reg.get("weight_vol_percentile", 0.4))
            + float(reg.get("weight_correlation", 0.2))
        )
        if abs(w - 1.0) > 1e-6:
            raise ValueError(
                f"regime.weight_* must sum to 1.0, got {w:.6f}. "
                "Adjust weight_drawdown / weight_vol_percentile / weight_correlation."
            )
    return JuniAutoConfig.model_validate(interpolated)
