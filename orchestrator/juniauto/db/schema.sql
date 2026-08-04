-- JuniAuto QuestDB schema
-- All tables partition by day; SYMBOL columns are dictionary-encoded and cheap to filter.
-- Sections in comments refer to PRINCIPLESLONG.md.

-- ============================================================
-- Market data
-- ============================================================
CREATE TABLE IF NOT EXISTS bars (
    symbol       SYMBOL CAPACITY 8192 CACHE,
    ts           TIMESTAMP,
    open         DOUBLE,
    high         DOUBLE,
    low          DOUBLE,
    close        DOUBLE,
    volume       LONG,
    vwap         DOUBLE,
    trade_count  LONG,
    session      SYMBOL CAPACITY 8 CACHE     -- regular | premarket | after_hours | closed
) TIMESTAMP(ts) PARTITION BY DAY WAL DEDUP UPSERT KEYS(ts, symbol);

CREATE TABLE IF NOT EXISTS quotes (
    symbol         SYMBOL CAPACITY 8192 CACHE,
    ts             TIMESTAMP,               -- publish time; IEX floor = 15 min delayed
    bid            DOUBLE,
    ask            DOUBLE,
    bid_size       LONG,
    ask_size       LONG,
    quote_age_min  DOUBLE                    -- (§2.24) age in minutes at snapshot
) TIMESTAMP(ts) PARTITION BY DAY WAL DEDUP UPSERT KEYS(ts, symbol);

-- ============================================================
-- Feature vectors (§1.4-1.5)
-- ============================================================
CREATE TABLE IF NOT EXISTS features (
    symbol                    SYMBOL CAPACITY 8192 CACHE,
    ts                        TIMESTAMP,
    -- Technical (§1.4.1)
    trend_slope               DOUBLE,
    relative_strength         DOUBLE,
    breakout_strength         DOUBLE,
    ma_distance               DOUBLE,
    price_acceleration        DOUBLE,
    volume_confirmation       DOUBLE,
    support_defense           DOUBLE,
    -- Fundamental (§1.4.2)
    earnings_quality          DOUBLE,
    revenue_growth            DOUBLE,
    profitability             DOUBLE,
    balance_sheet_strength    DOUBLE,
    valuation_quality         DOUBLE,
    analyst_revision          DOUBLE,
    -- Event (§1.4.3)
    catalyst_score            DOUBLE,
    earnings_surprise         DOUBLE,
    guidance_change           DOUBLE,
    -- Semantic (§1.4.4)
    context_alignment         DOUBLE,
    sector_context            DOUBLE,
    -- Liquidity (§1.4.5)
    spread_bps                DOUBLE,
    dollar_volume             DOUBLE,
    relative_volume           DOUBLE,
    depth_proxy               DOUBLE,
    -- Risk (§1.4.6)
    realized_vol_bps          DOUBLE,
    beta                      DOUBLE,
    gap_risk                  DOUBLE,
    crowding                  DOUBLE,
    -- Weights (§1.3, §1.7)
    freshness_weight          DOUBLE,
    data_quality              DOUBLE
) TIMESTAMP(ts) PARTITION BY DAY WAL DEDUP UPSERT KEYS(ts, symbol);

-- ============================================================
-- Bayesian posteriors (§2.7)
-- ============================================================
CREATE TABLE IF NOT EXISTS posterior_groups (
    group_id       SYMBOL CAPACITY 32 CACHE,
    ts             TIMESTAMP,
    gamma          DOUBLE,
    beta_mean      DOUBLE,
    beta_var       DOUBLE,
    tau            DOUBLE,
    n_eff          DOUBLE,
    utility_score  DOUBLE,      -- m_k - rho * sqrt(V_k,k)
    last_update    TIMESTAMP
) TIMESTAMP(ts) PARTITION BY MONTH WAL;

-- ============================================================
-- Predictions per (symbol, horizon)
-- ============================================================
CREATE TABLE IF NOT EXISTS predictions (
    symbol             SYMBOL CAPACITY 8192 CACHE,
    ts                 TIMESTAMP,
    horizon            SYMBOL CAPACITY 8 CACHE,      -- 1d | 2-3d | 1wk | 2wk | 1mo
    mu_edge_bps        DOUBLE,
    sigma_edge_bps     DOUBLE,
    sigma_total_bps    DOUBLE,
    p_positive         DOUBLE,
    conservative_edge  DOUBLE,                       -- (§2.6) mu - zq*sigma_total
    role               SYMBOL CAPACITY 4 CACHE,      -- primary | secondary | retained
    membership_edge    DOUBLE,
    composite_edge     DOUBLE                        -- (§2.22a)
) TIMESTAMP(ts) PARTITION BY DAY WAL DEDUP UPSERT KEYS(ts, symbol, horizon);

-- ============================================================
-- Gateway decisions & action evaluations (§2.24-2.26)
-- ============================================================
CREATE TABLE IF NOT EXISTS gateway_actions (
    symbol                SYMBOL CAPACITY 8192 CACHE,
    ts                    TIMESTAMP,
    action_type           SYMBOL CAPACITY 8 CACHE,   -- BUY | SELL | ROTATE | REPLACE | CANCEL | HOLD
    role                  SYMBOL CAPACITY 4 CACHE,   -- primary | secondary | retained
    horizon               SYMBOL CAPACITY 8 CACHE,
    gross_edge_bps        DOUBLE,
    entry_cost_bps        DOUBLE,
    exit_cost_reserved    DOUBLE,
    queue_delay_bps       DOUBLE,
    cancel_replace_bps    DOUBLE,   -- (§2.24) REPLACE / CANCEL surcharge
    action_memory_bps     DOUBLE,
    cash_waiting_value    DOUBLE,
    operational_bps       DOUBLE,
    total_cost_bps        DOUBLE,
    net_edge_bps          DOUBLE,
    hurdle_bps            DOUBLE,
    friction_multiplier   DOUBLE,
    executed              BOOLEAN,
    reject_reason         SYMBOL CAPACITY 32 CACHE
) TIMESTAMP(ts) PARTITION BY DAY WAL;

-- ============================================================
-- Positions snapshot (§3.1 account/PDT queries)
-- ============================================================
CREATE TABLE IF NOT EXISTS positions (
    ts                 TIMESTAMP,
    symbol             SYMBOL CAPACITY 8192 CACHE,
    qty                DOUBLE,
    avg_entry_price    DOUBLE,
    market_value       DOUBLE,
    unrealized_pl      DOUBLE,
    side               SYMBOL CAPACITY 4 CACHE,      -- long | short
    target_weight      DOUBLE,                        -- (weighting) last computed target w*_i (§2.28-2.30)
    actual_weight      DOUBLE,                        -- (weighting) market_value / equity at snapshot time
    weight_drift_bps   DOUBLE                         -- (weighting) (actual - target) * 10000
) TIMESTAMP(ts) PARTITION BY DAY WAL;

-- ============================================================
-- Phantom decision cycle output (cadence validation, no orders placed)
-- Structure mirrors gateway_actions with an extra cycle_time SYMBOL so
-- we can compare basket-forward-returns across different intraday
-- decision times (09:40, 12:30, 15:55, ...) without placing trades.
-- ============================================================
CREATE TABLE IF NOT EXISTS phantom_gateway_actions (
    symbol                SYMBOL CAPACITY 8192 CACHE,
    ts                    TIMESTAMP,
    cycle_time            SYMBOL CAPACITY 8 CACHE,   -- 0940 | 1230 | 1555
    action_type           SYMBOL CAPACITY 8 CACHE,
    role                  SYMBOL CAPACITY 4 CACHE,
    horizon               SYMBOL CAPACITY 8 CACHE,
    gross_edge_bps        DOUBLE,
    entry_cost_bps        DOUBLE,
    exit_cost_reserved    DOUBLE,
    queue_delay_bps       DOUBLE,
    cancel_replace_bps    DOUBLE,
    action_memory_bps     DOUBLE,
    cash_waiting_value    DOUBLE,
    operational_bps       DOUBLE,
    total_cost_bps        DOUBLE,
    net_edge_bps          DOUBLE,
    hurdle_bps            DOUBLE,
    friction_multiplier   DOUBLE,
    executed              BOOLEAN,
    reject_reason         SYMBOL CAPACITY 32 CACHE,
    rebalance_kind        SYMBOL CAPACITY 8 CACHE,
    target_weight         DOUBLE,
    current_weight        DOUBLE,
    delta_weight          DOUBLE,
    realized_return_bps   DOUBLE,                    -- backfilled by phantom resolver
    resolved_at           TIMESTAMP
) TIMESTAMP(ts) PARTITION BY DAY WAL;

-- ============================================================
-- Additive migrations for existing tables
-- ============================================================
-- QuestDB 8.x does NOT support `ADD COLUMN IF NOT EXISTS` (that's a Postgres
-- extension). On fresh installs the columns are already declared in the
-- CREATE TABLE bodies above so these ALTERs raise "column already exists" —
-- benign, caught by apply_schema()'s idempotency handler. On upgrades from
-- pre-weighting installs, they succeed and add the missing columns.

-- positions.* new columns
ALTER TABLE positions ADD COLUMN target_weight DOUBLE;
ALTER TABLE positions ADD COLUMN actual_weight DOUBLE;
ALTER TABLE positions ADD COLUMN weight_drift_bps DOUBLE;

-- gateway_actions.* rebalance audit columns
ALTER TABLE gateway_actions ADD COLUMN rebalance_kind SYMBOL CAPACITY 8 CACHE;
ALTER TABLE gateway_actions ADD COLUMN target_weight DOUBLE;
ALTER TABLE gateway_actions ADD COLUMN current_weight DOUBLE;
ALTER TABLE gateway_actions ADD COLUMN delta_weight DOUBLE;

-- gateway_actions.* trailing-stop hurdle-bump audit column (added 2026-08-04)
-- Records the bps added to minimum_hurdle_bps for a symbol that recently
-- stopped out. reject_reason='hurdle_bump' when this caused the rejection.
ALTER TABLE gateway_actions ADD COLUMN hurdle_bump_bps DOUBLE;
ALTER TABLE phantom_gateway_actions ADD COLUMN hurdle_bump_bps DOUBLE;

-- ============================================================
-- Order / execution telemetry (§3.2 step 7)
-- ============================================================
-- order_id is STRING (not SYMBOL). SYMBOL columns are dictionary-encoded and
-- capped in cardinality; every fill produces a unique order_id, so a large
-- backfill or a few years of live trading would exceed any reasonable cap
-- and cause ILP "Connection reset by peer" errors on flush. STRING has no
-- such cap and is written as a normal column via ILP `columns={}` (not
-- `symbols={}`) — see execution/order_manager.py, bayesian/backfill.py, and
-- backtest/engine.py._persist_fills.
CREATE TABLE IF NOT EXISTS executions (
    order_id             STRING,
    symbol               SYMBOL CAPACITY 8192 CACHE,
    ts                   TIMESTAMP,
    action_type          SYMBOL CAPACITY 8 CACHE,
    side                 SYMBOL CAPACITY 4 CACHE,   -- buy | sell
    qty                  DOUBLE,
    fill_price           DOUBLE,
    decision_ref_price   DOUBLE,
    slippage_bps         DOUBLE,
    spread_bps           DOUBLE,
    market_impact_bps    DOUBLE,
    model_edge_bps       DOUBLE,
    realized_return_bps  DOUBLE,                    -- filled by resolution loop
    horizon              SYMBOL CAPACITY 8 CACHE,
    day_trade            BOOLEAN,
    session              SYMBOL CAPACITY 8 CACHE
) TIMESTAMP(ts) PARTITION BY DAY WAL;

-- ============================================================
-- Account state snapshots (§3.1)
-- ============================================================
CREATE TABLE IF NOT EXISTS account_state (
    ts                 TIMESTAMP,
    equity             DOUBLE,
    cash               DOUBLE,
    buying_power       DOUBLE,
    day_trade_count    INT,                         -- rolling 5-day
    position_count     INT,
    unrealized_pl      DOUBLE,
    realized_pl        DOUBLE,
    pdt_blocked        BOOLEAN
) TIMESTAMP(ts) PARTITION BY MONTH WAL;

-- ============================================================
-- Day-trade audit log (§3.1 PDT enforcement)
-- ============================================================
CREATE TABLE IF NOT EXISTS day_trades (
    ts        TIMESTAMP,
    symbol    SYMBOL CAPACITY 8192 CACHE,
    open_ts   TIMESTAMP,
    close_ts  TIMESTAMP
) TIMESTAMP(ts) PARTITION BY MONTH WAL;

-- ============================================================
-- Shadow monitor deltas (§2.41)
-- ============================================================
CREATE TABLE IF NOT EXISTS shadow_deltas (
    ts                 TIMESTAMP,
    symbol             SYMBOL CAPACITY 8192 CACHE,
    horizon            SYMBOL CAPACITY 8 CACHE,
    delta_net_bps      DOUBLE,
    n_clean            LONG,
    n_eff              LONG,
    positive_share     DOUBLE,
    delta_post         DOUBLE,
    delta_post_se      DOUBLE,
    consecutive_pass   INT,
    promotion_ready    BOOLEAN
) TIMESTAMP(ts) PARTITION BY DAY WAL;

-- ============================================================
-- Market regime observations (observation-only per coordinator review)
-- One row per decision cycle (live + phantom). Records the raw components
-- of the stress score, the EMA-smoothed stress, and the *implied* γ_risk
-- multiplier — but the multiplier is NOT applied to sizing at this stage.
-- After ~60 sessions of shadow data we compare counterfactual weight
-- vectors against live and decide whether to promote to applied.
-- ============================================================
CREATE TABLE IF NOT EXISTS market_regime (
    ts                    TIMESTAMP,
    cycle_type            SYMBOL CAPACITY 8 CACHE,   -- live | 0940 | 1230
    spy_drawdown_pct      DOUBLE,                    -- (spy_close / spy_63d_high) - 1, clipped to [-1, 0]
    spy_vol_pct_rank      DOUBLE,                    -- 21d realized vol percentile in trailing 252d, [0, 1]
    avg_pairwise_corr     DOUBLE,                    -- mean pairwise 21d return correlation among top-50 by dollar volume
    stress_raw            DOUBLE,                    -- blended stress score in [0, 1]
    stress_ema            DOUBLE,                    -- EMA-smoothed stress (halflife = regime_stress_ema_halflife_days)
    gamma_multiplier      DOUBLE,                    -- 1.0 + stress_ema * (gamma_risk_max/gamma_risk_base - 1.0). Observation-only.
    n_symbols_corr        INT                        -- how many names actually contributed to the pairwise-corr average
) TIMESTAMP(ts) PARTITION BY DAY WAL;

-- ============================================================
-- Bayesian training event log — one row per successful retrain_from_db()
-- across all callers (BayesianModel.__init__ warm-up, hourly resolution
-- loop, backfill --retrain flag, backtest engine init). Feeds the Grafana
-- "Bayesian retrain events" panel so we have direct evidence that the
-- ridge is being updated. Absence of rows here = model is stale.
-- ============================================================
CREATE TABLE IF NOT EXISTS bayesian_training_events (
    ts           TIMESTAMP,
    source       SYMBOL CAPACITY 8 CACHE,     -- init | resolution | backfill | backtest
    n_samples    INT,
    n_features   INT,
    y_mean       DOUBLE,
    y_std        DOUBLE,
    y_min        DOUBLE,
    y_max        DOUBLE
) TIMESTAMP(ts) PARTITION BY MONTH WAL;

-- ============================================================
-- Source package evidence scores (§2.20)
-- ============================================================
CREATE TABLE IF NOT EXISTS source_evidence (
    ts               TIMESTAMP,
    package_id       SYMBOL CAPACITY 32 CACHE,
    evidence_bps     DOUBLE,       -- G_p,t
    is_active        BOOLEAN
) TIMESTAMP(ts) PARTITION BY MONTH WAL;

-- ============================================================
-- Trailing stop management (design decision 2026-08-04)
--
-- The system holds mostly fractional-share positions. Alpaca's fractional
-- support permits stop-market orders as DAY-only, so every scan cycle
-- resubmits the daily stop (natural expiry at 16:00 ET) plus the 09:45
-- phantom cycle can REPLACE a live stop when the recomputed level
-- differs from the standing broker-side level by more than the hysteresis
-- threshold.
--
-- Level formula (chandelier + posterior-conditional hybrid):
--   chandelier = high_water_mark - k_chandelier * ATR_20
--   if conservative_edge >= 0:
--       posterior = close_t * (1 - k_loose_vol * daily_vol_pct)
--   else:
--       posterior = close_t * (1 - k_tight_vol * daily_vol_pct)
--   stop = max(chandelier, posterior, prior_stop)   -- trailing, ratchet-up only
--
-- Re-entry: EV-hurdle bump added to minimum_hurdle_bps for symbols that
-- recently stopped out. Decays exponentially (bump_halflife_sessions).
-- ============================================================

-- Per-position stop state. One row per (symbol, cycle_type) tuple per day.
-- `active` = true means alpaca_order_id refers to a currently-standing
-- broker-side order. `triggered` = true after reconciliation detects a fill.
CREATE TABLE IF NOT EXISTS active_stops (
    ts                       TIMESTAMP,           -- last recompute time
    symbol                   SYMBOL CAPACITY 8192 CACHE,
    entry_ts                 TIMESTAMP,           -- position open (for min-hold check)
    entry_price              DOUBLE,
    high_water_mark          DOUBLE,              -- max close since entry
    current_stop_price       DOUBLE,              -- what's live at Alpaca right now
    chandelier_component     DOUBLE,              -- audit trail: F1 raw output
    posterior_component      DOUBLE,              -- audit trail: F3 raw output
    posterior_edge_bps       DOUBLE,              -- Bayesian mu at calc time
    posterior_sigma_bps      DOUBLE,              -- Bayesian sigma at calc time
    conservative_edge_bps    DOUBLE,              -- mu - zq * sigma (§2.6)
    atr_20                   DOUBLE,              -- ATR used in chandelier component
    daily_vol_pct            DOUBLE,              -- daily vol used in posterior component
    qty                      DOUBLE,              -- fractional-share aware
    alpaca_order_id          STRING,              -- current standing stop order id
    cycle_type               SYMBOL CAPACITY 8 CACHE,  -- 0945 | 1230 | 1555
    method                   SYMBOL CAPACITY 16 CACHE, -- hybrid | chandelier | posterior
    active                   BOOLEAN,             -- broker-side order live
    triggered                BOOLEAN,             -- reconciliation detected a fill
    in_canary                BOOLEAN,             -- was in canary list at submit time
    submitted_to_broker      BOOLEAN              -- vs shadow-only (canary excluded)
) TIMESTAMP(ts) PARTITION BY DAY WAL;

-- Audit log of stop fills. Written by the resolution loop when it detects
-- a filled stop order at Alpaca. Feeds Grafana "stop trigger events" panel
-- and provides training data (realized_return_bps -> executions row).
CREATE TABLE IF NOT EXISTS stop_triggers (
    ts                       TIMESTAMP,           -- fill time
    symbol                   SYMBOL CAPACITY 8192 CACHE,
    entry_ts                 TIMESTAMP,
    entry_price              DOUBLE,
    stop_price               DOUBLE,              -- level that fired
    fill_price               DOUBLE,              -- actual fill (may differ from stop_price)
    slippage_vs_stop_bps     DOUBLE,              -- (fill_price - stop_price) / stop_price * 10000
    realized_return_bps      DOUBLE,              -- (fill - entry) / entry * 10000
    qty                      DOUBLE,
    holding_days             INT,                 -- trading days from entry to trigger
    alpaca_order_id          STRING
) TIMESTAMP(ts) PARTITION BY MONTH WAL;

-- Re-entry hurdle-bump state. Written when a stop fires; read by
-- _evaluate_gateway to bump minimum_hurdle_bps for that symbol.
-- Bump decays exponentially with bump_halflife_sessions.
CREATE TABLE IF NOT EXISTS stop_penalties (
    ts                       TIMESTAMP,           -- when the stop fired
    symbol                   SYMBOL CAPACITY 8192 CACHE,
    initial_bump_bps         DOUBLE,
    halflife_sessions        DOUBLE,
    triggered_return_bps     DOUBLE,              -- realized_return_bps from stop_triggers
    active                   BOOLEAN              -- false once bump decays below 1 bp
) TIMESTAMP(ts) PARTITION BY MONTH WAL;
