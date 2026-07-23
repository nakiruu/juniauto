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
    side               SYMBOL CAPACITY 4 CACHE       -- long | short
) TIMESTAMP(ts) PARTITION BY DAY WAL;

-- ============================================================
-- Order / execution telemetry (§3.2 step 7)
-- ============================================================
CREATE TABLE IF NOT EXISTS executions (
    order_id             SYMBOL CAPACITY 65536 CACHE,
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
-- Source package evidence scores (§2.20)
-- ============================================================
CREATE TABLE IF NOT EXISTS source_evidence (
    ts               TIMESTAMP,
    package_id       SYMBOL CAPACITY 32 CACHE,
    evidence_bps     DOUBLE,       -- G_p,t
    is_active        BOOLEAN
) TIMESTAMP(ts) PARTITION BY MONTH WAL;
