-- ============================================================
-- JuniAuto backtest tables (coordinator design review 2026-07-24)
-- Isolated from live tables via `run_id SYMBOL` column; multiple
-- historical runs coexist and are dashboarded via Grafana's $run_id
-- variable. PARTITION BY MONTH (not DAY) because a 2y run has ~24
-- monthly partitions vs ~500 daily ones — simpler to drop a run.
--
-- To purge a run: DELETE WHERE run_id = 'foo' across every table.
-- The DDL below is deliberately additive-only (CREATE IF NOT EXISTS)
-- so schema_backtest.sql can be reapplied idempotently at boot.
-- ============================================================

-- Run metadata (one row per backtest invocation).
CREATE TABLE IF NOT EXISTS backtest_metadata (
    run_id       SYMBOL CAPACITY 1024 CACHE,
    started_at   TIMESTAMP,
    ended_at     TIMESTAMP,
    start_date   TIMESTAMP,                              -- backtest window start
    end_date     TIMESTAMP,                              -- backtest window end
    cli_args     STRING,                                 -- verbatim argv for reproducibility
    config_hash  STRING,                                 -- sha256 of the resolved config
    git_sha      STRING,
    fill_model   SYMBOL CAPACITY 8 CACHE,                -- next_open | close | vwap | delayed_mid
    walkforward_days INT,
    universe_size    INT,
    n_cycles         INT,
    notes        STRING
) TIMESTAMP(started_at) PARTITION BY MONTH WAL;

-- Per-cycle equity curve. One row per (ts, run_id, curve_type).
-- curve_type: main (PDT-enforced), unconstrained (PDT bypassed),
--             benchmark_spy, benchmark_ew, benchmark_fixed5.
CREATE TABLE IF NOT EXISTS backtest_equity_curve (
    ts              TIMESTAMP,
    run_id          SYMBOL CAPACITY 1024 CACHE,
    curve_type      SYMBOL CAPACITY 8 CACHE,
    equity          DOUBLE,
    cash            DOUBLE,
    invested        DOUBLE,
    position_count  INT,
    pdt_day_trade_count INT,
    pdt_blocked     BOOLEAN,
    daily_return_bps DOUBLE,                              -- (equity_t / equity_{t-1} - 1) * 10000
    cum_return_bps   DOUBLE                               -- (equity_t / equity_0 - 1) * 10000
) TIMESTAMP(ts) PARTITION BY MONTH WAL;

-- Gateway actions (mirror of live gateway_actions + run_id + curve_type).
CREATE TABLE IF NOT EXISTS backtest_gateway_actions (
    ts                    TIMESTAMP,
    run_id                SYMBOL CAPACITY 1024 CACHE,
    curve_type            SYMBOL CAPACITY 8 CACHE,
    symbol                SYMBOL CAPACITY 8192 CACHE,
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
    delta_weight          DOUBLE
) TIMESTAMP(ts) PARTITION BY MONTH WAL;

-- Predictions per (symbol, horizon) per run.
CREATE TABLE IF NOT EXISTS backtest_predictions (
    ts                 TIMESTAMP,
    run_id             SYMBOL CAPACITY 1024 CACHE,
    symbol             SYMBOL CAPACITY 8192 CACHE,
    horizon            SYMBOL CAPACITY 8 CACHE,
    mu_edge_bps        DOUBLE,
    sigma_edge_bps     DOUBLE,
    sigma_total_bps    DOUBLE,
    p_positive         DOUBLE,
    conservative_edge  DOUBLE,
    role               SYMBOL CAPACITY 4 CACHE,
    membership_edge    DOUBLE,
    composite_edge     DOUBLE
) TIMESTAMP(ts) PARTITION BY MONTH WAL;

-- Order fills recorded by the SimBroker per run. order_id STRING (not
-- SYMBOL) — same rationale as executions.order_id in schema.sql; a 2y
-- backtest with 300 symbols generates hundreds of thousands of unique
-- order_ids which would exhaust any SYMBOL cap.
CREATE TABLE IF NOT EXISTS backtest_executions (
    ts                   TIMESTAMP,
    run_id               SYMBOL CAPACITY 1024 CACHE,
    curve_type           SYMBOL CAPACITY 8 CACHE,
    order_id             STRING,
    symbol               SYMBOL CAPACITY 8192 CACHE,
    action_type          SYMBOL CAPACITY 8 CACHE,
    side                 SYMBOL CAPACITY 4 CACHE,
    qty                  DOUBLE,
    fill_price           DOUBLE,
    decision_ref_price   DOUBLE,
    slippage_bps         DOUBLE,
    spread_bps           DOUBLE,
    model_edge_bps       DOUBLE,
    realized_return_bps  DOUBLE,
    horizon              SYMBOL CAPACITY 8 CACHE,
    day_trade            BOOLEAN,
    fill_model           SYMBOL CAPACITY 8 CACHE           -- next_open | close | vwap | delayed_mid
) TIMESTAMP(ts) PARTITION BY MONTH WAL;

-- Per-cycle position snapshots (post-fills).
CREATE TABLE IF NOT EXISTS backtest_positions (
    ts                 TIMESTAMP,
    run_id             SYMBOL CAPACITY 1024 CACHE,
    curve_type         SYMBOL CAPACITY 8 CACHE,
    symbol             SYMBOL CAPACITY 8192 CACHE,
    qty                DOUBLE,
    avg_entry_price    DOUBLE,
    market_value       DOUBLE,
    unrealized_pl      DOUBLE,
    target_weight      DOUBLE,
    actual_weight      DOUBLE,
    weight_drift_bps   DOUBLE
) TIMESTAMP(ts) PARTITION BY MONTH WAL;

-- Bayesian posterior snapshots taken at each walk-forward retrain event.
CREATE TABLE IF NOT EXISTS backtest_bayes_snapshots (
    ts             TIMESTAMP,
    run_id         SYMBOL CAPACITY 1024 CACHE,
    n_samples      INT,
    y_mean         DOUBLE,
    y_std          DOUBLE,
    group_id       SYMBOL CAPACITY 32 CACHE,
    gamma          DOUBLE,
    beta_mean      DOUBLE,
    beta_var       DOUBLE,
    tau            DOUBLE,
    n_eff          DOUBLE,
    utility_score  DOUBLE
) TIMESTAMP(ts) PARTITION BY MONTH WAL;

-- Aggregate metrics computed post-run (Sharpe, alpha, beta, max_dd, etc).
-- Long/tall format so adding new metrics doesn't require schema migrations.
CREATE TABLE IF NOT EXISTS backtest_metrics (
    ts             TIMESTAMP,                             -- computed_at
    run_id         SYMBOL CAPACITY 1024 CACHE,
    curve_type     SYMBOL CAPACITY 8 CACHE,               -- which curve the metric applies to
    metric_name    SYMBOL CAPACITY 128 CACHE,             -- e.g. "sharpe_annualized", "alpha_capm_bps"
    metric_value   DOUBLE,
    metric_unit    SYMBOL CAPACITY 16 CACHE               -- "bps" | "ratio" | "pct" | "days" | "count"
) TIMESTAMP(ts) PARTITION BY MONTH WAL;

-- Regime observations recorded during backtest (same shape as live market_regime).
CREATE TABLE IF NOT EXISTS backtest_market_regime (
    ts                    TIMESTAMP,
    run_id                SYMBOL CAPACITY 1024 CACHE,
    cycle_type            SYMBOL CAPACITY 8 CACHE,
    spy_drawdown_pct      DOUBLE,
    spy_vol_pct_rank      DOUBLE,
    avg_pairwise_corr     DOUBLE,
    stress_raw            DOUBLE,
    stress_ema            DOUBLE,
    gamma_multiplier      DOUBLE,
    n_symbols_corr        INT
) TIMESTAMP(ts) PARTITION BY MONTH WAL;
