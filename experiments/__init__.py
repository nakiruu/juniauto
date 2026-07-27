"""JuniAuto sandbox experiments.

Each subpackage is a self-contained modification of the main strategy.
Experiments MUST NOT alter the live paper-trader or any main-project
configuration. Backtest results are persisted to `backtest_*` tables
with a distinct `run_id` per experiment so Grafana can overlay them
against the production baseline.

Promotion protocol (see individual README.md):
  1. Sharpe >= 0.85 on the full 4.6y window
  2. CAGR >= 13% net of the cost model
  3. MaxDD <= 30%
  4. Improvement holds on walk-forward split
  5. 60 sessions of paper-trading shadow-monitor agreement > 60%

Only experiments clearing all five conditions are candidates for
merging into main.
"""
