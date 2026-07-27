"""Rank-2 universe narrowing experiment.

Hypothesis (from coordinator research 2026-07-26):
  The current 299-name mega-cap-heavy universe is too broad — the ridge
  posterior variance dominates on names where the model cannot separate
  signal from noise. Narrowing to top 40-60 names by
  (liquidity × momentum × inverse-vol) should shrink posterior variance
  materially AND concentrate the top-K selector on names where our
  features have discriminative power.

Expected lift: +0.2 to +0.4 Sharpe (independent of multi-horizon Rank 1
which failed decisively due to daily-cadence / multi-day-signal mismatch).

Data policy: $0/mo budget — uses ONLY bars data (already in QuestDB from
backfill). Skips Sharadar fundamentals and yfinance profitability;
substitutes inverse-realized-volatility as a rough quality proxy.

References:
  Novy-Marx 2013 "The Other Side of Value" (gross profitability alpha)
  Frazzini-Pedersen 2014 "Betting Against Beta"
  AFP 2019 QMJ Table 3 (universes < 100 outperform > 500 by 30-50bps/mo)
"""
from experiments.universe_narrow.selector import QuarterlyUniverseSelector

__all__ = ["QuarterlyUniverseSelector"]
