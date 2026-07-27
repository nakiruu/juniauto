"""Signal-separation experiment — combined Fix 1 + Fix 3 from harsh audit.

Hypothesis (from 2026-07-27 audit):
  1. Composite edge = μ + 138 bps constant. The membership_bps * friction
     term is dominated by the 138 bps constant, drowning out the Bayesian
     μ signal (typical range ±15 bps). Kelly weights end up ≈ 138/σ²,
     effectively "buy the K lowest-vol names."
  2. 11 of 26 features (fundamental + event + semantic families) are
     noise or zero — no reliable data source. They consume regularization
     budget and pollute X'X conditioning.

Fix 1: use raw Bayesian μ (not μ + 138) as Kelly ranking numerator AND
sizing weight input. Composite unchanged for gateway execution check —
that's what §2.22a intends.

Fix 3: retrain Bayesian on 15 features only (technical + liquidity + risk).
Zero out the 11 dead columns before ridge.update() and before predict().

Baseline: v2_trained Sharpe 0.44, CAGR 10.64%, MaxDD ~15%.
Expected combined: Sharpe +0.3-0.6 (audit estimate).
"""
from experiments.signal_separation.training import ReducedFeatureBayesian

__all__ = ["ReducedFeatureBayesian"]
