"""Multi-horizon Bayesian ridge experiment.

Hypothesis (from coordinator research 2026-07-26):
  The 1-day forward-return target has poor SNR (y_std ~= 207 bps vs
  y_mean ~= 6 bps). Multi-day horizons should have proportionally
  better SNR (y_std scales as sqrt(H), y_mean scales linearly if signal
  is real). Blending 5d and 20d predictions as the primary signal with
  1d as a small overlay is expected to lift Sharpe from 0.44 to
  0.75-0.95 without any new data sources or model classes.

Blend weights (chosen): 5d=0.50, 20d=0.35, 1d=0.15.

References:
  Gu-Kelly-Xiu 2020 "Empirical Asset Pricing via ML" (monthly R^2 is 4-6x daily)
  Moskowitz-Ooi-Pedersen 2012 "Time Series Momentum" (Sharpe 1.14 cross-asset)
  Asness-Frazzini-Pedersen 2019 "Quality Minus Junk"
"""
from experiments.multi_horizon.training import MultiHorizonModel

__all__ = ["MultiHorizonModel"]
