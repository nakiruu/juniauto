"""Dynamic action-surface challenger — ridge regression per (symbol_bucket, session_bucket).

Implements DynamicChallenger per §2.42:
    - Ridge regression with λ=5 (config bayesian.ridge_lambda).
    - Two-level shrinkage toward per-bucket fallback target with κ=20
      (config bayesian.prior_strength_kappa).
    - Per-bucket ring buffer of samples; bounded capacity.
    - Feature MAD-standardization before ridge (scale-invariance per §1.2).
    - Exponential decay for regime adaptation (decay factor γ=0.75).

CRITICAL: κ₀ = 7 in monitor.py is the shadow-promotion prior (§2.41).
          κ = 20 here is the ridge-bucket prior (§2.42). Different constants.
See docs/knowledge-base/part5-shadow-and-replay.md §2.42.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np

from juniauto.config import BayesianConfig
from juniauto.utils import get_logger

log = get_logger(__name__)

# Ring buffer capacity per bucket (unspecified in spec — TODO §2.42).
# Default chosen as 200 samples: enough for ~n_eff_decay≈4 at γ=0.75 steady state
# while bounding memory. Revisit when empirical data is available.
_MAX_SAMPLES_PER_BUCKET: int = 200

# Minimum samples before ridge activates over pure shrinkage fallback
# (unspecified in spec — TODO §2.42). Default = 5 (≈n_eff at γ=0.75 convergence).
_MIN_SAMPLES_FOR_RIDGE: int = 5

# Decay factor for regime adaptation (§2.41 approximate n_eff_decay≈4 at γ=0.75)
_DECAY_GAMMA: float = 0.75


@dataclass
class _BucketState:
    """Mutable per-bucket state for the challenger ridge model."""

    # Bounded ring buffer of (feature_vector, realized_bps) pairs
    features: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=_MAX_SAMPLES_PER_BUCKET))
    labels: Deque[float] = field(default_factory=lambda: deque(maxlen=_MAX_SAMPLES_PER_BUCKET))
    weights: Deque[float] = field(default_factory=lambda: deque(maxlen=_MAX_SAMPLES_PER_BUCKET))

    # Cached ridge coefficients (refreshed lazily after each update)
    beta: np.ndarray | None = None

    # Running MAD stats for feature standardization
    feature_sums: np.ndarray | None = None   # sum of raw features
    feature_count: int = 0

    # Fallback target in bps (set from static provenance or zero)
    fallback_bps: float = 0.0


class DynamicChallenger:
    """Per-bucket ridge challenger for action-value estimation.

    Bucket key: (symbol_bucket, session_bucket).
    Feature vector: 6-dimensional per §2.42 context features.

    Two-level shrinkage estimate:
        action_value_hat = (κ·fallback + n·(bucket_mean + adjustment)) / (κ + n)
    where adjustment = (x - mean_x)' @ beta_ridge.

    See docs/knowledge-base/part5-shadow-and-replay.md §2.42.
    """

    def __init__(self, cfg: BayesianConfig) -> None:
        # λ=5 ridge regularization; config key: bayesian.ridge_lambda; §2.42
        self._lambda: float = cfg.ridge_lambda
        # κ=20 prior strength (NOT κ₀=7 from shadow monitor); config key: bayesian.prior_strength_kappa; §2.42
        self._kappa: float = cfg.prior_strength_kappa
        self._buckets: dict[tuple[str, str], _BucketState] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def predict(
        self,
        symbol_bucket: str,
        session_bucket: str,
        features: np.ndarray,
    ) -> float:
        """Predict action value in bps for the given bucket and feature vector.

        Args:
            symbol_bucket:  Bucket identifier for the symbol group
                            (e.g. "sector_tech", "large_cap").
            session_bucket: Session identifier (e.g. "regular", "premarket").
            features:       Feature vector (6-dim per §2.42; MAD-standardized
                            inside this method).

        Returns:
            Predicted mu_bps (action value estimate in basis points).
        """
        key = (symbol_bucket, session_bucket)
        state = self._buckets.get(key)

        if state is None or state.feature_count == 0:
            return 0.0  # cold start → fallback is zero (no prior data)

        # MAD-standardize the query feature vector using bucket statistics.
        # §1.2 note: standardize before ridge so λ is scale-invariant.
        x_std = self._standardize(features, state)

        n = len(state.labels)
        if n == 0:
            return state.fallback_bps

        bucket_mean = sum(state.labels) / n

        # Compute ridge adjustment if enough samples.
        adjustment = 0.0
        if n >= _MIN_SAMPLES_FOR_RIDGE and state.beta is not None:
            x_centered = x_std - self._mean_features(state)
            adjustment = float(np.dot(x_centered, state.beta))

        # Two-level shrinkage estimate per §2.42.
        # action_value_hat = (κ·fallback + n·(bucket_mean + adjustment)) / (κ + n)
        action_value = (
            self._kappa * state.fallback_bps + n * (bucket_mean + adjustment)
        ) / (self._kappa + n)

        return float(action_value)

    def update(
        self,
        symbol_bucket: str,
        session_bucket: str,
        features: np.ndarray,
        realized_bps: float,
        fallback_bps: float = 0.0,
        weight: float = 1.0,
    ) -> None:
        """Append a realized observation and refresh ridge coefficients.

        Args:
            symbol_bucket:  Bucket identifier.
            session_bucket: Session identifier.
            features:       Feature vector at decision time.
            realized_bps:   Realized net_bps outcome.
            fallback_bps:   Static provenance fallback for this bucket (§2.42).
            weight:         Observation weight (e.g., data_quality from §1.7).
        """
        key = (symbol_bucket, session_bucket)
        if key not in self._buckets:
            self._buckets[key] = _BucketState(fallback_bps=fallback_bps)

        state = self._buckets[key]
        state.fallback_bps = fallback_bps

        # Update running feature sum for mean tracking.
        d = len(features)
        if state.feature_sums is None:
            state.feature_sums = np.zeros(d, dtype=float)
        state.feature_sums += features
        state.feature_count += 1

        state.features.append(features.astype(float))
        state.labels.append(float(realized_bps))
        state.weights.append(float(weight))

        # Refresh ridge if enough samples.
        if len(state.labels) >= _MIN_SAMPLES_FOR_RIDGE:
            self._fit_ridge(state)

    def decay(self, factor: float = _DECAY_GAMMA) -> None:
        """Apply exponential weight decay to all bucket samples for regime adaptation.

        Multiplies each sample's weight by `factor`. Samples with near-zero
        weight contribute negligibly to the ridge fit.

        Args:
            factor: Decay multiplier (default γ=0.75 per §2.41 n_eff_decay≈4).
        """
        for state in self._buckets.values():
            new_weights = deque(maxlen=_MAX_SAMPLES_PER_BUCKET)
            for w in state.weights:
                new_weights.append(w * factor)
            state.weights = new_weights
            # Refresh ridge with updated weights.
            if len(state.labels) >= _MIN_SAMPLES_FOR_RIDGE:
                self._fit_ridge(state)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _fit_ridge(self, state: _BucketState) -> None:
        """Fit weighted ridge regression: beta = (X'WX + λI)^{-1} X'Wy.

        Features are MAD-standardized before the ridge solve so λ is
        scale-invariant (§1.2 note, §2.42).
        """
        X_raw = np.array(list(state.features), dtype=float)
        y = np.array(list(state.labels), dtype=float)
        w = np.array(list(state.weights), dtype=float)

        if X_raw.ndim == 1:
            X_raw = X_raw.reshape(-1, 1)

        # MAD-standardize columns.
        X_std = self._standardize_matrix(X_raw)

        # Center y by weighted mean.
        w_sum = w.sum()
        if w_sum <= 0:
            state.beta = None
            return
        y_mean = np.dot(w, y) / w_sum
        y_centered = y - y_mean

        # Weighted ridge: (X'WX + λI)^{-1} X'Wy
        W = np.diag(w)
        XtW = X_std.T @ W
        A = XtW @ X_std + self._lambda * np.eye(X_std.shape[1])
        b = XtW @ y_centered

        try:
            state.beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            state.beta = None

    @staticmethod
    def _standardize_matrix(X: np.ndarray) -> np.ndarray:
        """MAD-standardize each column of X.

        x_standardized = (x - median) / (1.4826 * MAD)
        Columns with zero MAD are left unchanged (set to zero).
        §1.2, §2.42 note on scale-invariance.
        """
        X_std = np.zeros_like(X, dtype=float)
        for j in range(X.shape[1]):
            col = X[:, j]
            med = float(np.median(col))
            mad = float(np.median(np.abs(col - med)))
            scale = 1.4826 * mad
            if scale > 0:
                X_std[:, j] = (col - med) / scale
            # else: leave as zero (all observations identical)
        return X_std

    @staticmethod
    def _standardize(features: np.ndarray, state: _BucketState) -> np.ndarray:
        """MAD-standardize a single feature vector using bucket sample stats."""
        if not state.features:
            return features.astype(float)
        X_raw = np.array(list(state.features), dtype=float)
        if X_raw.ndim == 1:
            X_raw = X_raw.reshape(-1, 1)
        meds = np.median(X_raw, axis=0)
        mads = np.median(np.abs(X_raw - meds), axis=0)
        scales = 1.4826 * mads
        out = features.astype(float).copy()
        for j in range(len(out)):
            if scales[j] > 0:
                out[j] = (out[j] - meds[j]) / scales[j]
        return out

    @staticmethod
    def _mean_features(state: _BucketState) -> np.ndarray:
        """Compute column means of the stored feature vectors."""
        if not state.features:
            return np.zeros(1)
        X = np.array(list(state.features), dtype=float)
        return X.mean(axis=0)
