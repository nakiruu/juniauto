"""ReducedFeatureBayesian — subclass of BayesianModel that zeros out 11
dead feature columns (fundamental + event + semantic families).

Rationale (audit Finding 3): those families either have unreliable data
(yfinance failures on ~15% of symbols) or no data source at all (semantic
is hardcoded to 0.5). They consume ridge regularization budget and
pollute X'X conditioning without contributing signal.

Implementation constraint: C++ `qe.GroupedRidge` requires exactly 26
features (per `qe.feature_dim()`). We can't change that from Python.
Workaround: zero the dead columns in both training and prediction. The
ridge learns coefficient ≈ 0 for zero-variance columns naturally, and
the reduced conditioning improves the numerics on the surviving 15.

Invariants (assertions in code):
  - _DEAD_IDX is disjoint from _KEEP_IDX
  - Union of dead + keep = full 26-column set
  - Every predict() call and every retrain_from_db() zeros dead columns
    with a log emission counting rows/columns touched
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from juniauto.bayesian.training import (
    _CANONICAL_FEATURE_COLS,
    BayesianModel,
    build_training_matrix,
)
from juniauto.utils import get_logger

log = get_logger(__name__)


# The 11 dead feature names per audit Finding 3.
_DEAD_FEATURE_NAMES: set[str] = {
    # Fundamental (yfinance flaky; 10-15% symbols return "empty info")
    "earnings_quality",
    "revenue_growth",
    "profitability",
    "balance_sheet_strength",
    "valuation_quality",
    "analyst_revision",
    # Event (mostly zero outside a narrow earnings window)
    "catalyst_score",
    "earnings_surprise",
    "guidance_change",
    # Semantic (NO DATA SOURCE — hardcoded 0.5 in signals/semantic.py)
    "context_alignment",
    "sector_context",
}

# Resolve to column indices in the canonical order used by C++ GroupedRidge.
_DEAD_IDX: list[int] = [
    i for i, c in enumerate(_CANONICAL_FEATURE_COLS) if c in _DEAD_FEATURE_NAMES
]
_KEEP_IDX: list[int] = [
    i for i, c in enumerate(_CANONICAL_FEATURE_COLS) if c not in _DEAD_FEATURE_NAMES
]

# Invariant assertions — surface a mistake immediately at import time.
assert set(_DEAD_IDX).isdisjoint(_KEEP_IDX), "dead/keep partition overlap"
assert set(_DEAD_IDX) | set(_KEEP_IDX) == set(range(len(_CANONICAL_FEATURE_COLS))), (
    "dead+keep must cover all 26 columns"
)
assert len(_DEAD_IDX) == 11, f"expected 11 dead features, got {len(_DEAD_IDX)}"
assert len(_KEEP_IDX) == 15, f"expected 15 kept features, got {len(_KEEP_IDX)}"


class ReducedFeatureBayesian(BayesianModel):
    """Bayesian ridge that zeros dead features in training + prediction.

    Overrides only `retrain_from_db` (to zero X before ridge.update) and
    `predict` (to zero the feature vector before ridge.predict_mean/variance).
    All spec constants (κ, λ, zq, ρ) preserved from the base class.
    """

    def retrain_from_db(self, source: str = "unknown") -> int:
        """Rebuild the ridge posterior with dead columns zeroed."""
        result = build_training_matrix(self._db)
        if result is None:
            log.info("rf_bayesian_retrain_skipped",
                     reason="insufficient_resolved_rows", source=source)
            return 0

        X, y, weights, col_groups = result
        # Zero out dead columns. Ridge learns coef ≈ 0 for zero-variance
        # columns, which is exactly what we want — the useful 15 features
        # get the entire regularization budget.
        n_zeroed_before = int((X[:, _DEAD_IDX] != 0.0).sum())
        X = X.copy()
        X[:, _DEAD_IDX] = 0.0
        log.info(
            "rf_dead_features_zeroed",
            n_rows=X.shape[0],
            n_dead_cols=len(_DEAD_IDX),
            n_kept_cols=len(_KEEP_IDX),
            n_nonzero_zeroed=n_zeroed_before,
            source=source,
        )

        try:
            self._ridge.update(X, y, weights, col_groups)
            self._n_samples = int(X.shape[0])
            log.info(
                "rf_bayesian_trained",
                n_samples=self._n_samples,
                n_features=X.shape[1],
                n_features_effective=len(_KEEP_IDX),
                y_mean=round(float(np.mean(y)), 3),
                y_std=round(float(np.std(y)), 3),
                source=source,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("rf_bayesian_ridge_update_failed",
                      error=str(exc), source=source)
            self._n_samples = 0
            return 0

        # Persist training event so the "Bayesian retrain events" panel
        # differentiates baseline vs reduced-feature runs.
        try:
            self._persist_training_event(source=f"rf_{source}", X=X, y=y)
        except Exception as exc:  # noqa: BLE001
            log.warning("rf_persist_event_failed", error=str(exc))

        return self._n_samples

    def predict(self, feature_row: pd.Series) -> tuple[float, float]:
        """Zero dead columns before delegating to the C++ ridge."""
        if not self.is_trained():
            return 0.0, 0.0
        try:
            vec = np.array(
                [
                    float(feature_row[c])
                    if c in feature_row.index and pd.notna(feature_row[c])
                    else 0.0
                    for c in _CANONICAL_FEATURE_COLS
                ],
                dtype=np.float64,
            )
            # Zero dead columns. The trained ridge already has coef ≈ 0 for
            # them (they were zero in training), so this is belt-and-braces
            # protection against any live features that DO have data leaking
            # a spurious signal through unshrunk coefficients.
            vec[_DEAD_IDX] = 0.0
            mu = float(self._ridge.predict_mean(vec))
            var = float(self._ridge.predict_variance(vec))
            sigma = float(np.sqrt(max(0.0, var)))
            return mu, sigma
        except Exception as exc:  # noqa: BLE001
            log.warning("rf_bayesian_predict_failed", error=str(exc))
            return 0.0, 0.0
