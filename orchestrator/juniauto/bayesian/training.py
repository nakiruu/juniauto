"""Bayesian ridge training module — feature layout, training matrix builder, and BayesianModel.

Mirrors the C++ feature layout in engine/src/data/features.cpp exactly so that
Python-side row vectors are compatible with GroupedRidge.update() and predict_mean().

References: PRINCIPLESLONG.md §2.6-§2.8, §2.42.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import quant_engine as qe

if TYPE_CHECKING:
    from juniauto.config import JuniAutoConfig
    from juniauto.db import QuestDBClient

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical feature column order — must match engine/src/data/features.cpp kLayout
# exactly (same name, same index, same group). Freshness/data_quality are row-level
# scalars used as observation weights, NOT regression features; they are excluded here.
# ---------------------------------------------------------------------------
_CANONICAL_FEATURE_COLS: list[str] = [
    # Technical Momentum (§1.4.1)
    "trend_slope",
    "relative_strength",
    "breakout_strength",
    "ma_distance",
    "price_acceleration",
    "volume_confirmation",
    # Technical Chart Structure (§1.5)
    "support_defense",
    # Fundamental Quality (§1.4.2)
    "earnings_quality",
    "revenue_growth",
    "profitability",
    "balance_sheet_strength",
    "valuation_quality",
    "analyst_revision",
    # Event / Regime (§1.4.3)
    "catalyst_score",
    "earnings_surprise",
    "guidance_change",
    # Semantic — mapped to EventRegime group per features.cpp comment (§1.4.4)
    "context_alignment",
    "sector_context",
    # Liquidity / Microstructure (§1.4.5)
    "spread_bps",
    "dollar_volume",
    "relative_volume",
    "depth_proxy",
    # Volatility / Risk (§1.4.6)
    "realized_vol_bps",
    "beta",
    "gap_risk",
    "crowding",
]

# Per-column group assignment — mirrors kLayout group tags in features.cpp.
_COL_TO_GROUP: dict[str, qe.GroupId] = {
    "trend_slope": qe.GroupId.TechMomentum,
    "relative_strength": qe.GroupId.TechMomentum,
    "breakout_strength": qe.GroupId.TechMomentum,
    "ma_distance": qe.GroupId.TechMomentum,
    "price_acceleration": qe.GroupId.TechMomentum,
    "volume_confirmation": qe.GroupId.TechMomentum,
    "support_defense": qe.GroupId.TechChartStructure,
    "earnings_quality": qe.GroupId.FundamentalQuality,
    "revenue_growth": qe.GroupId.FundamentalQuality,
    "profitability": qe.GroupId.FundamentalQuality,
    "balance_sheet_strength": qe.GroupId.FundamentalQuality,
    "valuation_quality": qe.GroupId.FundamentalQuality,
    "analyst_revision": qe.GroupId.FundamentalQuality,
    "catalyst_score": qe.GroupId.EventRegime,
    "earnings_surprise": qe.GroupId.EventRegime,
    "guidance_change": qe.GroupId.EventRegime,
    "context_alignment": qe.GroupId.EventRegime,
    "sector_context": qe.GroupId.EventRegime,
    "spread_bps": qe.GroupId.Liquidity,
    "dollar_volume": qe.GroupId.Liquidity,
    "relative_volume": qe.GroupId.Liquidity,
    "depth_proxy": qe.GroupId.Liquidity,
    "realized_vol_bps": qe.GroupId.VolatilityRange,
    "beta": qe.GroupId.VolatilityRange,
    "gap_risk": qe.GroupId.VolatilityRange,
    "crowding": qe.GroupId.VolatilityRange,
}

# Startup assertion: mismatch surfaces loudly before any trading.
assert len(_CANONICAL_FEATURE_COLS) == qe.feature_dim(), (
    f"_CANONICAL_FEATURE_COLS has {len(_CANONICAL_FEATURE_COLS)} columns "
    f"but qe.feature_dim()={qe.feature_dim()}. "
    "Update training.py to match engine/src/data/features.cpp kLayout."
)

# Ordered group list matching _CANONICAL_FEATURE_COLS — passed to GroupedRidge.update().
_COL_GROUPS: list[qe.GroupId] = [_COL_TO_GROUP[c] for c in _CANONICAL_FEATURE_COLS]

_MIN_ROWS_TO_BUILD = 10   # minimum resolved rows to attempt training matrix build
_MIN_ROWS_TO_TRAIN = 30   # minimum to consider the model "trained" (is_trained gate)


def build_training_matrix(
    db: "QuestDBClient",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[qe.GroupId]] | None:
    """Pull resolved executions joined with features from QuestDB.

    Queries executions WHERE realized_return_bps != 0.0 (already resolved),
    joins to features on (symbol, date), and returns the training arrays.

    Args:
        db: Active QuestDBClient instance.

    Returns:
        ``(X, y, weights, col_groups)`` or ``None`` when fewer than
        ``_MIN_ROWS_TO_BUILD`` resolved rows are available.

        - X: float64 ndarray of shape (n, 26), NaN → 0.
        - y: float64 ndarray of shape (n,) — realized_return_bps.
        - weights: float64 ndarray of shape (n,) — freshness_weight * data_quality,
          clipped to [0, 1].
        - col_groups: list of qe.GroupId, length 26, matching column order.
    """
    # Build the SELECT with only the 26 feature columns we need.
    feature_cols_sql = ", ".join(f"f.{c}" for c in _CANONICAL_FEATURE_COLS)

    # QuestDB timestamp truncation: cast ts to date for the join key.
    # executions.ts is the submission timestamp; features.ts is the cycle timestamp.
    # Both are daily, so date-level join is accurate enough.
    sql = f"""
        SELECT
            e.realized_return_bps,
            f.freshness_weight,
            f.data_quality,
            {feature_cols_sql}
        FROM executions e
        INNER JOIN features f
            ON e.symbol = f.symbol
            AND cast(e.ts AS DATE) = cast(f.ts AS DATE)
        WHERE e.realized_return_bps != 0.0
          AND e.action_type IN ('BUY', 'ROTATE')
        ORDER BY e.ts ASC
    """
    try:
        rows = db.query(sql)
    except Exception as exc:
        log.error("build_training_matrix_query_failed", error=str(exc))
        return None

    if len(rows) < _MIN_ROWS_TO_BUILD:
        log.info(
            "build_training_matrix_insufficient",
            n_rows=len(rows),
            min_required=_MIN_ROWS_TO_BUILD,
        )
        return None

    n = len(rows)
    p = len(_CANONICAL_FEATURE_COLS)

    y = np.empty(n, dtype=np.float64)
    weights = np.empty(n, dtype=np.float64)
    X = np.zeros((n, p), dtype=np.float64)

    for i, row in enumerate(rows):
        # row layout: realized_return_bps, freshness_weight, data_quality, feat0..feat25
        y[i] = float(row[0]) if row[0] is not None else 0.0
        fw = float(row[1]) if row[1] is not None else 1.0
        dq = float(row[2]) if row[2] is not None else 1.0
        weights[i] = float(np.clip(fw * dq, 0.0, 1.0))
        for j in range(p):
            val = row[3 + j]
            X[i, j] = float(val) if val is not None and not (
                isinstance(val, float) and np.isnan(val)
            ) else 0.0

    log.info(
        "build_training_matrix_ok",
        n_rows=n,
        n_features=p,
        y_mean=float(np.mean(y)),
        y_std=float(np.std(y)),
    )
    return X, y, weights, _COL_GROUPS


class BayesianModel:
    """Wrapper around qe.GroupedRidge for online Bayesian training and prediction.

    Instantiates the C++ GroupedRidge on construction and immediately tries to
    retrain from the database. Prediction is gated on ``is_trained()`` so the
    cold-start behaviour (return 0.0) remains active until sufficient data
    has been observed.

    No pickle or state persistence — the model is rebuilt from the DB on
    every container start, which is correct given append-only QuestDB storage.
    """

    def __init__(self, db: "QuestDBClient", cfg: "JuniAutoConfig") -> None:
        self._db = db
        self._cfg = cfg
        reg_cfg = qe.RegressionConfig()
        # Wire Bayesian config from production.yaml into the C++ struct.
        # All fields have def_readwrite in pybind_module.cpp so direct assignment works.
        reg_cfg.zq = float(cfg.bayesian.zq)
        reg_cfg.rho = float(cfg.bayesian.rho)
        reg_cfg.ridge_lambda = float(cfg.bayesian.ridge_lambda)
        reg_cfg.prior_strength_kappa = float(cfg.bayesian.prior_strength_kappa)
        self._ridge: qe.GroupedRidge = qe.GroupedRidge(reg_cfg)
        self._n_samples: int = 0
        # Attempt immediate warm-up from any already-resolved rows in the DB.
        try:
            self.retrain_from_db()
        except Exception as exc:  # noqa: BLE001
            log.warning("bayesian_init_retrain_failed", error=str(exc))

    def retrain_from_db(self) -> int:
        """Rebuild the ridge posterior from all resolved executions.

        Returns:
            Number of training samples used. 0 if training was skipped.
        """
        result = build_training_matrix(self._db)
        if result is None:
            log.info("bayesian_retrain_skipped", reason="insufficient_resolved_rows")
            return 0

        X, y, weights, col_groups = result
        try:
            self._ridge.update(X, y, weights, col_groups)
            self._n_samples = int(X.shape[0])
            log.info(
                "bayesian_trained",
                n_samples=self._n_samples,
                n_features=X.shape[1],
            )
        except Exception as exc:  # noqa: BLE001
            log.error("bayesian_ridge_update_failed", error=str(exc))
            self._n_samples = 0
            return 0

        return self._n_samples

    def predict(self, feature_row: pd.Series) -> tuple[float, float]:
        """Predict (mu_edge_bps, sigma_total_bps) for a single candidate.

        Converts the pandas Series to a numpy vector in ``_CANONICAL_FEATURE_COLS``
        order, replacing any NaN with 0.0, then calls the C++ ridge.

        Args:
            feature_row: A row from the features DataFrame indexed by column name.

        Returns:
            ``(mu_edge_bps, sigma_total_bps)`` as floats, or ``(0.0, 0.0)``
            on any failure (the cold-start fallback).
        """
        try:
            vec = np.array(
                [
                    float(feature_row[c]) if c in feature_row.index and pd.notna(feature_row[c]) else 0.0
                    for c in _CANONICAL_FEATURE_COLS
                ],
                dtype=np.float64,
            )
            mu = float(self._ridge.predict_mean(vec))
            # predict_variance returns x'Σx + σ² — take sqrt for sigma units.
            var = float(self._ridge.predict_variance(vec))
            sigma = float(np.sqrt(max(0.0, var)))
            return mu, sigma
        except Exception as exc:  # noqa: BLE001
            log.warning("bayesian_predict_failed", error=str(exc))
            return 0.0, 0.0

    def is_trained(self) -> bool:
        """Return True if the model has been trained on at least 30 resolved samples.

        The 30-sample gate prevents the model from influencing order decisions
        before the posterior has enough data to deviate meaningfully from the prior.
        """
        return self._n_samples >= _MIN_ROWS_TO_TRAIN

    @property
    def n_samples(self) -> int:
        """Number of training samples used in the most recent retrain."""
        return self._n_samples
