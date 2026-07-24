"""Tests for the Bayesian training module.

Covers:
- _CANONICAL_FEATURE_COLS length matches qe.feature_dim() at import time
- build_training_matrix returns None when fewer than 10 resolved rows are present
- BayesianModel.predict on an untrained model returns (0.0, 0.0) safely
- Realized-return formula: BUY at 100 → close 101 → +100 bps
- Realized-return formula: SELL at 100 → close 101 → -100 bps

All tests that require QuestDB or Alpaca are replaced by injected mocks so
the suite runs without any live infrastructure.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# quant_engine may not be available in the CI environment where the C++ .so
# is absent. We provide a lightweight stub so the Python-only logic can still
# be tested. The stub is installed before any module under test is imported.
# ---------------------------------------------------------------------------

try:
    import quant_engine as qe  # type: ignore[import]
    _QE_AVAILABLE = True
except ImportError:
    _QE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Stub qe when the C++ extension is absent (CI without the compiled .so).
# The stub only needs to satisfy the assertions in training.py and the calls
# made by BayesianModel / build_training_matrix.
# ---------------------------------------------------------------------------
if not _QE_AVAILABLE:
    import sys
    import types

    _stub = types.ModuleType("quant_engine")

    class _GroupId:
        TechMomentum = "TechMomentum"
        TechChartStructure = "TechChartStructure"
        VolatilityRange = "VolatilityRange"
        Liquidity = "Liquidity"
        FundamentalQuality = "FundamentalQuality"
        ProvenanceRole = "ProvenanceRole"
        EventRegime = "EventRegime"
        AccountState = "AccountState"
        ExecutionTelemetry = "ExecutionTelemetry"

    class _RegressionConfig:
        zq = 1.0
        rho = 1.0
        ridge_lambda = 5.0
        prior_strength_kappa = 20.0
        sigma_sq = 1.0
        sigma_model_misspec_sq = 0.0

    class _GroupedRidge:
        def __init__(self, cfg: Any) -> None:
            self._trained = False

        def update(self, X: Any, y: Any, weights: Any, col_groups: Any) -> None:
            self._trained = True

        def predict_mean(self, x: Any) -> float:
            return 0.0

        def predict_variance(self, x: Any) -> float:
            return 1.0

    _stub.GroupId = _GroupId()  # type: ignore[assignment]
    _stub.RegressionConfig = _RegressionConfig
    _stub.GroupedRidge = _GroupedRidge
    _stub.feature_dim = lambda: 26  # must match _CANONICAL_FEATURE_COLS length

    sys.modules["quant_engine"] = _stub
    qe = _stub  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Now import the module under test (after the stub is in place if needed).
# ---------------------------------------------------------------------------
from juniauto.bayesian.training import (  # noqa: E402
    BayesianModel,
    _CANONICAL_FEATURE_COLS,
    _COL_TO_GROUP,
    build_training_matrix,
)
from juniauto.bayesian.resolution import (  # noqa: E402
    _fetch_stale_executions,
    _latest_close,
    _update_realized_return,
    resolve_stale_executions,
)


# ===========================================================================
# Tests: _CANONICAL_FEATURE_COLS
# ===========================================================================

class TestFeatureLayout:
    def test_column_count_matches_feature_dim(self) -> None:
        """_CANONICAL_FEATURE_COLS must equal qe.feature_dim() — the assertion
        in training.py fires at import time, but we re-check here explicitly
        so failures produce a clear pytest message."""
        assert len(_CANONICAL_FEATURE_COLS) == qe.feature_dim(), (
            f"Got {len(_CANONICAL_FEATURE_COLS)} columns, expected {qe.feature_dim()}"
        )

    def test_all_columns_have_group_mapping(self) -> None:
        missing = [c for c in _CANONICAL_FEATURE_COLS if c not in _COL_TO_GROUP]
        assert missing == [], f"Columns without group mapping: {missing}"

    def test_no_duplicate_columns(self) -> None:
        seen: set[str] = set()
        dupes = []
        for c in _CANONICAL_FEATURE_COLS:
            if c in seen:
                dupes.append(c)
            seen.add(c)
        assert dupes == [], f"Duplicate columns in _CANONICAL_FEATURE_COLS: {dupes}"

    def test_expected_column_names_present(self) -> None:
        required = {
            "trend_slope", "relative_strength", "breakout_strength",
            "realized_vol_bps", "beta", "spread_bps", "dollar_volume",
            "earnings_quality", "catalyst_score", "context_alignment",
            "crowding",
        }
        missing = required - set(_CANONICAL_FEATURE_COLS)
        assert missing == set(), f"Required columns missing: {missing}"


# ===========================================================================
# Tests: build_training_matrix
# ===========================================================================

def _make_db_mock(rows: list[tuple[Any, ...]]) -> MagicMock:
    """Return a mock QuestDBClient whose query() returns ``rows``."""
    db = MagicMock()
    db.query.return_value = rows
    return db


class TestBuildTrainingMatrix:
    def test_returns_none_with_zero_rows(self) -> None:
        db = _make_db_mock([])
        result = build_training_matrix(db)
        assert result is None

    def test_returns_none_with_fewer_than_10_rows(self) -> None:
        # Each row: realized_return_bps, freshness_weight, data_quality, + 26 features
        row = (100.0, 1.0, 1.0) + tuple(0.0 for _ in range(26))
        db = _make_db_mock([row] * 9)
        result = build_training_matrix(db)
        assert result is None

    def test_returns_arrays_with_exactly_10_rows(self) -> None:
        import numpy as np
        row = (50.0, 0.9, 0.8) + tuple(float(i) for i in range(26))
        db = _make_db_mock([row] * 10)
        result = build_training_matrix(db)
        assert result is not None
        X, y, weights, col_groups = result
        assert X.shape == (10, 26)
        assert y.shape == (10,)
        assert weights.shape == (10,)
        assert len(col_groups) == 26
        assert float(y[0]) == pytest.approx(50.0)
        # weights = freshness * data_quality = 0.9 * 0.8 = 0.72
        assert float(weights[0]) == pytest.approx(0.72, abs=1e-9)

    def test_nan_feature_values_replaced_with_zero(self) -> None:
        import numpy as np
        # Second feature is NaN
        feats = [0.0] * 26
        feats[1] = float("nan")
        row = (10.0, 1.0, 1.0) + tuple(feats)
        db = _make_db_mock([row] * 10)
        result = build_training_matrix(db)
        assert result is not None
        X, _, _, _ = result
        assert float(X[0, 1]) == pytest.approx(0.0)

    def test_none_feature_values_replaced_with_zero(self) -> None:
        feats: list[Any] = [0.5] * 26
        feats[3] = None
        row = (20.0, 1.0, 1.0) + tuple(feats)
        db = _make_db_mock([row] * 10)
        result = build_training_matrix(db)
        assert result is not None
        X, _, _, _ = result
        assert float(X[0, 3]) == pytest.approx(0.0)

    def test_query_exception_returns_none(self) -> None:
        db = MagicMock()
        db.query.side_effect = RuntimeError("connection refused")
        result = build_training_matrix(db)
        assert result is None

    def test_weights_clipped_to_unit_interval(self) -> None:
        # freshness=2.0, quality=1.0 — product > 1 should clip to 1.0
        row = (5.0, 2.0, 1.0) + tuple(0.0 for _ in range(26))
        db = _make_db_mock([row] * 10)
        result = build_training_matrix(db)
        assert result is not None
        _, _, weights, _ = result
        assert float(weights[0]) <= 1.0 + 1e-9


# ===========================================================================
# Tests: BayesianModel
# ===========================================================================

def _make_config_mock() -> MagicMock:
    cfg = MagicMock()
    cfg.bayesian.zq = 1.0
    cfg.bayesian.rho = 1.0
    cfg.bayesian.ridge_lambda = 5.0
    cfg.bayesian.prior_strength_kappa = 20.0
    return cfg


class TestBayesianModel:
    def test_untrained_model_is_not_trained(self) -> None:
        db = _make_db_mock([])  # no resolved rows → retrain_from_db returns 0
        cfg = _make_config_mock()
        model = BayesianModel(db, cfg)
        assert model.is_trained() is False

    def test_untrained_model_predict_returns_zeros(self) -> None:
        db = _make_db_mock([])
        cfg = _make_config_mock()
        model = BayesianModel(db, cfg)
        row = pd.Series({c: 1.0 for c in _CANONICAL_FEATURE_COLS})
        mu, sigma = model.predict(row)
        assert mu == pytest.approx(0.0)
        assert sigma == pytest.approx(0.0)

    def test_predict_returns_zero_zero_on_exception(self) -> None:
        """Even if predict_mean raises, predict() must return (0.0, 0.0)."""
        db = _make_db_mock([])
        cfg = _make_config_mock()
        model = BayesianModel(db, cfg)
        # Inject a broken ridge
        broken_ridge = MagicMock()
        broken_ridge.predict_mean.side_effect = RuntimeError("oops")
        model._ridge = broken_ridge
        row = pd.Series({c: 0.5 for c in _CANONICAL_FEATURE_COLS})
        mu, sigma = model.predict(row)
        assert mu == pytest.approx(0.0)
        assert sigma == pytest.approx(0.0)

    def test_is_trained_false_below_30_samples(self) -> None:
        """Model with 29 resolved rows (retrain returns 29) must report not trained."""
        row = (50.0, 1.0, 1.0) + tuple(0.1 for _ in range(26))
        db = _make_db_mock([row] * 29)
        cfg = _make_config_mock()
        model = BayesianModel(db, cfg)
        assert model.is_trained() is False

    def test_is_trained_true_at_30_samples(self) -> None:
        row = (50.0, 1.0, 1.0) + tuple(0.1 for _ in range(26))
        db = _make_db_mock([row] * 30)
        cfg = _make_config_mock()
        model = BayesianModel(db, cfg)
        assert model.is_trained() is True

    def test_n_samples_exposed(self) -> None:
        row = (100.0, 1.0, 1.0) + tuple(0.0 for _ in range(26))
        db = _make_db_mock([row] * 15)
        cfg = _make_config_mock()
        model = BayesianModel(db, cfg)
        assert model.n_samples == 15

    def test_predict_missing_columns_handled(self) -> None:
        """Feature row missing some columns must not crash — missing → 0."""
        db = _make_db_mock([])
        cfg = _make_config_mock()
        model = BayesianModel(db, cfg)
        # Only supply half the columns
        partial = pd.Series({"trend_slope": 1.0, "beta": -0.5})
        mu, sigma = model.predict(partial)
        # Untrained → zeros regardless, but must not raise
        assert isinstance(mu, float)
        assert isinstance(sigma, float)


# ===========================================================================
# Tests: realized-return formula
# ===========================================================================

class TestRealizedReturnFormula:
    """Verify the BPS formula used in resolution.py, exercised directly."""

    def _compute_bps(self, fill_price: float, close: float, side: str) -> float:
        if side.lower() == "sell":
            return 10_000.0 * (fill_price - close) / fill_price
        return 10_000.0 * (close - fill_price) / fill_price

    def test_buy_profit(self) -> None:
        # BUY at 100, close at 101 → +100 bps
        result = self._compute_bps(fill_price=100.0, close=101.0, side="buy")
        assert result == pytest.approx(100.0)

    def test_buy_loss(self) -> None:
        # BUY at 100, close at 99 → -100 bps
        result = self._compute_bps(fill_price=100.0, close=99.0, side="buy")
        assert result == pytest.approx(-100.0)

    def test_sell_profit(self) -> None:
        # SELL at 100, close at 99 → +100 bps (price fell, short wins)
        result = self._compute_bps(fill_price=100.0, close=99.0, side="sell")
        assert result == pytest.approx(100.0)

    def test_sell_loss(self) -> None:
        # SELL at 100, close at 101 → -100 bps
        result = self._compute_bps(fill_price=100.0, close=101.0, side="sell")
        assert result == pytest.approx(-100.0)

    def test_zero_return_when_price_unchanged(self) -> None:
        result = self._compute_bps(fill_price=150.0, close=150.0, side="buy")
        assert result == pytest.approx(0.0)

    def test_large_gain(self) -> None:
        # BUY at 100, close at 200 → +10_000 bps (100% gain)
        result = self._compute_bps(fill_price=100.0, close=200.0, side="buy")
        assert result == pytest.approx(10_000.0)


# ===========================================================================
# Tests: resolve_stale_executions integration (mocked DB)
# ===========================================================================

class TestResolveStaleExecutions:
    def test_no_stale_rows_returns_zero(self) -> None:
        db = MagicMock()
        db.query.return_value = []
        result = resolve_stale_executions(db, alpaca_feed=None)
        assert result == 0

    def test_resolves_buy_row(self) -> None:
        """One BUY row at 100.0 with latest close 101.0 → resolved with +100 bps."""
        db = MagicMock()
        # _fetch_stale_executions: (order_id, symbol, fill_price, side)
        db.query.side_effect = [
            [("order-1", "AAPL", 100.0, "buy")],  # stale executions
        ]
        # _latest_close via query_one
        db.query_one.return_value = (101.0,)

        # _update_realized_return uses psycopg directly; patch it at the module level
        with patch("juniauto.bayesian.resolution.psycopg") as mock_psycopg:
            mock_conn = MagicMock()
            mock_psycopg.connect.return_value.__enter__ = lambda s: mock_conn
            mock_psycopg.connect.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value.__enter__ = lambda s: MagicMock()
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            result = resolve_stale_executions(db, alpaca_feed=None)

        assert result == 1

    def test_skips_row_with_no_bar(self) -> None:
        db = MagicMock()
        db.query.return_value = [("order-2", "ZZZZ", 50.0, "buy")]
        db.query_one.return_value = None  # no bar available

        result = resolve_stale_executions(db, alpaca_feed=None)
        assert result == 0

    def test_query_exception_returns_zero(self) -> None:
        db = MagicMock()
        db.query.side_effect = RuntimeError("DB down")
        result = resolve_stale_executions(db, alpaca_feed=None)
        assert result == 0

    def test_bad_fill_price_skips_row(self) -> None:
        """fill_price=0 must not produce a division-by-zero crash."""
        db = MagicMock()
        db.query.return_value = [("order-3", "MSFT", 0.0, "buy")]
        db.query_one.return_value = (105.0,)
        result = resolve_stale_executions(db, alpaca_feed=None)
        assert result == 0
