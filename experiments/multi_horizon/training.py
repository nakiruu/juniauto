"""Multi-horizon Bayesian ridge training.

Three GroupedRidge instances, one per horizon (1d/5d/20d), trained on
the same 26 grouped features. Blended prediction returns weighted-mean
mu and max sigma across the three horizons.

Training data flow:
  1. Fetch bars from QuestDB via REST (bypasses PG wire fragility).
  2. Fetch features from QuestDB via REST (batched by year per the fix
     shipped in orchestrator/juniauto/bayesian/training.py).
  3. Compute y_1d, y_5d, y_20d from bars for each (symbol, date).
  4. Merge with features on (symbol, date).
  5. Fit three ridges.

Reuses:
  - qe.GroupedRidge  (C++ ridge from the main pybind module)
  - _CANONICAL_FEATURE_COLS, _COL_GROUPS  (from main training module — spec §1.4)
  - _rest_query_df, _fetch_features_batched  (main module)
"""
from __future__ import annotations

import urllib.parse
import urllib.request
from datetime import datetime, timezone
from io import StringIO
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import quant_engine as qe

from juniauto.bayesian.training import (
    _CANONICAL_FEATURE_COLS,
    _COL_GROUPS,
    _fetch_features_batched,
)
from juniauto.utils import get_logger

if TYPE_CHECKING:
    from juniauto.config import JuniAutoConfig
    from juniauto.db import QuestDBClient

log = get_logger(__name__)


_MIN_TRAIN_ROWS = 30            # matches main model gate


class MultiHorizonModel:
    """Three-ridge multi-horizon predictor.

    Blend weights are stored on the instance and asserted to sum to 1.0
    at construction. Predictions produce (mu_blend, sigma_blend) where:
        mu_blend = sum(w_h * mu_h for h in horizons)
        sigma_blend = max(sigma_h for h in horizons)  # conservative
    """

    def __init__(
        self,
        db: "QuestDBClient",
        cfg: "JuniAutoConfig",
        horizons: tuple[int, ...] = (1, 5, 20),
        blend_weights: tuple[float, ...] = (0.15, 0.50, 0.35),
    ) -> None:
        if len(horizons) != len(blend_weights):
            raise ValueError("horizons and blend_weights length mismatch")
        w_sum = float(sum(blend_weights))
        if abs(w_sum - 1.0) > 1e-6:
            raise ValueError(f"blend_weights must sum to 1.0, got {w_sum}")

        self._db = db
        self._cfg = cfg
        self.horizons = tuple(int(h) for h in horizons)
        self.blend_weights = tuple(float(w) for w in blend_weights)
        self._weight_by_h = dict(zip(self.horizons, self.blend_weights))

        # One GroupedRidge per horizon — same C++ class as the main model.
        reg_cfg = qe.RegressionConfig()
        reg_cfg.zq = float(cfg.bayesian.zq)
        reg_cfg.rho = float(cfg.bayesian.rho)
        reg_cfg.ridge_lambda = float(cfg.bayesian.ridge_lambda)
        reg_cfg.prior_strength_kappa = float(cfg.bayesian.prior_strength_kappa)
        self._ridges: dict[int, qe.GroupedRidge] = {
            h: qe.GroupedRidge(reg_cfg) for h in self.horizons
        }
        self._n_samples: dict[int, int] = {h: 0 for h in self.horizons}

        # Attempt immediate warm-up.
        try:
            self.retrain_all()
        except Exception as exc:  # noqa: BLE001
            log.warning("mh_init_retrain_failed", error=str(exc))

    # ---- Training ----
    def retrain_all(self) -> dict[int, int]:
        """Fetch training data and fit all ridges. Returns {horizon: n_samples}."""
        matrices = _build_multihorizon_training_matrices(
            self._db, horizons=self.horizons
        )
        if matrices is None:
            log.info("mh_retrain_skipped", reason="insufficient_data")
            return {h: 0 for h in self.horizons}

        for h in self.horizons:
            X, y, w = matrices[h]
            if X.shape[0] < _MIN_TRAIN_ROWS:
                log.info("mh_horizon_undertrained", horizon_days=h, n_rows=X.shape[0])
                self._n_samples[h] = 0
                continue
            try:
                self._ridges[h].update(X, y, w, _COL_GROUPS)
                self._n_samples[h] = int(X.shape[0])
                log.info(
                    "mh_horizon_trained",
                    horizon_days=h,
                    n_samples=self._n_samples[h],
                    y_mean=round(float(np.mean(y)), 3),
                    y_std=round(float(np.std(y)), 3),
                )
                self._persist_training_event(horizon=h, X=X, y=y)
            except Exception as exc:  # noqa: BLE001
                log.error("mh_horizon_train_failed", horizon_days=h, error=str(exc))
                self._n_samples[h] = 0

        return dict(self._n_samples)

    def is_trained(self) -> bool:
        """True when EVERY horizon has met the minimum sample count."""
        return all(n >= _MIN_TRAIN_ROWS for n in self._n_samples.values())

    @property
    def n_samples(self) -> int:
        """Total training samples across all horizons (for observability)."""
        return sum(self._n_samples.values())

    # ---- Prediction ----
    def predict(self, feature_row: pd.Series) -> tuple[float, float]:
        """Blended (mu, sigma) prediction. Returns (0.0, 0.0) if untrained
        or if the feature vector build fails."""
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
            mu_blend = 0.0
            sigma_max = 0.0
            for h in self.horizons:
                mu_h = float(self._ridges[h].predict_mean(vec))
                var_h = float(self._ridges[h].predict_variance(vec))
                sigma_h = float(np.sqrt(max(0.0, var_h)))
                mu_blend += self._weight_by_h[h] * mu_h
                if sigma_h > sigma_max:
                    sigma_max = sigma_h
            return mu_blend, sigma_max
        except Exception as exc:  # noqa: BLE001
            log.warning("mh_predict_failed", error=str(exc))
            return 0.0, 0.0

    # ---- Persistence ----
    def _persist_training_event(
        self, *, horizon: int, X: np.ndarray, y: np.ndarray
    ) -> None:
        try:
            with self._db.sender() as s:
                s.row(
                    "bayesian_training_events",
                    symbols={"source": f"mh_h{horizon}d"},
                    columns={
                        "n_samples": int(X.shape[0]),
                        "n_features": int(X.shape[1]),
                        "y_mean": float(np.mean(y)),
                        "y_std": float(np.std(y)),
                        "y_min": float(np.min(y)),
                        "y_max": float(np.max(y)),
                    },
                    at=datetime.now(tz=timezone.utc),
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("mh_persist_event_failed", horizon_days=horizon, error=str(exc))


# ================================================================
# Training data assembly
# ================================================================
def _build_multihorizon_training_matrices(
    db: "QuestDBClient",
    horizons: tuple[int, ...],
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] | None:
    """Fetch bars + features and build (X, y_h, w) matrices per horizon.

    Approach:
      1. Fetch bars via REST year-batched. Compute y_h from close prices.
      2. Fetch features via _fetch_features_batched (reused from main).
      3. Merge features with per-horizon y on (symbol, date).
      4. Emit (X, y_h, w) tuples keyed by horizon.

    Returns None if any critical fetch fails.
    """
    # ---- 1. Bars for the whole universe & window ----
    bars_df = _fetch_all_bars_batched(db)
    if bars_df is None or bars_df.empty:
        log.error("mh_train_no_bars")
        return None
    log.info("mh_bars_loaded", n_rows=len(bars_df),
             n_symbols=bars_df["symbol"].nunique())

    # ---- 2. Compute y_h from bars per horizon ----
    bars_df = bars_df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    bars_df["_date"] = pd.to_datetime(bars_df["ts"], utc=True).dt.date
    y_frames: dict[int, pd.DataFrame] = {}
    for h in horizons:
        g = bars_df.groupby("symbol", sort=False)
        future_close = g["close"].shift(-h)
        y_bps = 10_000.0 * (future_close - bars_df["close"]) / bars_df["close"]
        y_df = pd.DataFrame({
            "symbol": bars_df["symbol"],
            "_date": bars_df["_date"],
            "y": y_bps.astype(np.float64),
        }).dropna(subset=["y"])
        # Drop rows where future close was 0 or negative (bad data).
        y_df = y_df[np.isfinite(y_df["y"])].reset_index(drop=True)
        y_frames[h] = y_df
        log.info(
            "mh_y_computed", horizon_days=h,
            n_rows=len(y_df), y_mean=round(float(y_df["y"].mean()), 3),
            y_std=round(float(y_df["y"].std()), 3),
        )

    # ---- 3. Features via batched REST (reuse main pipeline) ----
    feat_cols_sql = ", ".join(_CANONICAL_FEATURE_COLS)
    feat_df = _fetch_features_batched(db, feat_cols_sql)
    if feat_df is None or feat_df.empty:
        log.error("mh_train_no_features")
        return None
    feat_df["_date"] = pd.to_datetime(feat_df["ts"], utc=True).dt.date
    log.info("mh_features_loaded", n_rows=len(feat_df))

    # ---- 4. Merge and emit per-horizon (X, y, w) ----
    result: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for h in horizons:
        merged = y_frames[h].merge(
            feat_df.drop(columns=["ts"]),
            on=["symbol", "_date"],
            how="inner",
        )
        if merged.empty:
            log.warning("mh_horizon_merge_empty", horizon_days=h)
            result[h] = (np.zeros((0, len(_CANONICAL_FEATURE_COLS))),
                         np.zeros(0), np.zeros(0))
            continue
        y = merged["y"].astype(np.float64).to_numpy()
        fw = merged.get("freshness_weight", pd.Series(1.0, index=merged.index)).fillna(1.0)
        dq = merged.get("data_quality", pd.Series(1.0, index=merged.index)).fillna(1.0)
        weights = np.clip(fw.astype(np.float64) * dq.astype(np.float64), 0.0, 1.0).to_numpy()
        X = merged[list(_CANONICAL_FEATURE_COLS)].fillna(0.0).astype(np.float64).to_numpy()
        # Invariant check: X, y, w row counts match.
        assert X.shape[0] == y.shape[0] == weights.shape[0], (
            f"row count mismatch: X={X.shape[0]} y={y.shape[0]} w={weights.shape[0]}"
        )
        result[h] = (X, y, weights)
        log.info(
            "mh_horizon_matrix_built",
            horizon_days=h, n_rows=X.shape[0],
            n_features=X.shape[1],
            y_mean=round(float(np.mean(y)), 3),
            y_std=round(float(np.std(y)), 3),
        )
    return result


def _fetch_all_bars_batched(db: "QuestDBClient") -> pd.DataFrame | None:
    """Year-batched bars fetch via REST. Same pattern as
    _fetch_features_batched for consistency and partition-corruption
    tolerance.
    """
    current_year = datetime.now(tz=timezone.utc).year
    years = list(range(2020, current_year + 2))
    parts: list[pd.DataFrame] = []
    failed_years: list[int] = []
    for yr in years:
        sql = (
            "SELECT symbol, ts, close FROM bars "
            f"WHERE ts >= '{yr}-01-01' AND ts < '{yr + 1}-01-01' "
            "ORDER BY symbol, ts"
        )
        try:
            part = _rest_query_df(db, sql, timeout=120.0)
        except Exception as exc:  # noqa: BLE001
            log.warning("mh_bars_batch_failed", year=yr,
                        error=str(exc), error_type=type(exc).__name__)
            failed_years.append(yr)
            continue
        if not part.empty:
            log.info("mh_bars_batch", year=yr, n_rows=len(part))
            parts.append(part)
    if not parts:
        log.error("mh_bars_all_batches_failed", failed_years=failed_years)
        return None
    if failed_years:
        log.warning("mh_bars_partial", failed_years=failed_years,
                    n_successful_batches=len(parts))
    return pd.concat(parts, ignore_index=True)


def _rest_query_df(db: "QuestDBClient", sql: str, timeout: float = 120.0) -> pd.DataFrame:
    """Local copy of the REST helper — avoids importing the underscore-
    prefixed private from the main module. Same behavior."""
    cfg = db._cfg  # noqa: SLF001
    url = f"http://{cfg.host}:9000/exp?" + urllib.parse.urlencode({"query": sql})
    req = urllib.request.Request(url, headers={"User-Agent": "juniauto-mh/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8")
    return pd.read_csv(StringIO(payload))
