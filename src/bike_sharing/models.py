"""Estimator factory for the Bike Sharing Demand modeling pipeline.

Each model returned by :func:`get_model` is sklearn-compatible (``fit`` +
``predict``) so the train/evaluation code can treat baselines and real
models uniformly.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bike_sharing.preprocessing import from_log1p

# Feature partitioning for the linear model. Raw ordinal time columns
# (hour, month, dayofweek, year) are intentionally dropped: a linear
# model treats them as monotone trends and extrapolates beyond the
# training range (e.g., train month <= 5, validation month = 9). The
# cyclic sin/cos encodings represent the same information in a
# scale-bounded way. Tree models (Phase 5) will use the raw ints.
LINEAR_NUMERIC_COLUMNS = ["temp", "atemp", "humidity", "windspeed"]
LINEAR_PASSTHROUGH_COLUMNS = [
    "holiday",
    "workingday",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    # experiment: 2026-06-01_leakage-safe-feature-sweep.md - the second
    # harmonic and workingday-gated cyclic terms let Ridge represent two
    # daily peaks (holdout RMSLE 0.91 -> 0.72). Bounded, so passthrough.
    "hour_sin2",
    "hour_cos2",
    "hour_sin_workday",
    "hour_cos_workday",
]
LINEAR_ONE_HOT_COLUMNS = ["season", "weather"]


class MeanBaseline(BaseEstimator, RegressorMixin):
    """Predicts the training-set mean of the target for every row."""

    def fit(self, X, y):
        self.mean_ = float(np.asarray(y).mean())
        return self

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.full(n, self.mean_, dtype=float)


class HourlyMeanBaseline(BaseEstimator, RegressorMixin):
    """Predicts the per-hour mean of the target.

    Expects ``X`` to contain an ``hour`` column (provided by
    :func:`bike_sharing.features.build_features`). Falls back to the
    global mean when a test-time hour is absent from training.
    """

    def fit(self, X, y):
        self._check_hour_column(X)
        hour = np.asarray(X["hour"])
        y = np.asarray(y, dtype=float)
        self.global_mean_ = float(y.mean())
        self.hour_means_ = {
            int(h): float(y[hour == h].mean()) for h in np.unique(hour)
        }
        return self

    def predict(self, X):
        self._check_hour_column(X)
        hour = np.asarray(X["hour"]).astype(int)
        return np.array(
            [self.hour_means_.get(h, self.global_mean_) for h in hour], dtype=float
        )

    @staticmethod
    def _check_hour_column(X) -> None:
        if not isinstance(X, pd.DataFrame) or "hour" not in X.columns:
            raise ValueError(
                "HourlyMeanBaseline expects a pandas DataFrame with an "
                "'hour' column. Run build_features before fitting."
            )


def _log_target(regressor) -> TransformedTargetRegressor:
    """Wrap an estimator so it trains on log1p(count) and inverts safely.

    inverse_func is from_log1p (expm1 + clip-at-0), the project's target
    inversion contract: predictions can never be negative. Since
    from_log1p(log1p(y)) == y for non-negative y, check_inverse passes.
    """
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=from_log1p,
    )


def _build_ridge(cfg: dict[str, Any], params: dict[str, Any]) -> TransformedTargetRegressor:
    """ColumnTransformer + Ridge over a log1p-transformed target.

    The ColumnTransformer drops raw ordinal time columns (see module
    constants) so Ridge cannot extrapolate them. One-hots ``season`` and
    ``weather`` to avoid imposing a meaningless ordinal scale.
    """
    seed = int(cfg.get("seed", 42))
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), LINEAR_NUMERIC_COLUMNS),
            ("pass", "passthrough", LINEAR_PASSTHROUGH_COLUMNS),
            (
                "oh",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                LINEAR_ONE_HOT_COLUMNS,
            ),
        ],
        remainder="drop",
    )
    inner = Pipeline(
        [
            ("pre", preprocessor),
            ("ridge", Ridge(alpha=params.get("alpha", 1.0), random_state=seed)),
        ]
    )
    return _log_target(inner)


def build_experimental_ridge(
    cfg: dict[str, Any],
    params: dict[str, Any],
    extra_numeric: tuple[str, ...] = (),
    extra_passthrough: tuple[str, ...] = (),
) -> TransformedTargetRegressor:
    """Experiment-only Ridge: the production Ridge with extra candidate
    columns routed into the ColumnTransformer.

    Used by ``scripts/run_feature_experiment.py`` to test whether candidate
    features help the linear baseline (production Ridge drops any column not
    in the ``LINEAR_*`` lists, so it cannot see them otherwise). Production
    ``get_model('ridge')`` / ``_build_ridge`` are unchanged.
    """
    seed = int(cfg.get("seed", 42))
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), LINEAR_NUMERIC_COLUMNS + list(extra_numeric)),
            (
                "pass",
                "passthrough",
                LINEAR_PASSTHROUGH_COLUMNS + list(extra_passthrough),
            ),
            (
                "oh",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                LINEAR_ONE_HOT_COLUMNS,
            ),
        ],
        remainder="drop",
    )
    inner = Pipeline(
        [
            ("pre", preprocessor),
            ("ridge", Ridge(alpha=params.get("alpha", 1.0), random_state=seed)),
        ]
    )
    return _log_target(inner)


def _build_random_forest(
    cfg: dict[str, Any], params: dict[str, Any]
) -> TransformedTargetRegressor:
    """Random Forest over a log1p target, using the full feature set.

    Unlike Ridge, trees are scale-invariant, so the raw ordinal time
    columns (hour, month, dayofweek, year) are kept - no ColumnTransformer.
    """
    seed = int(cfg.get("seed", 42))
    model = RandomForestRegressor(random_state=seed, **params)
    return _log_target(model)


def _build_gradient_boosting(
    cfg: dict[str, Any], params: dict[str, Any]
) -> TransformedTargetRegressor:
    """Gradient Boosting over a log1p target, using the full feature set."""
    seed = int(cfg.get("seed", 42))
    model = GradientBoostingRegressor(random_state=seed, **params)
    return _log_target(model)


def _build_xgboost(
    cfg: dict[str, Any], params: dict[str, Any]
) -> TransformedTargetRegressor:
    """XGBoost over a log1p target, using the full feature set.

    xgboost is imported lazily so it stays an optional dependency: the
    baselines, Ridge, and the scikit-learn trees all work without it
    installed (AGENTS.md s1 lists XGBoost as a later candidate, not an
    initial requirement).
    """
    try:
        from xgboost import XGBRegressor
    except Exception as exc:  # pragma: no cover - only when xgboost is unusable
        # ImportError if the package is absent; XGBoostError/OSError if the
        # native library cannot load - on macOS that usually means the
        # OpenMP runtime (libomp) is missing.
        raise ImportError(
            "xgboost could not be imported. Install it with "
            "`pip install -r requirements.txt` (or `pip install xgboost`); "
            "on macOS the native runtime also needs OpenMP (`brew install libomp`)."
        ) from exc
    seed = int(cfg.get("seed", 42))
    model = XGBRegressor(random_state=seed, **params)
    return _log_target(model)


MODEL_FACTORIES = {
    "mean_baseline": lambda cfg, params: MeanBaseline(),
    "hourly_mean_baseline": lambda cfg, params: HourlyMeanBaseline(),
    "ridge": _build_ridge,
    "random_forest": _build_random_forest,
    "gradient_boosting": _build_gradient_boosting,
    "xgboost": _build_xgboost,
}


def get_model(name: str, cfg: dict[str, Any], params: dict[str, Any] | None = None):
    """Return an unfit sklearn-compatible estimator by name.

    ``params`` are model-specific hyperparameters (from
    ``config/models.yaml``); when omitted, each factory uses its own
    sensible defaults.
    """
    if name not in MODEL_FACTORIES:
        raise ValueError(
            f"Unknown model name: {name!r}. "
            f"Available: {sorted(MODEL_FACTORIES)}."
        )
    return MODEL_FACTORIES[name](cfg, params or {})
