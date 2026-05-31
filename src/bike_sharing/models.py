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
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def _build_ridge(cfg: dict[str, Any]) -> TransformedTargetRegressor:
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
            ("ridge", Ridge(alpha=1.0, random_state=seed)),
        ]
    )
    return TransformedTargetRegressor(
        regressor=inner,
        func=np.log1p,
        inverse_func=np.expm1,
    )


MODEL_FACTORIES = {
    "mean_baseline": lambda cfg: MeanBaseline(),
    "hourly_mean_baseline": lambda cfg: HourlyMeanBaseline(),
    "ridge": _build_ridge,
}


def get_model(name: str, cfg: dict[str, Any]):
    """Return an unfit sklearn-compatible estimator by name."""
    if name not in MODEL_FACTORIES:
        raise ValueError(
            f"Unknown model name: {name!r}. "
            f"Available: {sorted(MODEL_FACTORIES)}."
        )
    return MODEL_FACTORIES[name](cfg)
