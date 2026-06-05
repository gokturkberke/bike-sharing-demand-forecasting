"""Experiment-only fitted transforms for environmental recalibration.

These are NOT production feature engineering (which lives in ``features.py``
and is pure/pointwise). Each transform here *learns* parameters from training
rows, so it must be fit per fold: place it at the head of a model pipeline so
cross-validation clones and fits it on the train fold only, never on the
validation/holdout rows. Both transforms return a column-preserving copy of
the input DataFrame (same columns, same order).

Used by ``scripts/run_env_experiment.py``
(docs/experiments/2026-06-05_env-recalibration-and-humidex.md).
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression


class WindspeedZeroImputer(BaseEstimator, TransformerMixin):
    """Treat ``windspeed == 0`` as the anemometer floor (missing) and fill it.

    ``fit`` learns the median of the strictly-positive training windspeed;
    ``transform`` replaces zeros with that learned value and leaves every other
    column untouched. The fill is a function of the fit data only, so a
    held-out zero is filled with the training median - never its own value.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> "WindspeedZeroImputer":
        ws = np.asarray(X["windspeed"], dtype=float)
        positive = ws[ws > 0]
        self.fill_value_ = float(np.median(positive)) if positive.size else 0.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out["windspeed"] = out["windspeed"].mask(out["windspeed"] == 0, self.fill_value_)
        return out


class AtempRecalibrator(BaseEstimator, TransformerMixin):
    """Rebuild ``atemp`` from ``temp`` and ``humidity`` to remove sensor drift.

    ``fit`` learns a linear ``atemp ~ temp + humidity`` on the training rows;
    ``transform`` replaces ``atemp`` with the fitted prediction, leaving every
    other column untouched. Coefficients come from the fit data only.
    """

    FEATURES = ["temp", "humidity"]

    def fit(self, X: pd.DataFrame, y: Any = None) -> "AtempRecalibrator":
        self.model_ = LinearRegression().fit(X[self.FEATURES], X["atemp"])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out["atemp"] = self.model_.predict(X[self.FEATURES]).astype(out["atemp"].dtype)
        return out
