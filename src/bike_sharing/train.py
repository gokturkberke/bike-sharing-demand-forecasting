"""Training + validation for Bike Sharing Demand.

Two complementary validation views, both leakage-safe:

- ``fit_and_cv``: a chronological ``TimeSeriesSplit`` that simulates
  forecasting future months from past months.
- ``evaluate_holdout``: a single day-of-month split that mirrors the
  axis along which the competition's train/test sets actually differ.
  This is the more representative score for leaderboard-oriented model
  selection (see the threshold note below).

The orchestrator (``scripts/train_model.py``) records both.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from bike_sharing.evaluate import report

# The Kaggle Bike Sharing Demand split is by day-of-month: training rows
# are days 1-19 of every month, test rows are day 20 onward. The labeled
# data we have (train.csv) therefore stops at day 19, so a literal
# "days >= 20" holdout would be empty. The closest leakage-safe local
# analog is to hold out the *latest labeled days* within each month
# (train on days 1-15, validate on days 16-19). This isolates the same
# day-of-month extrapolation the leaderboard tests, which is a different
# axis than the chronological month-over-month split that fit_and_cv
# exercises.
KAGGLE_HOLDOUT_DAY_THRESHOLD = 16


def fit_and_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    datetime_series: pd.Series,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run ``TimeSeriesSplit`` cross-validation and return the metric summary.

    Rows are sorted by ``datetime_series`` before splitting so the train
    fold always precedes its validation fold in time. The estimator is
    cloned for each fold; the input ``model`` itself is left unfit.
    """
    n_splits = int(cfg["cv"]["n_splits"])
    order = np.argsort(np.asarray(datetime_series.values))
    X_sorted = X.iloc[order].reset_index(drop=True)
    y_sorted = np.asarray(y)[order]

    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: list[dict[str, float]] = []

    for train_idx, val_idx in splitter.split(X_sorted):
        fold_model = clone(model)
        fold_model.fit(X_sorted.iloc[train_idx], y_sorted[train_idx])
        y_pred = fold_model.predict(X_sorted.iloc[val_idx])
        fold_metrics.append(report(y_sorted[val_idx], y_pred))

    return _summarize(fold_metrics, n_splits)


def _summarize(
    fold_metrics: list[dict[str, float]], n_splits: int
) -> dict[str, Any]:
    keys = list(fold_metrics[0])
    mean_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    std_metrics = {k: float(np.std([m[k] for m in fold_metrics])) for k in keys}
    return {
        "n_splits": n_splits,
        "mean": mean_metrics,
        "std": std_metrics,
        "folds": fold_metrics,
    }


def kaggle_like_holdout_split(
    datetime_series: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positional ``(train_idx, holdout_idx)`` for the day-of-month split.

    Days below ``KAGGLE_HOLDOUT_DAY_THRESHOLD`` (1-15) go to train; the
    latest labeled days (16-19) go to holdout. See the module constant
    for why this is the closest local analog to the competition's
    1-19/20+ boundary. Indices are positional (0-based) into the series.
    """
    day = np.asarray(datetime_series.dt.day)
    positions = np.arange(len(day))
    train_idx = positions[day < KAGGLE_HOLDOUT_DAY_THRESHOLD]
    holdout_idx = positions[day >= KAGGLE_HOLDOUT_DAY_THRESHOLD]
    return train_idx, holdout_idx


def evaluate_holdout(
    model,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    datetime_series: pd.Series,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Fit on the Kaggle-like train days and score on the holdout days.

    Returns a report dict plus the train/holdout row counts. The input
    ``model`` is cloned, so it is left unfit.
    """
    train_idx, holdout_idx = kaggle_like_holdout_split(datetime_series)
    y_arr = np.asarray(y)
    holdout_model = clone(model)
    holdout_model.fit(X.iloc[train_idx], y_arr[train_idx])
    y_pred = holdout_model.predict(X.iloc[holdout_idx])
    metrics = report(y_arr[holdout_idx], y_pred)
    return {
        "metrics": metrics,
        "n_train": int(len(train_idx)),
        "n_holdout": int(len(holdout_idx)),
    }
