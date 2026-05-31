"""Training + time-aware cross-validation for Bike Sharing Demand.

Single responsibility: take a fitted-estimator-shaped object plus
``(X, y, datetime)`` and return per-fold metrics aggregated across a
``TimeSeriesSplit``. The orchestrator (``scripts/train_model.py``) wires
this into the rest of the pipeline.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from bike_sharing.evaluate import report


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
