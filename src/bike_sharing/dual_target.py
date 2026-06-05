"""Optional stretch: dual-target modeling (separate casual + registered).

Experiment-only. Predicts ``casual`` and ``registered`` with independent
models and sums them on the original count scale, as an alternative to the
direct-``count`` model. Leakage-safe: the feature matrix excludes ``count``,
``casual``, and ``registered`` (and ``datetime``), so neither sub-target is
ever a feature for the other or for count (CLAUDE.md stretch rule). Each
sub-model trains on its own ``log1p`` target via ``get_model`` and inverts
with the project's non-negative inverse before summing.

Validation mirrors the direct-count path and is fold-safe: fresh models are
fit on each fold's train rows only, predict that fold's validation rows, and
the summed count-scale prediction is scored against true count. Used by
scripts/train_dual_target.py (docs/experiments/2026-06-05_dual-target.md).
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from bike_sharing.evaluate import report
from bike_sharing.models import get_model
from bike_sharing.train import day_of_month_holdout_split

DUAL_TARGET_COLUMNS = ("casual", "registered")


def dual_target_split(df: pd.DataFrame, cfg: dict[str, Any]):
    """Return ``(X, casual, registered, count, datetime)`` for the experiment.

    ``X`` excludes ``count``, ``casual``, ``registered``, and the datetime
    column, so neither sub-target (nor the count target) can leak into a
    sub-model's features. Requires the raw-derived frame: the processed
    parquet drops casual/registered, so build features on ``load_raw_train``.
    """
    target = cfg["target"]
    datetime_col = cfg["datetime_col"]
    missing = [c for c in (*DUAL_TARGET_COLUMNS, target) if c not in df.columns]
    if missing:
        raise ValueError(
            f"dual_target_split requires {missing}; build features on the raw "
            "train frame (the processed parquet drops casual/registered)."
        )
    drop = [c for c in (target, *DUAL_TARGET_COLUMNS, datetime_col) if c in df.columns]
    X = df.drop(columns=drop)
    return (
        X,
        df["casual"].to_numpy(float),
        df["registered"].to_numpy(float),
        df[target].to_numpy(float),
        df[datetime_col],
    )


def fit_and_predict_dual_target(
    name: str,
    cfg: dict[str, Any],
    params: dict[str, Any],
    X_train: pd.DataFrame,
    casual_train: np.ndarray,
    registered_train: np.ndarray,
    X_eval: pd.DataFrame,
) -> np.ndarray:
    """Fit a casual and a registered model on the train rows and return the
    summed, non-negative count prediction for ``X_eval``.

    Both sub-models come from ``get_model`` (log1p target, non-negative
    inverse); their inverted predictions are summed on the original scale.
    Fresh models are built per call, so this is safe to use inside a CV loop.
    """
    casual_model = get_model(name, cfg, params).fit(X_train, casual_train)
    registered_model = get_model(name, cfg, params).fit(X_train, registered_train)
    return np.asarray(casual_model.predict(X_eval), dtype=float) + np.asarray(
        registered_model.predict(X_eval), dtype=float
    )


def evaluate_dual_target(
    name: str, cfg: dict[str, Any], params: dict[str, Any], df: pd.DataFrame
) -> dict[str, Any]:
    """Dual-target metrics on both validation views, fold-safe.

    Returns ``{"cv": {...}, "holdout": {...}}`` with the four-metric report on
    each. CV uses ``TimeSeriesSplit`` on datetime-sorted rows; for each fold,
    fresh casual/registered models are fit on the fold's train rows only,
    predict the fold's validation rows, and the summed count-scale prediction
    is scored against true count. The holdout fits on days 1-15 and scores on
    days 16-19, matching the direct-count path.
    """
    X, casual, registered, count, dt = dual_target_split(df, cfg)

    # CV (fold-safe), sorted by datetime exactly like train.fit_and_cv.
    n_splits = int(cfg["cv"]["n_splits"])
    gap = int(cfg["cv"].get("gap", 0))
    order = np.argsort(np.asarray(dt.values))
    X_sorted = X.iloc[order].reset_index(drop=True)
    casual_s, registered_s, count_s = casual[order], registered[order], count[order]

    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    fold_metrics: list[dict[str, float]] = []
    for train_idx, val_idx in splitter.split(X_sorted):
        pred = fit_and_predict_dual_target(
            name, cfg, params,
            X_sorted.iloc[train_idx], casual_s[train_idx], registered_s[train_idx],
            X_sorted.iloc[val_idx],
        )
        fold_metrics.append(report(count_s[val_idx], pred))
    keys = list(fold_metrics[0])
    cv = {
        "n_splits": n_splits,
        "mean": {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys},
        "std": {k: float(np.std([m[k] for m in fold_metrics])) for k in keys},
        "folds": fold_metrics,
    }

    # Day-of-month holdout: fit on days 1-15, score on days 16-19.
    train_idx, holdout_idx = day_of_month_holdout_split(dt)
    pred = fit_and_predict_dual_target(
        name, cfg, params,
        X.iloc[train_idx], casual[train_idx], registered[train_idx], X.iloc[holdout_idx],
    )
    holdout = {
        "metrics": report(count[holdout_idx], pred),
        "n_train": int(len(train_idx)),
        "n_holdout": int(len(holdout_idx)),
    }
    return {"cv": cv, "holdout": holdout}
