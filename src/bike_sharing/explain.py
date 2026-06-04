"""Model-agnostic explainability for Bike Sharing Demand models.

Permutation importance complements the impurity-based feature importances
(notebook 04, figure 13). It is model-agnostic and measures the actual
increase in held-out error when a feature's values are shuffled, so it
reduces the impurity bias toward continuous/high-cardinality columns. It is
not a complete fix: importance among strongly correlated inputs (e.g.
temp/atemp, or the hour family) can still be distributed in ways that need
careful reading, so it is the stronger view, not the final word.

Importance is scored with the project's count-scale RMSLE - the same metric
the models are judged on - not a log-space loss or the default R2.
"""

from typing import Any

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer

from bike_sharing.evaluate import rmsle


def rmsle_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: Any,
    *,
    seed: int = 42,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Permutation importance scored with count-scale RMSLE.

    ``model`` must already be fitted, and its ``predict`` must return
    original-scale counts (the project's ``TransformedTargetRegressor``
    does this). ``X``/``y`` are the evaluation rows: for the report these
    are the day-of-month holdout (days 16-19), with the model fit only on
    days 1-15, so the importances are a genuine out-of-sample view. The
    model is never refit here; only ``predict`` is called on shuffled
    copies of ``X``.

    Returns a DataFrame with one row per column of ``X`` -
    ``feature``, ``importance_mean``, ``importance_std`` - sorted by
    ``importance_mean`` descending. A feature's importance is the average
    increase in RMSLE when its values are permuted; larger means the model
    relies on it more.
    """
    # greater_is_better=False negates RMSLE, so permutation_importance -
    # which reports (baseline - permuted) in the scorer's space - yields a
    # positive importance equal to the RMSLE increase a feature's shuffle
    # causes.
    scorer = make_scorer(rmsle, greater_is_better=False)
    result = permutation_importance(
        model,
        X,
        y,
        scoring=scorer,
        n_repeats=n_repeats,
        random_state=seed,
    )
    return (
        pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
