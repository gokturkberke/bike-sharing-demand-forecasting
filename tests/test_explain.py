"""Contracts for permutation-importance explainability.

Uses a small synthetic dataset and a fast RandomForest so the test does not
depend on xgboost being installed. A single feature drives the target and a
second is pure noise; the helper must rank the driver above the noise and
return a sorted, well-formed importance table.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from bike_sharing.explain import rmsle_permutation_importance


def _driver_noise_data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    driver = rng.uniform(0, 20, size=n)
    noise = rng.uniform(0, 20, size=n)
    X = pd.DataFrame({"driver": driver, "noise": noise})
    # Target depends only on the driver and stays positive, so log1p/RMSLE
    # is well defined.
    y = 10.0 + 5.0 * driver
    return X, y


def test_returns_sorted_importance_frame():
    X, y = _driver_noise_data()
    model = RandomForestRegressor(n_estimators=25, random_state=0).fit(X, y)
    imp = rmsle_permutation_importance(model, X, y, seed=0, n_repeats=5)

    assert list(imp.columns) == ["feature", "importance_mean", "importance_std"]
    assert len(imp) == X.shape[1]
    assert set(imp["feature"]) == {"driver", "noise"}
    # Sorted by importance_mean, descending.
    assert imp["importance_mean"].is_monotonic_decreasing
    assert np.isfinite(imp[["importance_mean", "importance_std"]].to_numpy()).all()


def test_relevant_feature_outranks_noise():
    X, y = _driver_noise_data()
    model = RandomForestRegressor(n_estimators=25, random_state=0).fit(X, y)
    imp = rmsle_permutation_importance(
        model, X, y, seed=0, n_repeats=5
    ).set_index("feature")

    # The driver must score strictly higher than pure noise and be positive.
    assert imp.loc["driver", "importance_mean"] > imp.loc["noise", "importance_mean"]
    assert imp.loc["driver", "importance_mean"] > 0
