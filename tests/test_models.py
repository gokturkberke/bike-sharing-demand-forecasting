"""Contracts for the model factory and baselines."""

import numpy as np
import pandas as pd
import pytest

from bike_sharing.models import (
    HourlyMeanBaseline,
    MeanBaseline,
    get_model,
)


@pytest.fixture
def cfg() -> dict:
    return {"seed": 42, "cv": {"n_splits": 5}}


def test_mean_baseline_predicts_training_mean(cfg):
    model = MeanBaseline().fit(pd.DataFrame({"x": [0, 1, 2]}), [10, 20, 30])
    preds = model.predict(pd.DataFrame({"x": [0, 0, 0, 0]}))
    assert np.allclose(preds, 20.0)
    assert len(preds) == 4


def test_hourly_mean_baseline_learns_per_hour(cfg):
    X = pd.DataFrame({"hour": [0, 0, 1, 1, 2]})
    y = np.array([10.0, 20.0, 100.0, 200.0, 50.0])
    model = HourlyMeanBaseline().fit(X, y)
    preds = model.predict(pd.DataFrame({"hour": [0, 1, 2]}))
    assert np.allclose(preds, [15.0, 150.0, 50.0])


def test_hourly_mean_baseline_falls_back_for_unseen_hour():
    X = pd.DataFrame({"hour": [0, 0, 1, 1]})
    y = np.array([10.0, 20.0, 100.0, 200.0])
    model = HourlyMeanBaseline().fit(X, y)
    # Hour 5 absent from training -> global mean (10+20+100+200)/4 = 82.5.
    preds = model.predict(pd.DataFrame({"hour": [5]}))
    assert preds[0] == pytest.approx(82.5)


def test_hourly_mean_baseline_requires_hour_column():
    with pytest.raises(ValueError, match="hour"):
        HourlyMeanBaseline().fit(pd.DataFrame({"x": [1]}), [1.0])


def test_factory_known_names_return_estimators(cfg):
    for name in ("mean_baseline", "hourly_mean_baseline", "ridge"):
        model = get_model(name, cfg)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")


def test_factory_rejects_unknown_name(cfg):
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("xgboost", cfg)


def test_ridge_predicts_in_original_scale(cfg):
    # The TransformedTargetRegressor must invert the log1p, so predictions
    # come back on the original count scale (positive, large numbers).
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    y = np.abs(rng.normal(loc=100, scale=30, size=200))
    model = get_model("ridge", cfg).fit(X, y)
    preds = model.predict(X)
    # Range sanity: should overlap the training target range, not log space.
    assert preds.mean() > 10
