"""Contracts for the model factory and baselines."""

import numpy as np
import pandas as pd
import pytest

from bike_sharing.models import (
    HourlyMeanBaseline,
    MeanBaseline,
    get_model,
)
from bike_sharing.preprocessing import from_log1p


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
    # The Ridge ColumnTransformer expects the project's feature schema.
    # Build the minimum frame it consumes; predictions should come back
    # in the original count scale (positive, large), not log space.
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "temp": rng.uniform(0, 40, size=n),
            "atemp": rng.uniform(0, 45, size=n),
            "humidity": rng.uniform(0, 100, size=n),
            "windspeed": rng.uniform(0, 50, size=n),
            "holiday": rng.integers(0, 2, size=n),
            "workingday": rng.integers(0, 2, size=n),
            "is_weekend": rng.integers(0, 2, size=n),
            "hour_sin": rng.uniform(-1, 1, size=n),
            "hour_cos": rng.uniform(-1, 1, size=n),
            "month_sin": rng.uniform(-1, 1, size=n),
            "month_cos": rng.uniform(-1, 1, size=n),
            "season": rng.integers(1, 5, size=n),
            "weather": rng.integers(1, 4, size=n),
        }
    )
    y = np.abs(rng.normal(loc=100, scale=30, size=n))
    model = get_model("ridge", cfg).fit(X, y)
    preds = model.predict(X)
    assert (preds >= 0).all()
    assert preds.mean() > 10


def test_ridge_inverse_clips_negative_predictions(cfg):
    # The Ridge target inversion must be the project's clipped contract
    # (from_log1p), not bare expm1, so a submission can never carry
    # negative demand even if the linear model emits a negative log value.
    model = get_model("ridge", cfg)
    assert model.inverse_func is from_log1p
