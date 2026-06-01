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
    for name in (
        "mean_baseline",
        "hourly_mean_baseline",
        "ridge",
        "random_forest",
        "gradient_boosting",
    ):
        model = get_model(name, cfg)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")


def test_xgboost_factory_returns_estimator(cfg):
    # xgboost is an optional dependency; skip cleanly if it cannot be used.
    # Not just "not installed": its native runtime can fail to load (e.g.
    # missing OpenMP/libomp on macOS), which raises XGBoostError - NOT an
    # ImportError - so pytest.importorskip would not catch it.
    try:
        import xgboost  # noqa: F401
    except Exception as exc:
        pytest.skip(f"xgboost unavailable: {exc}")
    model = get_model("xgboost", cfg)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert model.inverse_func is from_log1p


def test_factory_rejects_unknown_name(cfg):
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("lightgbm", cfg)


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
            "hour_sin2": rng.uniform(-1, 1, size=n),
            "hour_cos2": rng.uniform(-1, 1, size=n),
            "hour_sin_workday": rng.uniform(-1, 1, size=n),
            "hour_cos_workday": rng.uniform(-1, 1, size=n),
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
    # (from_log1p), not bare expm1, so the prediction artifact can never
    # carry negative demand even if the linear model emits a negative
    # log value.
    model = get_model("ridge", cfg)
    assert model.inverse_func is from_log1p


def _tree_feature_frame(n=300, seed=0):
    # The full feature set the tree models consume (raw ordinals kept).
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "season": rng.integers(1, 5, size=n),
            "holiday": rng.integers(0, 2, size=n),
            "workingday": rng.integers(0, 2, size=n),
            "weather": rng.integers(1, 4, size=n),
            "temp": rng.uniform(0, 40, size=n),
            "atemp": rng.uniform(0, 45, size=n),
            "humidity": rng.uniform(0, 100, size=n),
            "windspeed": rng.uniform(0, 50, size=n),
            "hour": rng.integers(0, 24, size=n),
            "dayofweek": rng.integers(0, 7, size=n),
            "month": rng.integers(1, 13, size=n),
            "year": rng.integers(2011, 2013, size=n),
            "is_weekend": rng.integers(0, 2, size=n),
            "hour_sin": rng.uniform(-1, 1, size=n),
            "hour_cos": rng.uniform(-1, 1, size=n),
            "month_sin": rng.uniform(-1, 1, size=n),
            "month_cos": rng.uniform(-1, 1, size=n),
        }
    )
    y = np.abs(rng.normal(loc=100, scale=40, size=n))
    return X, y


@pytest.mark.parametrize("name", ["random_forest", "gradient_boosting"])
def test_tree_models_fit_predict_non_negative(cfg, name):
    X, y = _tree_feature_frame()
    # Small, fast params for the test; real defaults live in models.yaml.
    params = {"n_estimators": 20} if name == "random_forest" else {"n_estimators": 20}
    model = get_model(name, cfg, params).fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)
    assert (preds >= 0).all()


@pytest.mark.parametrize("name", ["random_forest", "gradient_boosting"])
def test_tree_models_inverse_clips(cfg, name):
    # Trees also use the clipped from_log1p inverse for the non-negativity
    # contract.
    assert get_model(name, cfg).inverse_func is from_log1p


def test_params_override_reaches_estimator(cfg):
    # A hyperparameter passed via params must land on the underlying model.
    model = get_model("random_forest", cfg, {"n_estimators": 7})
    assert model.regressor.n_estimators == 7
