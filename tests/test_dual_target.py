"""Contracts for the optional dual-target (casual + registered) experiment.

The core property is the stretch-rule leakage guard: neither sub-target (nor
count) may be a feature, and the summed prediction is the two component
models' non-negative predictions added on the original count scale.
"""

from pathlib import Path

import numpy as np
import pytest

from bike_sharing.config import load_config
from bike_sharing.data import load_raw_train
from bike_sharing.dual_target import (
    dual_target_split,
    evaluate_dual_target,
    fit_and_predict_dual_target,
)
from bike_sharing.features import build_features
from bike_sharing.models import get_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def featured_raw(cfg):
    # Raw-derived: the processed parquet drops casual/registered, which the
    # dual-target experiment needs as targets.
    return build_features(load_raw_train(cfg), cfg)


def test_split_excludes_targets_from_features(featured_raw, cfg):
    X, casual, registered, count, dt = dual_target_split(featured_raw, cfg)
    for col in ("count", "casual", "registered", cfg["datetime_col"]):
        assert col not in X.columns
    assert len(casual) == len(registered) == len(count) == len(X)
    # In the raw data the components sum to count.
    assert np.allclose(casual + registered, count)


def test_fit_and_predict_is_nonnegative_component_sum(featured_raw, cfg):
    X, casual, registered, count, dt = dual_target_split(featured_raw, cfg)
    X_tr, casual_tr, reg_tr = X.iloc[:500], casual[:500], registered[:500]
    X_eval = X.iloc[500:600]
    params = {"n_estimators": 20, "max_depth": 3}
    pred = fit_and_predict_dual_target(
        "gradient_boosting", cfg, params, X_tr, casual_tr, reg_tr, X_eval
    )
    assert pred.shape == (100,)
    assert (pred >= 0).all()
    # Exactly the sum of the two component models' predictions.
    casual_model = get_model("gradient_boosting", cfg, params).fit(X_tr, casual_tr)
    reg_model = get_model("gradient_boosting", cfg, params).fit(X_tr, reg_tr)
    assert np.allclose(pred, casual_model.predict(X_eval) + reg_model.predict(X_eval))


def test_evaluate_dual_target_returns_metric_set(featured_raw, cfg):
    # Small subset + tiny model so the CV loop stays fast; we only check shape.
    subset = featured_raw.iloc[:3000]
    out = evaluate_dual_target("random_forest", cfg, {"n_estimators": 10}, subset)
    assert set(out) == {"cv", "holdout"}
    assert set(out["cv"]["mean"]) == {"rmsle", "rmse", "mae", "r2"}
    assert set(out["holdout"]["metrics"]) == {"rmsle", "rmse", "mae", "r2"}
    assert out["holdout"]["n_train"] > 0 and out["holdout"]["n_holdout"] > 0
