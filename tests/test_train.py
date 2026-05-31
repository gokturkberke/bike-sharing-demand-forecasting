"""Contracts for fit_and_cv."""

from pathlib import Path

import numpy as np
import pytest

from bike_sharing.config import load_config
from bike_sharing.data import load_raw_train
from bike_sharing.features import build_features
from bike_sharing.models import get_model
from bike_sharing.preprocessing import drop_leakage_columns
from bike_sharing.train import (
    evaluate_holdout,
    fit_and_cv,
    kaggle_like_holdout_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def featured(cfg):
    df = drop_leakage_columns(build_features(load_raw_train(cfg), cfg), cfg)
    y = df["count"].to_numpy(dtype=float)
    datetime = df[cfg["datetime_col"]]
    X = df.drop(columns=["count", cfg["datetime_col"]])
    return X, y, datetime


def test_fit_and_cv_returns_summary_structure(cfg, featured):
    X, y, dt = featured
    out = fit_and_cv(get_model("mean_baseline", cfg), X, y, dt, cfg)
    assert out["n_splits"] == cfg["cv"]["n_splits"]
    assert set(out["mean"]) == {"rmsle", "rmse", "mae", "r2"}
    assert set(out["std"]) == {"rmsle", "rmse", "mae", "r2"}
    assert len(out["folds"]) == cfg["cv"]["n_splits"]


def test_ridge_beats_mean_baseline_on_rmsle(cfg, featured):
    X, y, dt = featured
    mean_out = fit_and_cv(get_model("mean_baseline", cfg), X, y, dt, cfg)
    ridge_out = fit_and_cv(get_model("ridge", cfg), X, y, dt, cfg)
    assert ridge_out["mean"]["rmsle"] < mean_out["mean"]["rmsle"]


def test_hourly_mean_beats_global_mean(cfg, featured):
    X, y, dt = featured
    global_out = fit_and_cv(get_model("mean_baseline", cfg), X, y, dt, cfg)
    hourly_out = fit_and_cv(
        get_model("hourly_mean_baseline", cfg), X, y, dt, cfg
    )
    assert hourly_out["mean"]["rmsle"] < global_out["mean"]["rmsle"]


def test_kaggle_like_holdout_split_partitions_by_day(cfg, featured):
    from bike_sharing.train import KAGGLE_HOLDOUT_DAY_THRESHOLD

    _, _, dt = featured
    train_idx, holdout_idx = kaggle_like_holdout_split(dt)
    # No overlap, full coverage.
    assert set(train_idx).isdisjoint(holdout_idx)
    assert len(train_idx) + len(holdout_idx) == len(dt)
    # Train holds the earlier day-of-month values, holdout the later ones.
    assert dt.iloc[train_idx].dt.day.max() < KAGGLE_HOLDOUT_DAY_THRESHOLD
    assert dt.iloc[holdout_idx].dt.day.min() >= KAGGLE_HOLDOUT_DAY_THRESHOLD


def test_evaluate_holdout_structure_and_counts(cfg, featured):
    X, y, dt = featured
    out = evaluate_holdout(get_model("hourly_mean_baseline", cfg), X, y, dt, cfg)
    assert set(out["metrics"]) == {"rmsle", "rmse", "mae", "r2"}
    assert out["n_train"] + out["n_holdout"] == len(dt)
    assert out["n_train"] > 0 and out["n_holdout"] > 0


def test_ridge_holdout_predictions_are_non_negative(cfg, featured):
    # The clipped inverse must hold on a genuine out-of-sample split.
    X, y, dt = featured
    train_idx, holdout_idx = kaggle_like_holdout_split(dt)
    model = get_model("ridge", cfg).fit(X.iloc[train_idx], y[train_idx])
    preds = model.predict(X.iloc[holdout_idx])
    assert (preds >= 0).all()
