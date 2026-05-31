"""Contracts for fit_and_cv."""

from pathlib import Path

import numpy as np
import pytest

from bike_sharing.config import load_config
from bike_sharing.data import load_raw_train
from bike_sharing.features import build_features
from bike_sharing.models import get_model
from bike_sharing.preprocessing import drop_leakage_columns
from bike_sharing.train import fit_and_cv

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
