"""Contracts for feature engineering."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bike_sharing.config import load_config
from bike_sharing.data import load_raw_test, load_raw_train
from bike_sharing.features import (
    ADDED_FEATURE_COLUMNS,
    CYCLIC_FEATURE_COLUMNS,
    TIME_FEATURE_COLUMNS,
    build_features,
)
from bike_sharing.preprocessing import drop_leakage_columns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def featured_train(cfg) -> pd.DataFrame:
    return build_features(load_raw_train(cfg), cfg)


def test_build_features_adds_expected_columns(featured_train):
    for col in ADDED_FEATURE_COLUMNS:
        assert col in featured_train.columns


def test_build_features_preserves_datetime(featured_train, cfg):
    assert cfg["datetime_col"] in featured_train.columns
    assert pd.api.types.is_datetime64_any_dtype(featured_train[cfg["datetime_col"]])


def test_build_features_preserves_target_and_leakage_cols(featured_train):
    # Leakage exclusion is preprocessing's job, not features'.
    for col in ("count", "casual", "registered"):
        assert col in featured_train.columns


def test_build_features_does_not_introduce_nan(featured_train):
    assert featured_train.isna().sum().sum() == 0


def test_cyclic_columns_within_unit_circle(featured_train):
    for col in CYCLIC_FEATURE_COLUMNS:
        values = featured_train[col].to_numpy()
        assert values.min() >= -1.0
        assert values.max() <= 1.0
        # The encoding must vary across the dataset — guard against a
        # constant column that would silently coerce to a useless feature.
        assert np.ptp(values) > 0.5


def test_time_columns_have_expected_ranges(featured_train):
    assert featured_train["hour"].between(0, 23).all()
    assert featured_train["dayofweek"].between(0, 6).all()
    assert featured_train["month"].between(1, 12).all()
    assert featured_train["year"].isin([2011, 2012]).all()
    assert featured_train["is_weekend"].isin([0, 1]).all()


def test_features_compose_with_drop_leakage(featured_train, cfg):
    safe = drop_leakage_columns(featured_train, cfg)
    assert "casual" not in safe.columns
    assert "registered" not in safe.columns
    # The target stays — it gets stripped from X at fit time, not here.
    assert "count" in safe.columns
    # The new features survive.
    for col in ADDED_FEATURE_COLUMNS:
        assert col in safe.columns


def test_build_features_on_test_set(cfg):
    df = build_features(load_raw_test(cfg), cfg)
    assert df.shape[0] == 6493
    for col in ADDED_FEATURE_COLUMNS:
        assert col in df.columns
    assert cfg["datetime_col"] in df.columns
    assert "count" not in df.columns


def test_cyclic_encoding_continuity():
    # Hour 23 should be adjacent to hour 0 in the sin/cos space; this is
    # the whole reason for the cyclic encoding.
    cfg = {"datetime_col": "datetime"}
    df = pd.DataFrame(
        {"datetime": pd.to_datetime(["2011-01-01 00:00:00", "2011-01-01 23:00:00"])}
    )
    out = build_features(df, cfg)
    hour_0 = (out.loc[0, "hour_sin"], out.loc[0, "hour_cos"])
    hour_23 = (out.loc[1, "hour_sin"], out.loc[1, "hour_cos"])
    # Adjacent on the unit circle => Euclidean distance < the distance
    # between hour 0 and hour 12 (the antipodal point).
    adj = np.hypot(hour_0[0] - hour_23[0], hour_0[1] - hour_23[1])
    far = np.hypot(hour_0[0] - 0.0, hour_0[1] - (-1.0))
    assert adj < far
