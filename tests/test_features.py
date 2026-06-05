"""Contracts for feature engineering."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bike_sharing.config import load_config
from bike_sharing.data import load_raw_test, load_raw_train
from bike_sharing.features import (
    ADDED_FEATURE_COLUMNS,
    CANDIDATE_FEATURE_COLUMNS,
    CANDIDATE_NUMERIC_COLUMNS,
    CANDIDATE_PASSTHROUGH_COLUMNS,
    CYCLIC_FEATURE_COLUMNS,
    INTERACTION_FEATURE_COLUMNS,
    TIME_FEATURE_COLUMNS,
    build_candidate_features,
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


def test_interaction_columns_within_unit_circle(featured_train):
    # Workingday-gated cyclic terms are cyclic value * {0,1}, so they stay
    # within [-1, 1] and are zero on non-working days.
    for col in INTERACTION_FEATURE_COLUMNS:
        values = featured_train[col].to_numpy()
        assert values.min() >= -1.0
        assert values.max() <= 1.0
    workingday = featured_train["workingday"].to_numpy()
    assert (featured_train["hour_sin_workday"].to_numpy()[workingday == 0] == 0).all()


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


def test_day_is_not_a_feature(featured_train, cfg):
    # The Kaggle split puts days 1-19 in train and 20-31 in test, so
    # day-of-month is out-of-distribution at test time. The feature
    # pipeline must not surface it.
    assert "day" not in featured_train.columns
    assert "day" not in ADDED_FEATURE_COLUMNS


def test_train_and_test_predictor_schemas_match(cfg):
    # After feature engineering and leakage removal, the predictor
    # column list on train must equal the predictor column list on test.
    # This contract guards against regressions like the day-feature bug.
    train_processed = drop_leakage_columns(
        build_features(load_raw_train(cfg), cfg), cfg
    )
    test_processed = build_features(load_raw_test(cfg), cfg)
    train_predictors = train_processed.drop(columns=["count", cfg["datetime_col"]])
    test_predictors = test_processed.drop(columns=[cfg["datetime_col"]])
    assert list(train_predictors.columns) == list(test_predictors.columns)
    assert "casual" not in train_predictors.columns
    assert "registered" not in train_predictors.columns
    assert "day" not in train_predictors.columns


@pytest.fixture(scope="module")
def candidate_train(cfg) -> pd.DataFrame:
    return build_candidate_features(load_raw_train(cfg), cfg)


def test_candidate_features_do_not_leak_into_production(featured_train):
    # The experimental candidates must not be part of the production set.
    for col in CANDIDATE_FEATURE_COLUMNS:
        assert col not in ADDED_FEATURE_COLUMNS
        assert col not in featured_train.columns


def test_candidate_features_add_expected_columns(candidate_train):
    # Production features survive, candidates are added on top.
    for col in ADDED_FEATURE_COLUMNS:
        assert col in candidate_train.columns
    for col in CANDIDATE_FEATURE_COLUMNS:
        assert col in candidate_train.columns


def test_candidate_binaries_are_zero_one(candidate_train):
    binaries = [
        "is_morning_peak", "is_evening_peak", "is_rush_hour", "is_2012", "bad_weather",
    ]
    for col in binaries:
        assert candidate_train[col].isin([0, 1]).all()


def test_candidate_features_no_nan(candidate_train):
    assert candidate_train[list(CANDIDATE_FEATURE_COLUMNS)].isna().sum().sum() == 0


def test_candidate_features_train_test_parity(cfg):
    train_cand = build_candidate_features(load_raw_train(cfg), cfg)
    test_cand = build_candidate_features(load_raw_test(cfg), cfg)
    # Every candidate column must exist on both sets (computable at inference).
    for col in CANDIDATE_FEATURE_COLUMNS:
        assert col in train_cand.columns
        assert col in test_cand.columns
    # The numeric/passthrough split must cover the full candidate set exactly.
    assert set(CANDIDATE_NUMERIC_COLUMNS) | set(CANDIDATE_PASSTHROUGH_COLUMNS) == set(
        CANDIDATE_FEATURE_COLUMNS
    )


def test_comfort_index_finite_varies_and_experiment_only(candidate_train):
    # The Humidex comfort index must be a finite, varying candidate numeric -
    # and must not leak into the production feature set.
    vals = candidate_train["comfort_index"].to_numpy()
    assert np.isfinite(vals).all()
    assert np.ptp(vals) > 0  # not a constant column
    assert "comfort_index" in CANDIDATE_NUMERIC_COLUMNS
    assert "comfort_index" not in ADDED_FEATURE_COLUMNS


def test_cyclic_encoding_continuity():
    # Hour 23 should be adjacent to hour 0 in the sin/cos space; this is
    # the whole reason for the cyclic encoding.
    cfg = {"datetime_col": "datetime"}
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2011-01-01 00:00:00", "2011-01-01 23:00:00"]),
            "workingday": [1, 1],
        }
    )
    out = build_features(df, cfg)
    hour_0 = (out.loc[0, "hour_sin"], out.loc[0, "hour_cos"])
    hour_23 = (out.loc[1, "hour_sin"], out.loc[1, "hour_cos"])
    # Adjacent on the unit circle => Euclidean distance < the distance
    # between hour 0 and hour 12 (the antipodal point).
    adj = np.hypot(hour_0[0] - hour_23[0], hour_0[1] - hour_23[1])
    far = np.hypot(hour_0[0] - 0.0, hour_0[1] - (-1.0))
    assert adj < far
