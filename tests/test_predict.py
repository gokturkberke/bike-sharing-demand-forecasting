"""Contracts for the test-set prediction artifact."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bike_sharing.config import load_config
from bike_sharing.data import load_raw_test, load_raw_train
from bike_sharing.features import build_features
from bike_sharing.models import get_model
from bike_sharing.predict import (
    SUBMISSION_COLUMNS,
    make_prediction_frame,
    write_prediction_artifact,
)
from bike_sharing.preprocessing import drop_leakage_columns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

EXPECTED_TEST_ROWS = 6493


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def test_features(cfg) -> pd.DataFrame:
    return build_features(load_raw_test(cfg), cfg)


@pytest.fixture(scope="module")
def fitted_model(cfg):
    # A cheap model is enough to exercise the artifact schema; the schema
    # does not depend on which estimator produced the predictions.
    train = drop_leakage_columns(build_features(load_raw_train(cfg), cfg), cfg)
    y = train["count"].to_numpy(dtype=float)
    X = train.drop(columns=["count", cfg["datetime_col"]])
    return get_model("mean_baseline", cfg).fit(X, y)


def test_prediction_frame_schema_and_rows(cfg, fitted_model, test_features):
    frame = make_prediction_frame(fitted_model, test_features, cfg)
    assert list(frame.columns) == SUBMISSION_COLUMNS
    assert list(frame.columns) == ["datetime", "count"]
    assert len(frame) == EXPECTED_TEST_ROWS


def test_prediction_frame_non_negative_no_nan(cfg, fitted_model, test_features):
    frame = make_prediction_frame(fitted_model, test_features, cfg)
    assert frame["count"].notna().all()
    assert (frame["count"] >= 0).all()


def test_prediction_frame_preserves_datetime_order(cfg, fitted_model, test_features):
    frame = make_prediction_frame(fitted_model, test_features, cfg)
    assert frame["datetime"].equals(test_features["datetime"])


def test_write_prediction_artifact_roundtrip(cfg, fitted_model, test_features, tmp_path):
    frame = make_prediction_frame(fitted_model, test_features, cfg)
    out = write_prediction_artifact(frame, tmp_path / "sub.csv")
    reloaded = pd.read_csv(out)
    assert list(reloaded.columns) == ["datetime", "count"]
    assert len(reloaded) == EXPECTED_TEST_ROWS


def test_write_prediction_artifact_rejects_bad_schema(tmp_path):
    bad = pd.DataFrame({"datetime": ["2011-01-20 00:00:00"], "demand": [1.0]})
    with pytest.raises(ValueError, match="datetime"):
        write_prediction_artifact(bad, tmp_path / "bad.csv")
